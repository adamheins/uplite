import time

from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
    plot_trajectories,
)
from casadi import SX, vertcat, sin, cos, Function, diagcat
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import pybullet as pyb
import pybullet_data
import pyb_utils

import uplite


STEPS_PER_SECOND = 10
HORIZON = 5
TOTAL_STEPS = STEPS_PER_SECOND * HORIZON

SIM_TIMESTEP = 0.01

URDF_PATH = uplite.ASSETS_DIR / "combined.urdf"
ROBOT_HOME = np.array([0, -np.pi / 4, np.pi / 2, -np.pi / 4, np.pi / 2, 0])

# TODO
# * add sticking constraints


def setup_integrator(model, dt, steps=1):
    """Setup acados integrator for given model and time step."""
    sim = AcadosSim()
    sim.model = model

    sim.solver_options.T = dt  # simulation time
    sim.solver_options.num_steps = steps
    sim.code_export_directory = "c_generated_code_sim"
    return AcadosSimSolver(sim)


def get_solution_at_time(t, horizon, steps, solver):
    if t < 0 or t > horizon:
        raise Exception("time out of bounds")
    dt = horizon / steps
    idx = int(t / dt)
    if idx >= steps:
        idx = steps - 1
    x = solver.get(idx, "x")
    u = solver.get(idx, "u")
    return idx * dt, x, u


class CasadiRobotModel:
    def __init__(self, urdf_path, ee_name):
        model = pin.buildModelFromUrdf(urdf_path)
        self.model = cpin.Model(model)
        self.data = self.model.createData()

        assert self.model.existFrame(ee_name)
        self.ee_idx = self.model.getFrameId(ee_name)

    def forward(self, x, u):
        cpin.forwardKinematics(self.model, self.data, x[:6], x[6:], u)
        cpin.updateFramePlacements(self.model, self.data)

    def ee_pose(self):
        oMf = self.data.oMf[self.ee_idx]
        return oMf.translation, oMf.rotation

    # TODO we also need EE velocity and acceleration

    def make_acados_model(self, jerk_input=False, name="robot"):
        nx = self.model.nq + self.model.nv
        q = SX.sym("q", self.model.nq)
        v = SX.sym("v", self.model.nv)
        u = SX.sym("u", self.model.nv)

        x = vertcat(q, v)
        xdot = SX.sym("xdot", x.size1())

        f_expl = vertcat(v, u)
        f_impl = xdot - f_expl

        model = AcadosModel()

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.name = name

        model.x_labels = [f"q_{i}" for i in range(self.model.nq)] + [
            f"v_{i}" for i in range(self.model.nv)
        ]
        model.u_labels = [f"u_{i}" for i in range(self.model.nv)]
        model.t_label = "Time [s]"

        return model


def main():
    robot = CasadiRobotModel(urdf_path=URDF_PATH, ee_name="tool0")
    model = robot.make_acados_model()

    name = "robot_ocp"
    ocp = AcadosOcp()
    ocp.model = model
    ocp.name = name
    ocp.json_file = f"{name}.json"
    ocp.code_export_directory = f"c_generated_code_{name}"

    nq = robot.model.nq
    nv = robot.model.nv
    nx = model.x.rows()
    nu = model.u.rows()

    # set prediction horizon
    ocp.solver_options.N_horizon = TOTAL_STEPS
    ocp.solver_options.tf = HORIZON

    # cost matrices
    # TODO: split up Q
    Q = np.diag(np.concatenate((np.ones(3), 0 * np.ones(6), 0.1 * np.ones(6))))
    R = 0.01 * np.eye(nu)

    rd = np.array([0.5, 0.0, 0.5])  # desired end-effector position
    robot.forward(model.x, model.u)
    ee_pos = robot.ee_pose()[0]

    q0 = ROBOT_HOME
    v0 = np.zeros(nv)
    u0 = np.zeros(nu)
    x0 = np.concatenate((q0, v0))

    # path cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = vertcat(ee_pos, model.x, model.u)
    ocp.cost.yref = np.concatenate((rd, x0, u0))
    ocp.cost.W = diagcat(Q, R).full()

    # terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = vertcat(ee_pos, model.x)
    ocp.cost.yref_e = np.concatenate((rd, x0))
    ocp.cost.W_e = Q

    # input limits
    ocp.constraints.lbu = -10 * np.ones(nu)
    ocp.constraints.ubu = 10 * np.ones(nu)
    ocp.constraints.idxbu = np.arange(nu)

    # state limits
    # ocp.constraints.lbx = np.array([-Y_MAX, -Z_MAX, -V_MAX, -V_MAX])
    # ocp.constraints.ubx = np.array([Y_MAX, Z_MAX, V_MAX, V_MAX])
    # ocp.constraints.idxbx = np.array([0, 1, 3, 4])

    # initial state
    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"  # IRK, ERK
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = 2000
    ocp.solver_options.qp_solver_iter_max = 2000
    ocp.solver_options.qp_tol = 1e-6

    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((TOTAL_STEPS + 1, nx))
    simU = np.zeros((TOTAL_STEPS, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics()  # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f"acados returned status {status}.")

    integrator = setup_integrator(model, SIM_TIMESTEP)

    # simulation
    pyb.connect(pyb.GUI)
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(SIM_TIMESTEP)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    robot_id = pyb.loadURDF(
        URDF_PATH.as_posix(),
        [0, 0, 1],
        useFixedBase=True,
    )
    # TODO: name collision
    robot = pyb_utils.Robot(robot_id, tool_link_name="tool0")
    robot.reset_joint_configuration(q0)

    # get solution
    for i in range(TOTAL_STEPS):
        simX[i, :] = ocp_solver.get(i, "x")
        simU[i, :] = ocp_solver.get(i, "u")
    simX[TOTAL_STEPS, :] = ocp_solver.get(TOTAL_STEPS, "x")

    kp = 10
    xd = x0.copy()

    ts = []
    rs = []

    t = 0
    i = 0
    while t < HORIZON:
        _, _, u = get_solution_at_time(t, HORIZON, TOTAL_STEPS, ocp_solver)
        xd = integrator.simulate(x=xd, u=u)
        qd, vd = xd[:nq], xd[nq:]

        q, v = robot.get_joint_states()
        r = robot.get_link_frame_pose()[0]
        v_cmd = kp * (qd - q) + vd
        robot.command_velocity(v_cmd)

        ts.append(t)
        rs.append(r)

        pyb.stepSimulation()
        i += 1
        t = i * SIM_TIMESTEP
        time.sleep(SIM_TIMESTEP)

    rs = np.array(rs)

    plt.figure()
    plt.plot(ts, rd[0] - rs[:, 0], label="x")
    plt.plot(ts, rd[1] - rs[:, 1], label="y")
    plt.plot(ts, rd[2] - rs[:, 2] + 1, label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("Position error")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
