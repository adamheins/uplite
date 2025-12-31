from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
)
from casadi import SX, vertcat, sin, cos, Function, diagcat
import numpy as np


def _make_model(robot, jerk_input=False):
    nx = robot.model.nq + robot.model.nv
    q = SX.sym("q", robot.model.nq)
    v = SX.sym("v", robot.model.nv)
    u = SX.sym("u", robot.model.nv)

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
    model.name = "robot"
    return model


def _make_solver(robot, model, horizon, total_steps, q0, rd):
    ocp = AcadosOcp()
    ocp.model = model
    ocp.name = "robot_ocp"
    ocp.json_file = "acados/ocp.json"
    ocp.code_export_directory = "acados/c_generated_code_ocp"

    nq = robot.model.nq
    nv = robot.model.nv
    nx = model.x.rows()
    nu = model.u.rows()

    # set prediction horizon
    ocp.solver_options.N_horizon = total_steps
    ocp.solver_options.tf = horizon

    # cost matrices
    Qr = np.eye(3)
    Qq = 0 * np.eye(nq)
    Qv = 0.1 * np.eye(nv)
    Q = diagcat(Qr, Qq, Qv).full()
    R = 0.01 * np.eye(nu)

    robot.forward(model.x, model.u)
    r = robot.pose()[0]

    v0 = np.zeros(nv)
    u0 = np.zeros(nu)
    x0 = np.concatenate((q0, v0))

    # desired end-effector position
    # robot.forward(x0, u0)
    # r0 = robot.pose()[0]
    # rd = r0 + TARGET_POSITION

    # path cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = vertcat(r, model.x, model.u)
    ocp.cost.yref = np.concatenate((rd, x0, u0))
    ocp.cost.W = diagcat(Q, R).full()

    # terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = vertcat(r, model.x)
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

    return AcadosOcpSolver(ocp)


def _make_integrator(model, dt, steps=1):
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = dt  # simulation time
    sim.solver_options.num_steps = steps
    sim.code_export_directory = "acados/c_generated_code_sim"
    sim.json_file = "acados/sim.json"
    return AcadosSimSolver(sim)


def _get_input_at_time(t, horizon, steps, solver):
    if t < 0 or t > horizon:
        raise ValueError("time out of bounds")
    dt = horizon / steps
    idx = int(t / dt)
    if idx >= steps:
        idx = steps - 1
    return solver.get(idx, "u")


class Planner:
    def __init__(self, robot, horizon, steps_per_second, q0, rd):
        self.horizon = horizon
        self.total_steps = int(horizon * steps_per_second)

        crobot = robot.casadi()
        self.model = _make_model(crobot)
        self.solver = _make_solver(
            robot=crobot,
            model=self.model,
            horizon=horizon,
            total_steps=self.total_steps,
            q0=q0,
            rd=rd,
        )

    def solve(self, verbose=False):
        status = self.solver.solve()
        if verbose:
            self.solver.print_statistics()
        return status

    def rollout(self, dt, steps=1):
        integrator = _make_integrator(self.model, dt=dt, steps=steps)

        xd = self.solver.get(0, "x")
        xds = []
        us = []

        i = 0
        t = 0
        while t <= self.horizon:
            u = _get_input_at_time(t, self.horizon, self.total_steps, self.solver)
            xd = integrator.simulate(x=xd, u=u)
            xds.append(xd)
            us.append(u)
            i += 1
            t = i * dt
        return np.array(xds), np.array(us)
