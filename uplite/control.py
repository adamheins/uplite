from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
    ACADOS_INFTY,
)
import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

from .transport import adjoint, contact_jacobian


def _make_model(robot, nf=0):
    # state
    q = ca.SX.sym("q", robot.model.nq)
    v = ca.SX.sym("v", robot.model.nv)
    a = ca.SX.sym("a", robot.model.nv)

    x = ca.vertcat(q, v, a)
    xdot = ca.SX.sym("xdot", x.size1())

    # input
    j = ca.SX.sym("j", robot.model.nv)
    f = ca.SX.sym("f", nf)
    u = ca.vertcat(j, f)

    # dynamics
    f_expl = ca.vertcat(v, a, j)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = "robot"
    return model


def _split_x(x, nq, nv):
    q = x[:nq]
    v = x[nq : nq + nv]
    a = x[nq + nv :]
    return q, v, a


def _make_solver(
    robot, model, horizon, total_steps, q0, rd, params=None, contacts=None
):
    ocp = AcadosOcp()
    ocp.model = model
    ocp.name = "robot_ocp"
    ocp.json_file = "acados/ocp.json"
    ocp.code_export_directory = "acados/c_generated_code_ocp"

    nq = robot.model.nq
    nv = robot.model.nv
    nx = model.x.rows()

    na = nv
    nu = model.u.rows()

    if contacts is None:
        contacts = []
    nf = len(contacts)

    # set prediction horizon
    ocp.solver_options.N_horizon = total_steps
    ocp.solver_options.tf = horizon

    # initial state and input
    v0 = np.zeros(nv)
    a0 = np.zeros(nv)
    x0 = np.concatenate((q0, v0, a0))
    u0 = np.zeros(nu)

    # cost matrices
    Qr = np.eye(3)
    Qq = 0 * np.eye(nq)
    Qv = 0.1 * np.eye(nv)
    Qa = 0.01 * np.eye(nv)
    Q = ca.diagcat(Qr, Qq, Qv, Qa).full()
    R = 0.001 * np.eye(nu)

    # casadi forward kinematics
    q, v, a = _split_x(model.x, nq, nv)
    robot.forward(q=q, v=v, a=a)
    r, C = robot.pose()

    # path cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(r, model.x, model.u)
    ocp.cost.yref = np.concatenate((rd, x0, u0))
    ocp.cost.W = ca.diagcat(Q, R).full()

    # terminal cost
    # ocp.cost.cost_type_e = "NONLINEAR_LS"
    # ocp.model.cost_y_expr_e = ca.vertcat(r, model.x)
    # ocp.cost.yref_e = np.concatenate((rd, x0))
    # ocp.cost.W_e = Q

    # input and force limits
    j_max = 100
    ocp.constraints.lbu = np.concatenate((-j_max * np.ones(nv), np.zeros(nf)))
    ocp.constraints.ubu = np.concatenate(
        (j_max * np.ones(nv), ACADOS_INFTY * np.ones(nf))
    )
    ocp.constraints.idxbu = np.arange(nu)

    # state limits
    # ocp.constraints.lbx = np.array([-Y_MAX, -Z_MAX, -V_MAX, -V_MAX])
    # ocp.constraints.ubx = np.array([Y_MAX, Z_MAX, V_MAX, V_MAX])
    # ocp.constraints.idxbx = np.array([0, 1, 3, 4])

    # aligned constraint
    # a = robot.classical_acceleration()[0]
    # z = np.array([0, 0, 1])
    # g = 9.81 * z
    # con = ca.cross(z, a + C @ g)
    # ocp.model.con_h_expr = con
    # ocp.constraints.lh = np.zeros(3)
    # ocp.constraints.uh = np.zeros(3)

    # initial state
    ocp.constraints.x0 = x0

    # rigid body dynamics sticking constraint
    if nf > 0:
        # contact wrench
        wc = np.zeros(6)
        for i in range(nf):
            # assume all normals are vertical
            force = model.u[nv + i] * np.array([0, 0, 1])
            G = contact_jacobian(contacts[i, :]).T
            wc += G @ force

        # spatial gravity in body frame
        g = ca.vertcat(np.zeros(3), C.T @ np.array([0, 0, -9.81]))

        # Newton-Euler equation for rigid body dynamics
        ξ = ca.vertcat(*robot.spatial_velocity())
        dξdt = ca.vertcat(*robot.spatial_acceleration())
        h = params.ne(ξ, dξdt - g) - wc

        ocp.model.con_h_expr = h
        ocp.constraints.lh = np.zeros(6)
        ocp.constraints.uh = np.zeros(6)

    # TODO: terminal constraint - this does not seem to be working
    ocp.model.con_h_expr_e = ca.vertcat(r - rd, model.x[nq:])
    ocp.constraints.lh_e = np.zeros(3 + 2 * nv)
    ocp.constraints.uh_e = np.zeros(3 + 2 * nv)
    # ocp.model.con_h_expr_e = rd - r
    # ocp.constraints.lh_e = np.zeros(3)
    # ocp.constraints.uh_e = np.zeros(3)

    # ns = 6
    # ocp.constraints.idxsh = np.arange(ns)
    # ocp.cost.zl = 1 * np.ones(ns)
    # ocp.cost.Zl = 0 * np.ones(ns)
    # ocp.cost.zu = 1 * np.ones(ns)
    # ocp.cost.Zu = 0 * np.ones(ns)

    # set solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.qp_solver_iter_max = 1000

    return AcadosOcpSolver(ocp)


def _make_integrator(model, dt, int_steps=1):
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = dt  # simulation time
    sim.solver_options.num_steps = int_steps
    sim.code_export_directory = "acados/c_generated_code_sim"
    return AcadosSimSolver(sim, json_file="acados/sim.json")


class Planner:
    def __init__(
        self, robot, horizon, steps_per_second, q0, rd, params=None, contacts=None
    ):
        self.horizon = horizon
        self.total_steps = int(horizon * steps_per_second)
        self.dt = 1.0 / steps_per_second

        self.crobot = robot.casadi()
        nf = 0 if contacts is None else len(contacts)
        self.model = _make_model(self.crobot, nf=nf)
        self.solver = _make_solver(
            robot=self.crobot,
            model=self.model,
            horizon=self.horizon,
            total_steps=self.total_steps,
            q0=q0,
            rd=rd,
            params=params,
            contacts=contacts,
        )

    def solve(self, verbose=False):
        status = self.solver.solve()
        if verbose:
            self.solver.print_statistics()
        return status

    def get_solution_times(self):
        return np.arange(self.total_steps + 1) * self.dt

    def get_solution_states(self):
        return np.array([self.solver.get(i, "x") for i in range(self.total_steps + 1)])

    def get_solution_inputs(self):
        return np.array([self.solver.get(i, "u") for i in range(self.total_steps)])

    def get_solution_spline(self):
        ts = self.get_solution_times()
        xs = self.get_solution_states()
        qs = xs[:, : self.crobot.model.nq]
        return CubicSpline(ts, qs, axis=0)
