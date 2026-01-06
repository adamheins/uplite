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


def _make_model(kinematics, transobj=None):
    # state
    q = ca.SX.sym("q", kinematics.model.nq)
    v = ca.SX.sym("v", kinematics.model.nv)
    a = ca.SX.sym("a", kinematics.model.nv)

    x = ca.vertcat(q, v, a)
    xdot = ca.SX.sym("xdot", x.size1())

    # input
    nf = 0 if transobj is None else len(transobj.contacts)
    j = ca.SX.sym("j", kinematics.model.nv)
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
    model.name = "robot_waiter"
    return model


def _split_x(x, nq, nv):
    q = x[:nq]
    v = x[nq : nq + nv]
    a = x[nq + nv :]
    return q, v, a


def _make_solver(kinematics, model, horizon, total_steps, q0, goal, transobj=None):
    ocp = AcadosOcp()
    ocp.model = model
    ocp.name = "robot_waiter_ocp"
    ocp.json_file = "acados/ocp.json"
    ocp.code_export_directory = "acados/c_generated_code_ocp"

    nq = kinematics.model.nq
    nv = kinematics.model.nv
    nx = model.x.rows()
    nu = model.u.rows()
    nf = 0 if transobj is None else len(transobj.contacts)

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
    kinematics.forward(q=q, v=v, a=a)
    r, C = kinematics.pose()

    # path cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(r, model.x, model.u)
    ocp.cost.yref = np.concatenate((goal, x0, u0))
    ocp.cost.W = ca.diagcat(Q, R).full()

    # terminal cost
    # ocp.cost.cost_type_e = "NONLINEAR_LS"
    # ocp.model.cost_y_expr_e = ca.vertcat(r, model.x)
    # ocp.cost.yref_e = np.concatenate((goal, x0))
    # ocp.cost.W_e = Q

    # input and force limits
    # TODO put as constants somewhere
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
    # a = kinematics.classical_acceleration()[0]
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
            G = contact_jacobian(transobj.contacts[i, :]).T
            wc += G @ force

        # spatial gravity in body frame
        g = ca.vertcat(np.zeros(3), C.T @ np.array([0, 0, -9.81]))

        # Newton-Euler equation for rigid body dynamics
        ξ = ca.vertcat(*kinematics.spatial_velocity())
        dξdt = ca.vertcat(*kinematics.spatial_acceleration())
        h = transobj.params.ne(ξ, dξdt - g) - wc

        ocp.model.con_h_expr = h
        ocp.constraints.lh = np.zeros(6)
        ocp.constraints.uh = np.zeros(6)

    # TODO: terminal constraint - this does not seem to be working
    ocp.model.con_h_expr_e = ca.vertcat(r - goal, model.x[nq:])
    ocp.constraints.lh_e = np.zeros(3 + 2 * nv)
    ocp.constraints.uh_e = np.zeros(3 + 2 * nv)
    # ocp.model.con_h_expr_e = goal - r
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
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.qp_solver_iter_max = 1000

    return AcadosOcpSolver(ocp)


class Planner:
    def __init__(
        self, kinematics, horizon, steps_per_second, q0, goal, transobj=None):
        self.horizon = horizon
        self.total_steps = int(horizon * steps_per_second)
        self.dt = 1.0 / steps_per_second

        self.kinematics = kinematics.to_casadi()
        self.model = _make_model(self.kinematics, transobj=transobj)

        # TODO it would be nice to break this out
        self.solver = _make_solver(
            kinematics=self.kinematics,
            model=self.model,
            horizon=self.horizon,
            total_steps=self.total_steps,
            q0=q0,
            goal=goal,
            transobj=transobj,
        )

    def plan(self, verbose=False):
        status = self.solver.solve()
        if verbose:
            self.solver.print_statistics()
        return status

    def replan(self, x, verbose=False):
        pass

    def get_solution_times(self):
        return np.arange(self.total_steps + 1) * self.dt

    def get_solution_states(self):
        return np.array([self.solver.get(i, "x") for i in range(self.total_steps + 1)])

    def get_solution_inputs(self):
        return np.array([self.solver.get(i, "u") for i in range(self.total_steps)])

    def get_solution_spline(self):
        ts = self.get_solution_times()
        xs = self.get_solution_states()
        qs = xs[:, : self.kinematics.model.nq]
        return CubicSpline(ts, qs, axis=0)
