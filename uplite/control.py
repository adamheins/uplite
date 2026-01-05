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

from .transport import adjoint, contact_jacobian


def _make_model(robot, nf=0):
    # state
    q = ca.SX.sym("q", robot.model.nq)
    v = ca.SX.sym("v", robot.model.nv)
    x = ca.vertcat(q, v)
    xdot = ca.SX.sym("xdot", x.size1())

    # input
    a = ca.SX.sym("v", robot.model.nv)
    f = ca.SX.sym("f", nf)
    u = ca.vertcat(a, f)

    # dynamics
    f_expl = ca.vertcat(v, a)
    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = "robot"
    return model


def _make_solver(robot, model, horizon, total_steps, q0, rd, params=None, contacts=None):
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

    # cost matrices
    Qr = np.eye(3)
    Qq = 0 * np.eye(nq)
    Qv = 0.1 * np.eye(nv)
    Q = ca.diagcat(Qr, Qq, Qv).full()
    R = 0.01 * np.eye(nu)

    robot.forward(model.x, model.u[:na])
    r, C = robot.pose()

    v0 = np.zeros(nv)
    u0 = np.zeros(nu)
    x0 = np.concatenate((q0, v0))

    # path cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(r, model.x, model.u)
    ocp.cost.yref = np.concatenate((rd, x0, u0))
    ocp.cost.W = ca.diagcat(Q, R).full()

    # terminal cost
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.model.cost_y_expr_e = ca.vertcat(r, model.x)
    ocp.cost.yref_e = np.concatenate((rd, x0))
    ocp.cost.W_e = Q

    # input and force limits
    a_max = 10
    ocp.constraints.lbu = np.concatenate((-a_max * np.ones(na), np.zeros(nf)))
    ocp.constraints.ubu = np.concatenate(
        (a_max * np.ones(na), ACADOS_INFTY * np.ones(nf))
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
            force = model.u[na + i] * np.array([0, 0, 1])

            wc += contact_jacobian(contacts[i, :]).T @ force

        # assume all normals are vertical
        # normal_forces = model.u[na:]
        # normals = [np.array([0, 0, 1]) for _ in range(nf)]
        # forces = [f * n for f, n in zip(normal_forces, normals)]
        #
        # wc = np.sum([contact_jacobian(c).T @ f for c, f in zip(contacts, forces)])

        # spatial gravity in body frame
        g = ca.vertcat(np.zeros(3), C.T @ np.array([0, 0, -9.81]))

        # Newton-Euler equation for rigid body dynamics
        ξ = ca.vertcat(*robot.spatial_velocity())
        dξdt = ca.vertcat(*robot.spatial_acceleration())
        M = params.M
        V = adjoint(ξ)
        ocp.model.con_h_expr = M @ (dξdt - g) - V.T @ M @ ξ - wc
        ocp.constraints.lh = np.zeros(6)
        ocp.constraints.uh = np.zeros(6)

    # set solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
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
    def __init__(self, robot, horizon, steps_per_second, q0, rd, params=None, contacts=None):
        self.horizon = horizon
        self.total_steps = int(horizon * steps_per_second)
        self.dt = self.horizon / self.total_steps

        crobot = robot.casadi()
        nf = 0 if contacts is None else len(contacts)
        self.model = _make_model(crobot, nf=nf)
        self.solver = _make_solver(
            robot=crobot,
            model=self.model,
            horizon=horizon,
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

    def _get_input_at_time(self, t):
        if t < 0 or t > self.horizon:
            raise ValueError("time out of bounds")
        idx = int(t / self.dt)
        if idx >= self.total_steps:
            idx = self.total_steps - 1
        return self.solver.get(idx, "u")

    def rollout(self, dt, int_steps=1):
        integrator = _make_integrator(self.model, dt=dt, int_steps=int_steps)

        xd = self.solver.get(0, "x")
        xds = []
        us = []

        i = 0
        t = 0
        while t <= self.horizon:
            u = self._get_input_at_time(t)
            xd = integrator.simulate(x=xd, u=u)
            xds.append(xd)
            us.append(u)
            i += 1
            t = i * dt
        return np.array(xds), np.array(us)
