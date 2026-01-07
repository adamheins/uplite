import acados_template as at
import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

from .transport import adjoint, contact_jacobian

JOINT_JERK_LIMIT = 100  # rad/s^3
JOINT_ACC_LIMIT = 20  # rad/s^2
JOINT_VEL_LIMIT = np.pi  # rad/s
JOINT_POS_LIMIT = 2 * np.pi  # rad

EE_POS_COST = 1
JOINT_POS_COST = 0
JOINT_VEL_COST = 0.1
JOINT_ACC_COST = 0.01
JOINT_JERK_COST = 0.001
CONTACT_FORCE_COST = 0.001

GRAVITY = 9.81  # m/s^2

# slack penalties for robust sticking constraint
ROBUST_L1_SLACK_WEIGHT = 10
ROBUST_L2_SLACK_WEIGHT = 0


def _make_model(kinematics, transobj=None):
    """Make the acados model for the robot waiter dynamics.

    Parameters
    ----------
    kinematics : RobotKinematics
        The robot kinematics model.
    transobj : TransportedObject, optional
        The transported object model, by default None if no object is being
        modeled.

    Returns
    -------
    : AcadosModel
        The acados model.
    """
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

    model = at.AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = "robot_waiter"
    return model


def _split_x(x, nq, nv):
    """Split the state vector into its component parts."""
    q = x[:nq]
    v = x[nq : nq + nv]
    a = x[nq + nv :]
    return q, v, a


# NOTE: this is where the trajectory optimization problem is defined: modify
# for your needs
def _make_solver(
    kinematics,
    model,
    horizon,
    total_steps,
    q0,
    goal,
    constraint_type="none",
    transobj=None,
):
    """Make the acados solver for the robot waiter trajectory optimization problem.

    Parameters
    ----------
    kinematics : RobotKinematics
        The robot kinematics model (using casadi).
    model : AcadosModel
        The acados model.
    horizon : float
        Time horizon for the trajectory in seconds.
    total_steps : int
        Total number of discretization steps.
    q0 : array, shape (nq,)
        Initial joint configuration.
    goal : array, shape (3,)
        Desired end-effector goal position.
    constraint_type : str, optional
        Type of object constraint to use in the planner. Options are 'none'
        (default), 'upward', 'aligned', and 'robust'.
    transobj : TransportedObject, optional
        The transported object model (only required for robust constraints).

    Returns
    -------
    : AcadosOcpSolver
        The acados solver.
    """
    ocp = at.AcadosOcp()
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
    Qr = EE_POS_COST * np.eye(3)
    Qq = JOINT_POS_COST * np.eye(nq)
    Qv = JOINT_VEL_COST * np.eye(nv)
    Qa = JOINT_ACC_COST * np.eye(nv)
    Q = ca.diagcat(Qr, Qq, Qv, Qa).full()

    Rj = JOINT_JERK_COST * np.eye(nv)
    Rf = CONTACT_FORCE_COST * np.eye(nf)
    R = ca.diagcat(Rj, Rf).full()

    # casadi forward kinematics
    q, v, a = _split_x(model.x, nq, nv)
    kinematics.forward(q=q, v=v, a=a)
    r, C = kinematics.pose()

    # path cost
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_y_expr = ca.vertcat(r, model.x, model.u)
    ocp.cost.yref = np.concatenate((goal, x0, u0))
    ocp.cost.W = ca.diagcat(Q, R).full()

    # initial state
    ocp.constraints.x0 = x0

    # terminal constraint: EE at goal position with zero velocity and
    # acceleration
    ocp.model.con_h_expr_e = ca.vertcat(r - goal, model.x[nq:])
    ocp.constraints.lh_e = np.zeros(3 + 2 * nv)
    ocp.constraints.uh_e = np.zeros(3 + 2 * nv)

    # input/force limits (normal forces need to be non-negative)
    ocp.constraints.lbu = np.concatenate(
        (-JOINT_JERK_LIMIT * np.ones(nv), np.zeros(nf))
    )
    ocp.constraints.ubu = np.concatenate(
        (JOINT_JERK_LIMIT * np.ones(nv), at.ACADOS_INFTY * np.ones(nf))
    )
    ocp.constraints.idxbu = np.arange(nu)

    # state limits
    x_lim = np.concatenate(
        (
            JOINT_POS_LIMIT * np.ones(nq),
            JOINT_VEL_LIMIT * np.ones(nv),
            JOINT_ACC_LIMIT * np.ones(nv),
        )
    )
    ocp.constraints.lbx = -x_lim
    ocp.constraints.ubx = x_lim
    ocp.constraints.idxbx = np.arange(nx)

    z = np.array([0, 0, 1])  # all normals are assumed to be vertical
    if constraint_type == "upward":
        # keep the tray oriented upward
        ocp.model.con_h_expr = C @ z - z
        ocp.constraints.lh = np.zeros(3)
        ocp.constraints.uh = np.zeros(3)
    elif constraint_type == "aligned":
        # keep total acceleration aligned with tray's normal
        a = kinematics.classical_acceleration()[0]
        g = GRAVITY * z
        ocp.model.con_h_expr = ca.cross(z, a + C @ g)
        ocp.constraints.lh = np.zeros(3)
        ocp.constraints.uh = np.zeros(3)
    elif constraint_type == "robust":
        # more robust Newton-Euler constraint with contact forces
        # contact wrench
        wc = np.zeros(6)
        for i in range(nf):
            force = model.u[nv + i] * z
            G = contact_jacobian(transobj.contacts[i, :]).T
            wc += G @ force

        # spatial gravity in body frame
        g = ca.vertcat(np.zeros(3), C.T @ np.array([0, 0, -GRAVITY]))

        # Newton-Euler equation for rigid body dynamics
        ξ = ca.vertcat(*kinematics.spatial_velocity())
        dξdt = ca.vertcat(*kinematics.spatial_acceleration())
        h = transobj.params.ne(ξ, dξdt - g) - wc

        ocp.model.con_h_expr = h
        ocp.constraints.lh = np.zeros(6)
        ocp.constraints.uh = np.zeros(6)

        # slack variables to relax the constraint
        ns = 6
        ocp.cost.zl = ROBUST_L1_SLACK_WEIGHT * np.ones(ns)
        ocp.cost.Zl = ROBUST_L2_SLACK_WEIGHT * np.ones(ns)
        ocp.cost.zu = ROBUST_L1_SLACK_WEIGHT * np.ones(ns)
        ocp.cost.Zu = ROBUST_L2_SLACK_WEIGHT * np.ones(ns)
        ocp.constraints.idxsh = np.arange(ns)
    elif constraint_type == "none":
        pass
    else:
        raise ValueError(f"Unknown constraint: {constraint_type}")

    # set solver options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = 1000
    ocp.solver_options.qp_solver_iter_max = 1000

    return at.AcadosOcpSolver(ocp)


class Planner:
    """Trajectory planner for the robot waiter problem.

    Parameters
    ----------
    kinematics : RobotKinematics
        The robot kinematics model.
    horizon : float
        Time horizon for the trajectory in seconds.
    steps_per_second : int
        Number of discretization steps per second.
    q0 : array, shape (nq,)
        Initial joint configuration.
    goal : array, shape (3,)
        Desired end-effector goal position.
    constraint_type : str, optional
        Type of object constraint to use in the planner. Options are 'none'
        (default), 'upward', 'aligned', and 'robust'.
    transobj : TransportedObject, optional
        The transported object model (only required for robust constraints).
    """

    def __init__(
        self,
        kinematics,
        horizon,
        steps_per_second,
        q0,
        goal,
        constraint_type="none",
        transobj=None,
    ):
        self.horizon = horizon
        self.total_steps = int(horizon * steps_per_second)
        self.dt = 1.0 / steps_per_second

        self.kinematics = kinematics.to_casadi()
        self.model = _make_model(self.kinematics, transobj=transobj)

        self.solver = _make_solver(
            kinematics=self.kinematics,
            model=self.model,
            horizon=self.horizon,
            total_steps=self.total_steps,
            q0=q0,
            goal=goal,
            transobj=transobj,
            constraint_type=constraint_type,
        )

    def plan(self, verbose=False):
        """Plan the trajectory.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print solver statistics, by default False.

        Returns
        -------
        : int
            Solver status code. A non-zero value indicates an error.
        """
        status = self.solver.solve()
        if verbose:
            self.solver.print_statistics()
        return status

    # TODO: possible API for online replanning/MPC using RTI
    def replan(self, x, verbose=False):
        pass

    def get_solution_times(self):
        """Get the times at the solution knot points."""
        return np.arange(self.total_steps + 1) * self.dt

    def get_solution_states(self):
        """Get the optimal states at the solution knot points."""
        return np.array(
            [self.solver.get(i, "x") for i in range(self.total_steps + 1)]
        )

    def get_solution_inputs(self):
        """Get the optimal inputs at the solution knot points."""
        return np.array(
            [self.solver.get(i, "u") for i in range(self.total_steps)]
        )

    def get_solution_spline(self):
        """Get a spline representing the optimal joint position trajectory."""
        ts = self.get_solution_times()
        xs = self.get_solution_states()
        qs = xs[:, : self.kinematics.model.nq]
        return CubicSpline(ts, qs, axis=0)
