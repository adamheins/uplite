

def _make_model(robot, jerk_input=False):
    nx = robot.nq + robot.nv
    q = SX.sym("q", robot.nq)
    v = SX.sym("v", robot.nv)
    u = SX.sym("u", robot.nv)

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
    # model.name = name
    return model


def _make_solver(robot):
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
    ocp.solver_options.N_horizon = TOTAL_STEPS
    ocp.solver_options.tf = HORIZON

    # cost matrices
    Qr = np.eye(3)
    Qq = 0 * np.eye(nq)
    Qv = 0.1 * np.eye(nv)
    Q = diagcat(Qr, Qq, Qv).full()
    R = 0.01 * np.eye(nu)

    crobot.forward(model.x, model.u)
    ee_pos = crobot.pose()[0]

    q0 = ROBOT_HOME
    v0 = np.zeros(nv)
    u0 = np.zeros(nu)
    x0 = np.concatenate((q0, v0))

    # desired end-effector position
    robot.forward(x0, u0)
    r0 = robot.pose()[0]
    rd = r0 + TARGET_POSITION

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

    return AcadosOcpSolver(ocp)


def _make_integrator(model, dt, steps=1):
    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T = dt  # simulation time
    sim.solver_options.num_steps = steps
    sim.code_export_directory = "acados/c_generated_code_sim"
    sim.json_file = "acados/sim.json"
    return AcadosSimSolver(sim)



class Planner:
    def __init__(self, robot, horizon, steps_per_second, x0, target):
        self.horizon = horizon
        self.steps = int(horizon * steps_per_second)

        self.crobot = robot.casadi()
        self.model = _make_model(self.crobot)
        self.solver = _make_solver(self.crobot, horizon, steps_per_second)

    def solve(self, verbose=False):
        status = self.solver.solve()
        if verbose:
            self.solver.print_statistics()
        return status

    def rollout(self, dt, steps=1):
        integrator = _make_integrator(self.model, dt=dt, steps=steps)
