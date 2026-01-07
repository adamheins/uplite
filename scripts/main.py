import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import uplite


# planner timing
STEPS_PER_SECOND = 10
HORIZON = 5
TOTAL_STEPS = STEPS_PER_SECOND * HORIZON

# simulation timing
SIM_TIMESTEP = 0.01

# robot URDF
URDF_PATH = uplite.ASSETS_DIR / "combined.urdf"
TOOL_LINK_NAME = "tray"

# initial joint configuration
ROBOT_HOME = np.array([0, -np.pi / 4, np.pi / 2, -np.pi / 4, np.pi / 2, 0])

# EE goal position relative to its initial position
RELATIVE_TARGET = np.array([-0.8, 0.8, 0])

# width of the (square) base of the transported box
BASE_WIDTH = 0.1

# friction coefficient in the simulation
SIMULATED_FRICTION = 0.3

# controller gain for joint position error
P_GAIN = 10


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser(
        description="Robot waiter planner and simulation"
    )
    parser.add_argument(
        "-c",
        "--constraint",
        type=str,
        default="none",
        choices=["none", "upward", "aligned", "robust"],
        help="Type of upright constraint to use in the planner",
    )
    args = parser.parse_args()

    kinematics = uplite.RobotKinematics.from_urdf_file(
        urdf_path=URDF_PATH, tool_link_name=TOOL_LINK_NAME
    )

    # add a transported object
    params = uplite.InertialParameters(
        mass=1.0,
        com=[0, 0, 0.2],
        inertia=np.diag([0.01, 0.01, 0.01]),
        inertia_about_com=True,
    )
    transobj = uplite.TransportedObject.box(params=params, w=BASE_WIDTH)

    # plan trajectory
    kinematics.forward(q=ROBOT_HOME)
    r0 = kinematics.pose()[0]
    goal = r0 + RELATIVE_TARGET
    planner = uplite.Planner(
        kinematics=kinematics,
        horizon=HORIZON,
        steps_per_second=STEPS_PER_SECOND,
        q0=ROBOT_HOME,
        goal=goal,
        transobj=transobj,
        upright_constraint=args.constraint,
    )
    status = planner.plan(verbose=True)
    if status != 0:
        raise Exception(f"acados returned status {status}.")
    qspline = planner.get_solution_spline()
    vspline = qspline.derivative()

    # now simulate it
    sim = uplite.BulletSimulation(
        urdf_path=URDF_PATH,
        tool_link_name=TOOL_LINK_NAME,
        timestep=SIM_TIMESTEP,
        q0=ROBOT_HOME,
    )
    box = sim.add_transported_box(params=params, mu=SIMULATED_FRICTION, w=BASE_WIDTH)

    ts = []
    rs = []
    Qs = []
    qds = []
    vds = []
    qs = []
    vs = []

    t = 0
    while t < HORIZON:

        # track desired trajectory
        qd = qspline(t)
        vd = vspline(t)
        q, v = sim.robot.get_joint_states()
        v_cmd = P_GAIN * (qd - q) + vd
        sim.robot.command_velocity(v_cmd)

        # record
        r, Q = sim.robot.get_link_frame_pose()

        # TODO need to compute and plot object error
        ro = box.get_pose()[0]

        ts.append(t)
        rs.append(r)
        Qs.append(Q)
        qds.append(qd)
        vds.append(vd)
        qs.append(q)
        vs.append(v)

        t = sim.step()
        time.sleep(sim.timestep)

    ts = np.array(ts)
    rs = np.array(rs)
    Qs = np.array(Qs)
    qds = np.array(qds)
    vds = np.array(vds)
    qs = np.array(qs)
    vs = np.array(vs)

    t_sols = planner.get_solution_times()
    u_sols = planner.get_solution_inputs()
    x_sols = planner.get_solution_states()

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.figure()
    plt.plot(ts, goal[0] - rs[:, 0], label="x")
    plt.plot(ts, goal[1] - rs[:, 1], label="y")
    plt.plot(ts, goal[2] - rs[:, 2] + 1, label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("Position error")
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(ts, Qs[:, 0], label="x")
    plt.plot(ts, Qs[:, 1], label="y")
    plt.plot(ts, Qs[:, 2], label="z")
    plt.plot(ts, Qs[:, 3], label="w")
    plt.xlabel("Time [s]")
    plt.ylabel("Orientation quaternion")
    plt.title("Orientation")
    plt.grid()
    plt.legend()

    plt.figure()
    for i in range(kinematics.model.nq):
        plt.plot(ts, qs[:, i], label=f"q_{i+1}", color=colors[i])
        plt.plot(ts, qds[:, i], "--", label=f"qd_{i+1}", color=colors[i])
        plt.plot(t_sols, x_sols[:, i], ".", color=colors[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Joint angles [rad]")
    plt.title("Joint angles")
    plt.grid()
    plt.legend()

    plt.figure()
    for i in range(kinematics.model.nv):
        plt.plot(ts, vs[:, i], label=f"v_{i+1}", color=colors[i])
        plt.plot(ts, vds[:, i], "--", label=f"vd_{i+1}", color=colors[i])
        plt.plot(
            t_sols, x_sols[:, kinematics.model.nq + i], ".", color=colors[i]
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocities [rad/s]")
    plt.title("Joint velocities")
    plt.grid()
    plt.legend()

    nf = u_sols.shape[1] - kinematics.model.nv
    plt.figure()
    for i in range(nf):
        plt.plot(
            t_sols[:-1], u_sols[:, kinematics.model.nv + i], label=f"f_{i+1}"
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Contact forces [N]")
    plt.title("Contact forces")
    plt.grid()
    plt.legend()

    plt.figure()
    for i in range(kinematics.model.nv):
        plt.plot(t_sols[:-1], u_sols[:, i], label=f"u_{i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Jerk [rad/s^3]")
    plt.title("Joint input jerk")
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
