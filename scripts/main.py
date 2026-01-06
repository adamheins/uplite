import time

import matplotlib.pyplot as plt
import numpy as np

import uplite


STEPS_PER_SECOND = 10
HORIZON = 10
TOTAL_STEPS = STEPS_PER_SECOND * HORIZON

SIM_TIMESTEP = 0.01

URDF_PATH = uplite.ASSETS_DIR / "combined.urdf"
ROBOT_HOME = np.array([0, -np.pi / 4, np.pi / 2, -np.pi / 4, np.pi / 2, 0])
RELATIVE_TARGET = np.array([-0.5, 0.4, 0])  # relative EE target position

BASE_WIDTH = 0.1
SIMULATED_FRICTION = 0.5  # friction coefficient in the simulation
P_GAIN = 10

# TODO
# * add sticking constraints
# * can start with a basic upright constraint

# (a - R @ g) @ z = 0


def main():
    np.set_printoptions(precision=4, suppress=True)

    robot = uplite.RobotKinematics.from_urdf_file(
        urdf_path=URDF_PATH, tool_link_name="tray"
    )

    # add a transported object
    params = uplite.InertialParameters(
        mass=1.0,
        com=[0, 0, 0.1],
        inertia=np.diag([0.01, 0.01, 0.01]),
        inertia_about_com=True,
    )
    box = uplite.TransportedObject.box(params=params, w=BASE_WIDTH)

    # plan trajectory
    # TODO the API should eventually allow separate RTI settings
    robot.forward(q=ROBOT_HOME)
    r0 = robot.pose()[0]
    goal = r0 + RELATIVE_TARGET
    planner = uplite.Planner(
        robot=robot,
        horizon=HORIZON,
        steps_per_second=STEPS_PER_SECOND,
        q0=ROBOT_HOME,
        goal=goal,
        transobj=box,
    )
    status = planner.solve(verbose=True)
    if status != 0:
        raise Exception(f"acados returned status {status}.")
    qspline = planner.get_solution_spline()
    vspline = qspline.derivative()

    # now simulate it
    sim = uplite.BulletSimulation(
        urdf_path=URDF_PATH,
        tool_link_name="tray",
        timestep=SIM_TIMESTEP,
        q0=ROBOT_HOME,
    )
    sim.add_transported_box(params=params, mu=SIMULATED_FRICTION, w=BASE_WIDTH)

    ts = []
    rs = []
    rds = []

    t = 0
    while t < HORIZON:

        # track desired trajectory
        qd = qspline(t)
        vd = vspline(t)
        q, v = sim.robot.get_joint_states()
        v_cmd = P_GAIN * (qd - q) + vd
        sim.robot.command_velocity(v_cmd)

        # record
        robot.forward(q=qd, v=vd)
        rd = robot.pose()[0].copy()

        r = sim.robot.get_link_frame_pose()[0]
        ts.append(t)
        rs.append(r)
        rds.append(rd)
        # TODO more stuff

        t = sim.step()
        time.sleep(sim.timestep)

    rs = np.array(rs)
    rds = np.array(rds)

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
    plt.plot(ts, rs[:, 0], label="x", color="r")
    plt.plot(ts, rs[:, 1], label="y", color="g")
    plt.plot(ts, rs[:, 2], label="z", color="b")
    plt.plot(ts, rds[:, 0], "--", label="xd", color="r")
    plt.plot(ts, rds[:, 1], "--", label="yd", color="g")
    plt.plot(ts, rds[:, 2] + 1, "--", label="zd", color="b")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Positions")
    plt.grid()
    plt.legend()

    t_sols = planner.get_solution_times()[:-1]
    u_sols = planner.get_solution_inputs()
    plt.figure()
    for i in range(4):
        plt.plot(t_sols, u_sols[:, 6 + i], label=f"f_{i+1}")
    plt.xlabel("Time [s]")
    plt.ylabel("Contact forces [N]")
    plt.title("Contact forces")
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
