import time

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

import uplite


STEPS_PER_SECOND = 10
HORIZON = 5
TOTAL_STEPS = STEPS_PER_SECOND * HORIZON

SIM_TIMESTEP = 0.01

URDF_PATH = uplite.ASSETS_DIR / "combined.urdf"
ROBOT_HOME = np.array([0, -np.pi / 4, np.pi / 2, -np.pi / 4, np.pi / 2, 0])
TARGET_POSITION = np.array([-0.5, 0.0, 0.5])  # desired end-effector position

# TODO
# * add sticking constraints


class RobotModel:
    def __init__(self, model, ee_name, pin=pin):
        self.model = model
        self.data = model.createData()

        assert self.model.existFrame(ee_name)
        self.ee_idx = self.model.getFrameId(ee_name)
        self.ee_name = ee_name

        # store internal reference to pinocchio so we can use either the
        # regular or casadi version
        self._pin = pin

    @classmethod
    def from_urdf_file(cls, urdf_path, ee_name):
        model = pin.buildModelFromUrdf(urdf_path)
        return cls(model=model, ee_name=ee_name)

    def casadi(self):
        """Convert the model to CasADi format."""
        model = cpin.Model(self.model)
        return RobotModel(model=model, ee_name=self.ee_name, pin=cpin)

    def forward(self, x, u=None):
        q, v = x[: self.model.nq], x[self.model.nq :]
        if u is None:
            u = np.zeros(self.model.nv)
        self._pin.forwardKinematics(self.model, self.data, q, v, u)
        self._pin.updateFramePlacements(self.model, self.data)

    def pose(self):
        oMf = self.data.oMf[self.ee_idx]
        return oMf.translation, oMf.rotation

    # TODO we also need EE velocity and acceleration


def main():
    robot = RobotModel.from_urdf_file(urdf_path=URDF_PATH, ee_name="tray")
    nq = robot.model.nq
    nv = robot.model.nv

    # add a transported object
    params = uplite.InertialParameters(
        mass=1.0,
        com=[0, 0, 0.1],
        inertia=np.diag([0.01, 0.01, 0.01]),
    )
    box = uplite.TransportedObject.box(params=params, mu=0.5, rx=0.05, ry=0.05)

    # plan trajectory
    # TODO the API should eventually allow separate RTI settings
    robot.forward(x=np.concatenate((ROBOT_HOME, np.zeros(nv))))
    r0 = robot.pose()[0]
    rd = r0 + TARGET_POSITION
    planner = uplite.Planner(
        robot=robot,
        horizon=HORIZON,
        steps_per_second=STEPS_PER_SECOND,
        q0=ROBOT_HOME,
        rd=rd,
    )
    status = planner.solve()
    if status != 0:
        raise Exception(f"acados returned status {status}.")

    # simulation
    sim = uplite.BulletSimulation(
        URDF_PATH, tool_link_name="tray", timestep=SIM_TIMESTEP, q0=ROBOT_HOME
    )
    sim.add_transported_box(params=params, mu=0.5, rx=0.05, ry=0.05)

    # rollout trajectory solution at simulated frequency
    xds, us = planner.rollout(dt=sim.timestep)

    kp = 10

    ts = []
    rs = []

    t = 0
    while t < HORIZON:

        # track desired trajectory
        xd = xds[sim.steps]
        qd, vd = xd[:nq], xd[nq:]
        q, v = sim.robot.get_joint_states()
        v_cmd = kp * (qd - q) + vd
        sim.robot.command_velocity(v_cmd)

        # record
        r = sim.robot.get_link_frame_pose()[0]
        ts.append(t)
        rs.append(r)

        t = sim.step()
        time.sleep(sim.timestep)

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
