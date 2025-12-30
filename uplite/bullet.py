import pybullet as pyb
import pybullet_data
import pyb_utils


class BulletSimulation:
    def __init__(
        self,
        urdf_path,
        tool_link_name="tray",
        timestep=0.01,
        q0=None,
        position=(0, 0, 1),
    ):
        pyb.connect(pyb.GUI)
        pyb.setGravity(0, 0, -9.81)
        pyb.setTimeStep(timestep)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        robot_id = pyb.loadURDF(
            str(urdf_path),
            position,
            useFixedBase=True,
        )
        self.robot = pyb_utils.Robot(robot_id, tool_link_name=tool_link_name)
        self.timestep = timestep
        self.steps = 0

        # pybullet uses multiplicative friction model, such that each contact
        # point's friction coefficient is the product of that of each object.
        # Therefore we set the tray friction to 1, so we can control the
        # contact friction with the transported object only.
        pyb.changeDynamics(self.robot.uid, self.robot.tool_idx, lateralFriction=1)

        if q0 is not None:
            self.robot.reset_joint_configuration(q0)

        pyb_utils.debug_frame(
            size=0.2,
            obj_uid=robot_id,
            link_index=self.robot.tool_idx,
        )

    def step(self):
        pyb.stepSimulation()
        self.steps += 1
        return self.steps * self.timestep

    def add_transported_box(self, params, mu, rx, ry, color=(1, 0, 0, 1)):
        p, R = self.robot.get_link_frame_pose(as_rotation_matrix=True)

        rz = params.com[2]
        box = pyb_utils.BulletBody.box(
            position=p + R @ params.com, half_extents=[rx, ry, rz], color=color
        )

        # set friction
        pyb.changeDynamics(box.uid, -1, lateralFriction=mu)

        # set inertia
        # TODO need parallel axis theorem (add to params class)

        return box
