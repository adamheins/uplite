import pybullet as pyb
import pybullet_data
import pyb_utils


class BulletSimulation:
    """PyBullet simulation environment for the waiter's problem.

    Parameters
    ----------
    urdf_path : str or Path
        Path to the robot URDF file.
    tool_link_name : str, optional
        Name of the tool (end-effector) link frame. Defaults to "tray".
    timestep : float, optional
        Simulation timestep in seconds. Defaults to 0.01s.
    q0 : array, shape (nq,), optional
        Initial joint configuration. If None, an all-zero configuration is used.
    gravity : tuple, shape (3,), optional
        Gravity vector. Defaults to (0, 0, -9.81).
    """
    def __init__(
        self,
        urdf_path,
        tool_link_name="tray",
        timestep=0.01,
        q0=None,
        gravity=(0, 0, -9.81),
    ):
        pyb.connect(pyb.GUI)
        pyb.setGravity(*gravity)
        pyb.setTimeStep(timestep)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

        robot_id = pyb.loadURDF(
            str(urdf_path),
            useFixedBase=True,
        )
        self.robot = pyb_utils.Robot(robot_id, tool_link_name=tool_link_name)
        self.timestep = timestep
        self.steps = 0

        # pybullet uses multiplicative friction model, such that each contact
        # point's friction coefficient is the product of that of each object.
        # Therefore we set the tray friction to 1, so we can control the
        # contact friction with the transported object only.
        pyb.changeDynamics(
            self.robot.uid, self.robot.tool_idx, lateralFriction=1
        )

        if q0 is not None:
            self.robot.reset_joint_configuration(q0)

        pyb_utils.debug_frame(
            size=0.2,
            obj_uid=robot_id,
            link_index=self.robot.tool_idx,
        )

    def step(self):
        """Step the simulation forward by one timestep.

        Returns
        -------
        : float
            Current simulation time.
        """
        pyb.stepSimulation()
        self.steps += 1
        return self.steps * self.timestep

    def add_transported_box(self, params, mu, w, color=(1, 0, 0, 1)):
        """Add a simulated box object on the robot tray.

        Parameters
        ----------
        params : InertialParameters
            Inertial parameters of the box. The box is assumed to be centered
            about the center of mass.
        mu : float, positive
            Friction coefficient between the box and the tray.
        w : float, positive
            Width of the (square) base of the box.
        color : tuple, shape (4,), optional
            RGBA color of the box. Defaults to red.

        Returns
        -------
        : pyb_utils.BulletBody
            The simulated box object.
        """

        p, R = self.robot.get_link_frame_pose(as_rotation_matrix=True)

        r = 0.5 * w
        rz = params.com[2]
        box = pyb_utils.BulletBody.box(
            mass=params.mass,
            position=p + R @ params.com,
            half_extents=[r, r, rz],
            color=color,
        )

        # set friction and inertia
        pyb.changeDynamics(
            box.uid, -1, lateralFriction=mu, localInertiaDiagonal=params.Ic_diag
        )

        return box
