import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin


class RobotKinematics:
    """Robot kinematics model.

    Parameters
    ----------
    model : pin.Model
        Pinocchio robot model.
    tool_link_name : str
        Name of the tool (end-effector) link frame.
    """
    def __init__(self, model, tool_link_name, pin=pin):
        self.model = model
        self.data = model.createData()

        assert self.model.existFrame(tool_link_name)
        self.tool_idx = self.model.getFrameId(tool_link_name)
        self.tool_link_name = tool_link_name

        # store internal reference to pinocchio so we can use either the
        # regular or casadi version
        self._pin = pin

    @classmethod
    def from_urdf_file(cls, urdf_path, tool_link_name):
        """Construct the model from a URDF file.

        Parameters
        ----------
        urdf_path : str or Path
            Path to the URDF file.
        tool_link_name : str
            Name of the tool (end-effector) link frame.
        """
        model = pin.buildModelFromUrdf(urdf_path)
        return cls(model=model, tool_link_name=tool_link_name)

    def to_casadi(self):
        """Convert the model to CasADi format."""
        model = cpin.Model(self.model)
        return RobotKinematics(
            model=model, tool_link_name=self.tool_link_name, pin=cpin
        )

    def forward(self, q, v=None, a=None):
        """Compute forward kinematics.

        This should be called before querying pose, velocity, or acceleration.

        Parameters
        ----------
        q : array, shape (nq,)
            Joint positions.
        v : array, shape (nv,), optional
            Joint velocities. If None, assumed to be zero.
        a : array, shape (nv,), optional
            Joint accelerations. If None, assumed to be zero.
        """
        if v is None:
            v = np.zeros(self.model.nv)
        if a is None:
            a = np.zeros(self.model.nv)
        self._pin.forwardKinematics(self.model, self.data, q, v, a)
        self._pin.updateFramePlacements(self.model, self.data)

    def pose(self, frame_idx=None):
        """Get the frame pose.

        Ensure that `forward` has been called before invoking this method.

        Parameters
        ----------
        frame_idx : int, optional
            Index of the frame. If None, uses the tool frame.

        Returns
        -------
        :
            A tuple consisting of the frame position and orientation as a
            rotation matrix.
        """
        if frame_idx is None:
            frame_idx = self.tool_idx
        oMf = self.data.oMf[frame_idx]
        return oMf.translation, oMf.rotation

    def spatial_velocity(self, frame_idx=None):
        """Frame spatial velocity.

        Ensure that `forward` has been called before invoking this method.

        Parameters
        ----------
        frame_idx : int, optional
            Index of the frame. If None, uses the tool frame.

        Returns
        -------
        :
            A tuple consisting of the angular and linear velocity.
        """
        if frame_idx is None:
            frame_idx = self.tool_idx
        v = self._pin.getFrameVelocity(
            self.model, self.data, frame_idx, pin.ReferenceFrame.LOCAL
        )
        return v.angular, v.linear

    def spatial_acceleration(self, frame_idx=None):
        """Frame spatial acceleration.

        Ensure that `forward` has been called before invoking this method. Note
        that this is different than the classical acceleration.

        Parameters
        ----------
        frame_idx : int, optional
            Index of the frame. If None, uses the tool frame.

        Returns
        -------
        :
            A tuple consisting of the angular and linear (spatial)
            acceleration.
        """
        if frame_idx is None:
            frame_idx = self.tool_idx
        a = self._pin.getFrameAcceleration(
            self.model, self.data, frame_idx, pin.ReferenceFrame.LOCAL
        )
        return a.angular, a.linear

    def classical_acceleration(self, frame_idx=None):
        """Frame classical acceleration.

        Ensure that `forward` has been called before invoking this method.

        Parameters
        ----------
        frame_idx : int, optional
            Index of the frame. If None, uses the tool frame.

        Returns
        -------
        :
            A tuple consisting of the angular and linear (classical)
            acceleration.
        """
        if frame_idx is None:
            frame_idx = self.tool_idx
        a = self._pin.getFrameClassicalAcceleration(
            self.model, self.data, frame_idx, pin.ReferenceFrame.LOCAL
        )
        return a.angular, a.linear
