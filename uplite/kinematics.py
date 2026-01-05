import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin


class RobotKinematics:
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
        return RobotKinematics(model=model, ee_name=self.ee_name, pin=cpin)

    def forward(self, q, v=None, a=None):
        if v is None:
            v = np.zeros(self.model.nv)
        if a is None:
            a = np.zeros(self.model.nv)
        self._pin.forwardKinematics(self.model, self.data, q, v, a)
        self._pin.updateFramePlacements(self.model, self.data)

    def pose(self):
        oMf = self.data.oMf[self.ee_idx]
        return oMf.translation, oMf.rotation

    def spatial_velocity(self):
        v = self._pin.getFrameVelocity(
            self.model, self.data, self.ee_idx, pin.ReferenceFrame.LOCAL
        )
        return v.angular, v.linear

    def spatial_acceleration(self):
        a = self._pin.getFrameAcceleration(
            self.model, self.data, self.ee_idx, pin.ReferenceFrame.LOCAL
        )
        return a.angular, a.linear

    def classical_acceleration(self):
        a = self._pin.getFrameClassicalAcceleration(
            self.model, self.data, self.ee_idx, pin.ReferenceFrame.LOCAL
        )
        return a.angular, a.linear
