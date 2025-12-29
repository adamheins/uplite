import numpy as np


class InertialParameters:
    def __init__(self, mass, com, inertia):
        self.mass = mass
        self.com = np.array(com)
        self.inertia = np.array(inertia)

        assert self.mass > 0, "Mass must be positive."
        assert self.com.shape == (
            3,
        ), "Center of mass must be a 3-dimensional vector."
        assert self.inertia.shape == (3, 3), "Inertia must be a 3x3 matrix."

    @property
    def Ic(self):
        pass


class ContactPoint:
    def __init__(self, position, mu, normal=(0, 0, 1)):
        self.position = np.array(position)
        self.normal = np.array(normal) / np.linalg.norm(normal)
        self.mu = mu

        assert self.mu >= 0, "Friction coefficient must be non-negative."

        # TODO tangent vectors


class TransportedObject:
    def __init__(self, params, contacts):
        self.params = params
        self.contacts = contacts

    @classmethod
    def box(cls, params, mu, rx, ry):
        vertices = np.array(
            [[rx, ry, 0], [rx, -ry, 0], [-rx, ry, 0], [-rx, -ry, 0]]
        )
        contacts = [ContactPoint(v, mu) for v in vertices]
        return cls(params, contacts)
