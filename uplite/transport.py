import numpy as np


def skew(v):
    """Convert a 3-vector to a skew-symmetric matrix."""
    # assert v.shape == (3,)
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def adjoint(ξ):
    """Adjoint of a spatial velocity."""
    ω, v = ξ[:3], ξ[3:]
    Sv = skew(v)
    Sω = skew(ω)
    return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])


def contact_jacobian(c):
    """Maps body spatial velocity ξ=(ω, v) to contact point velocity."""
    return np.hstack((-skew(c), np.eye(3)))


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
        """Inertia matrix about the center of mass."""
        C = skew(self.com)
        return np.inertia + self.mass * C @ C

    @property
    def M(self):
        """Spatial mass matrix."""
        S = self.mass * skew(self.com)
        return np.block([[self.inertia, S], [-S, self.mass * np.eye(3)]])


# class ContactPoint:
#     def __init__(self, position, mu, normal=(0, 0, 1)):
#         self.position = np.array(position)
#         self.normal = np.array(normal) / np.linalg.norm(normal)
#         self.mu = mu
#
#         assert self.mu >= 0, "Friction coefficient must be non-negative."


class TransportedObject:
    def __init__(self, params, contacts):
        self.params = params
        self.contacts = contacts

    # TODO: use half extents
    @classmethod
    def box(cls, params, rx, ry):
        vertices = np.array(
            [[rx, ry, 0], [rx, -ry, 0], [-rx, ry, 0], [-rx, -ry, 0]]
        )
        return cls(params=params, contacts=vertices)
