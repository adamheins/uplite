import numpy as np


def skew(v):
    """Convert a 3-vector ``v`` to a skew-symmetric matrix ``S(v)``, such that
    ``S(v) @ u = v x u`` for any 3-vectors ``v, u``.

    Parameters
    ----------
    v : array, shape (3,)
        Input vector.

    Returns
    -------
    : array, shape (3, 3)
        Skew-symmetric matrix corresponding to v.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def adjoint(ξ):
    """Adjoint of a spatial velocity.

    Parameters
    ----------
    ξ : array, shape (6,)
        Spatial velocity vector, with angular part ω and linear part v (in that
        order).

    Returns
    -------
    : array, shape (6, 6)
        Adjoint matrix corresponding to ξ.
    """
    ω, v = ξ[:3], ξ[3:]
    Sv = skew(v)
    Sω = skew(ω)
    return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])


def contact_jacobian(c):
    """Compute the Jacobian matrix that maps a body-frame spatial velocity to
    contact point velocity.

    Parameters
    ----------
    c : array, shape (3,)
        Contact point position in body frame.

    Returns
    -------
    : array, shape (3, 6)
        Contact Jacobian matrix.
    """
    return np.hstack((-skew(c), np.eye(3)))


class InertialParameters:
    """Rigid body inertial parameters.

    Attributes
    ----------
    mass : float, positive
        Mass of the rigid body.
    com : array, shape (3,)
        Center of mass position.
    Ic_diag : array, shape (3,)
        Diagonal of the inertia matrix about the center of mass.
    """

    def __init__(self, mass, com, Ic_diag):
        self.mass = mass
        self.com = np.array(com)
        self.Ic_diag = np.array(Ic_diag)

        assert self.mass > 0, "Mass must be positive."
        assert self.com.shape == (
            3,
        ), "Center of mass must be a 3-dimensional vector."
        assert self.Ic_diag.shape == (3,)
        assert np.all(
            self.Ic_diag >= 0
        ), "Inertia diagonal elements must be non-negative."

    @classmethod
    def uniform_density_box(cls, mass, com, half_extents):
        """Create inertia parameters for a box with uniform density.

        Parameters
        ----------
        mass : float, positive
            Mass of the box.
        com : array, shape (3,)
            Center of mass position.
        half_extents : array, shape (3,)
            Half-lengths of the box along each axis.
        """
        hx, hy, hz = half_extents
        Ic_xx = mass * (hy**2 + hz**2) / 3
        Ic_yy = mass * (hx**2 + hz**2) / 3
        Ic_zz = mass * (hx**2 + hy**2) / 3
        return cls(mass=mass, com=com, Ic_diag=[Ic_xx, Ic_yy, Ic_zz])

    @property
    def Ic(self):
        """Inertia matrix with respect to the center of mass."""
        return np.diag(self.Ic_diag)

    @property
    def I(self):
        """Inertia matrix with respect to the origin."""
        C = skew(self.com)
        return self.Ic + self.mass * C @ C.T

    @property
    def M(self):
        """Spatial mass matrix."""
        S = self.mass * skew(self.com)
        return np.block([[self.I, S], [-S, self.mass * np.eye(3)]])

    def ne(self, ξ, dξdt):
        """Compute the Newton-Euler equations."""
        M = self.M
        V = adjoint(ξ)
        return M @ dξdt - V.T @ M @ ξ


class TransportedObject:
    """Object to be transported on a tray by the robot.

    Attributes
    ----------
    params : InertialParameters
        Inertial parameters of the object.
    contacts : array, shape (n, 3)
        Contact point positions in body frame between the object and the tray.
        Currently we assume that the contact normals are all aligned with the
        z-axis.
    """
    def __init__(self, params, contacts):
        self.params = params
        self.contacts = contacts

    @classmethod
    def box(cls, params, w):
        """Create a transported box object with square base.

        Parameters
        ----------
        params : InertialParameters
            Inertial parameters of the box.
        w : float, positive
            Width of the square base.
        """
        assert w > 0, "Base width must be positive."
        r = 0.5 * w
        vertices = np.array([[r, r, 0], [r, -r, 0], [-r, r, 0], [-r, -r, 0]])
        return cls(params=params, contacts=vertices)
