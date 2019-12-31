import numpy as np
from numpy.linalg import norm
import math
from math import cos, sin


def lattice_constants_to_cartesian_vectors(lattice_constants):
    """
    Calculate a,b,c vectors in cartesian system from lattice constants.
    :param lattice_constants: a,b,c,alpha,beta,gamma lattice constants.
    :return: a, b, c vector
    """
    lattice_constants = np.array(lattice_constants)
    if lattice_constants.shape != (6,):
        raise ValueError('Lattice constants must be 1d array with 6 elements.')
    a, b, c = lattice_constants[:3]
    alpha, beta, gamma = np.deg2rad(lattice_constants[3:])
    av = np.array([a, 0, 0], dtype=float)
    bv = np.array([b * np.cos(gamma), b * np.sin(gamma), 0], dtype=float)
    # calculate vector c
    x = np.cos(beta)
    y = (np.cos(alpha) - x * np.cos(gamma)) / np.sin(gamma)
    z = np.sqrt(1. - x**2. - y**2.)
    cv = np.array([x, y, z], dtype=float)
    cv /= norm(cv)
    cv *= c

    return av, bv, cv


def calc_transform_matrix(lattice_constants):
    """
    Calculate transform matrix from lattice constants.
    :param lattice_constants: a,b,c,alpha,beta,gamma lattice constants in
                              angstrom and degree.
    :return: transform matrix A = [a*, b*, c*]
    """
    av, bv, cv = lattice_constants_to_cartesian_vectors(lattice_constants)
    a_star = (np.cross(bv, cv)) / (np.cross(bv, cv).dot(av))
    b_star = (np.cross(cv, av)) / (np.cross(cv, av).dot(bv))
    c_star = (np.cross(av, bv)) / (np.cross(av, bv).dot(cv))
    A = np.zeros((3, 3), dtype=np.float64)  # transform matrix
    A[:, 0] = a_star
    A[:, 1] = b_star
    A[:, 2] = c_star
    return A


def calc_wavelength(photon_energy):
    """Convert photon energy to wavelength.
    
    Args:
        photon_energy (float): photon energy in eV.
    """
    h = 4.135667662E-15  # Planck constant in eV*s
    c = 2.99792458E8  # light speed in m/s
    return (h * c) / photon_energy


def axis_angle_to_rotation_matrix(axis, angle):
    """Calculate rotation matrix from axis/angle form.
    
    Args:
        axis (1d-ndarray): 3-element axis vector.
        angle (float): angle in rad.
    
    Returns:
        2d-ndarray: Rotation matrix.
    """
    if norm(axis) < 1E-99:  # dummy case
        return np.identity(3)
    x, y, z = axis / norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    R = [[c+x**2.*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
         [y*x*(1-c)+z*s, c+y**2.*(1-c), y*z*(1-c)-x*s],
         [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z**2.*(1-c)]]
    return np.array(R)


def build_grid_image(dim0, dim1):
    idx, idy = np.indices((dim0, dim1))
    image = np.zeros((dim0, dim1))
    image[(idx+idy) % 2 == 0] = 1
    return image


def euler_angles_to_rotation_matrix(euler_angles):
    """Convert XYZ Euler angles to rotation matrix
    """
    t1, t2, t3 = np.deg2rad(euler_angles)
    rm = np.array([
        [cos(t2)*cos(t3), cos(t2)*sin(t3), -sin(t2)],
        [sin(t1)*sin(t2)*cos(t3) - cos(t1)*sin(t3), sin(t1) *
         sin(t2)*sin(t3)+cos(t1)*cos(t3), sin(t1)*cos(t2)],
        [cos(t1)*sin(t2)*cos(t3)+sin(t1)*sin(t3), cos(t1) *
         sin(t2)*sin(t3)-sin(t1)*cos(t3), cos(t1)*cos(t2)]
    ])
    return rm


def rotation_matrix_to_euler_angles(rm):
    t1 = math.atan2(rm[1, 2], rm[2, 2])
    c2 = math.sqrt(rm[0, 0]**2 + rm[0, 1]**2)
    t2 = math.atan2(-rm[0, 2], c2)
    s1, c1 = sin(t1), cos(t1)
    t3 = math.atan2(s1*rm[2, 0]-c1*rm[1, 0], c1*rm[1, 1]-s1*rm[2, 1])
    return np.rad2deg(t1), np.rad2deg(t2), np.rad2deg(t3)
