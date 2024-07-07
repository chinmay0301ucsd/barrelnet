import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def random_cylinder(x1, x2, r, npoints):
    """
    Generates uniformly distributed points within the volume of a defined cylinder
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    axis = x2 - x1
    h = np.linalg.norm(axis)
    axis = axis / h
    axnull = scipy.linalg.null_space(np.array([axis]))
    axnull1 = axnull[:, 0]
    axnull2 = axnull[:, 1]
    # sqrt radius to get uniform distribution
    rand_r = r * np.sqrt(np.random.random(npoints))
    rand_theta = np.random.random(npoints) * 2 * np.pi
    rand_h = np.random.random(npoints) * h
    
    cosval = np.tile(axnull1, (npoints, 1)) * rand_r[..., None] * np.cos(rand_theta)[..., None]
    sinval = np.tile(axnull2, (npoints, 1)) * rand_r[..., None] * np.sin(rand_theta)[..., None]
    xyzs = cosval + sinval + np.tile(x1, (npoints, 1)) + rand_h[..., None] * np.tile(axis, (npoints, 1))
    return xyzs


def classify_points(points, a, b, c, d):
    """Function to classify points based on the plane"""
    results = np.array([a, b, c]) @ points.T + d
    above_plane = results >= 0
    return (above_plane).sum(), (~above_plane).sum()


def monte_carlo_volume_ratio(n_points, x1, x2, r, a, b, c, d):
    """Monte Carlo method to estimate volume ratio"""
    points = random_cylinder(x1, x2, r, n_points)
    above_plane, below_plane = classify_points(points, a, b, c, d)
    ratio = above_plane / (above_plane + below_plane)
    return ratio