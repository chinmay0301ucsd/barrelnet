import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def random_cylinder_vol(x1, x2, r, npoints, sigma=0):
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
    xyzs += np.random.normal(0, sigma, xyzs.shape)
    return xyzs


def random_cylinder_surf(x1, x2, r, npoints, sigma=0, includecap=True):
    """
    Generates uniformly distributed points within the surface of a defined cylinder
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    axis = x2 - x1
    h = np.linalg.norm(axis)
    axis = axis / h
    axnull = scipy.linalg.null_space(np.array([axis]))
    axnull1 = axnull[:, 0]
    axnull2 = axnull[:, 1]
    # top and bottom caps
    cap_area = 2 * np.pi * r**2
    side_area = h * np.pi * r * 2
    # for a single cap
    if includecap:
        ncap = int((cap_area / (cap_area + side_area)) * npoints / 2)
    else:
        ncap = 0
    nside = npoints - 2 * ncap
    # sqrt radius to get uniform distribution
    rand_r = np.full(npoints, r, dtype=float)
    rand_r[:2 * ncap] = r * np.sqrt(np.random.random(2 * ncap))
    rand_theta = np.random.random(npoints) * 2 * np.pi
    rand_h = np.random.random(npoints) * h
    rand_h[:ncap] = 0.0
    rand_h[ncap:2 * ncap] = h

    cosval = np.tile(axnull1, (npoints, 1)) * rand_r[..., None] * np.cos(rand_theta)[..., None]
    sinval = np.tile(axnull2, (npoints, 1)) * rand_r[..., None] * np.sin(rand_theta)[..., None]
    xyzs = cosval + sinval + np.tile(x1, (npoints, 1)) + rand_h[..., None] * np.tile(axis, (npoints, 1))
    xyzs += np.random.normal(0, sigma, xyzs.shape)
    return xyzs


def classify_points(points, a, b, c, d):
    """Function to classify points based on the plane"""
    results = np.array([a, b, c]) @ points.T + d
    above_plane = results >= 0
    return (above_plane).sum(), (~above_plane).sum()


def monte_carlo_volume_ratio(n_points, x1, x2, r, a, b, c, d):
    """Monte Carlo method to estimate volume ratio"""
    points = random_cylinder_vol(x1, x2, r, n_points)
    above_plane, below_plane = classify_points(points, a, b, c, d)
    ratio = above_plane / (above_plane + below_plane)
    return ratio


def random_unitvec3(n=1):
    """
    Generates uniformly distributed 3D unit vector.
    
    Args:
        n: number of vectors to generate
        
    Returns:
        nx3 vector
    """
    # apparently the standard multivariate normal distribution
    # is rotation invariant, so its distributed uniformly
    unnormxyzs = np.random.normal(0.0, 1.0, size=(n, 3))
    xyzs = unnormxyzs / np.linalg.norm(unnormxyzs, axis=1)[..., None]
    return xyzs


def generate_oriented_barrel(r, h, npoints, sigma=0, zlims=None, includecap=True):
    """Uniformly random oriented barrel, with random height from cylinder centroid z=0"""
    if zlims is None:
        zlims = [0, 0]
    cylax = random_unitvec3(1)[0]
    # force vector to point up
    if cylax[2] < 0:
        cylax = -cylax
    heightchange = np.random.uniform(zlims[0], zlims[1])
    bar_bot = -h * cylax / 2 + heightchange
    bar_top = h * cylax / 2 + heightchange

    points_all = random_cylinder_surf(bar_bot, bar_top, r, npoints, sigma=sigma, includecap=includecap)

    # filter out buried points
    points = points_all[points_all[:, 2] >= 0, :]
    return points, points_all, cylax, heightchange


def get_cyl_endpoints(cylax, h, offset, axidx=2):
    """
    Get axis endpoints of a cylinder based on centroid with given axis up at 0.
    
    axidx is 0 (x), 1 (y), 2 (z) up
    
    Offset z input.
    """
    cylax = cylax / np.linalg.norm(cylax)
    # force vector to point up
    if cylax[axidx] < 0:
        cylax = -cylax
    x1 = -cylax * (h / 2)
    x1[axidx] += offset
    x2 = cylax * (h / 2)
    x2[axidx] += offset
    return x1, x2


def get_cylinder_surf(x1, x2, r, nt=100, nv=50):
    """
    parametrize the cylinder of radius r, and endpoints x1, x2.
    
    Args:
        nt: number of angles
        nv: number of heights
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    axis = x2 - x1
    h = np.linalg.norm(axis)
    axis = axis / h
    axnull = scipy.linalg.null_space(np.array([axis]))
    axnull1 = axnull[:, 0]
    axnull2 = axnull[:, 1]

    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(0, h, nv )
    theta, v = np.meshgrid(theta, v)
    gridshape = theta.shape
    gridsize = theta.size
    cosval = np.tile(axnull1, (gridsize, 1)) * r * np.cos(theta.reshape(-1))[..., None]
    sinval = np.tile(axnull2, (gridsize, 1)) * r * np.sin(theta.reshape(-1))[..., None]
    xyzs = cosval + sinval + np.tile(x1, (gridsize, 1)) + v.reshape(-1)[..., None] * np.tile(axis, (gridsize, 1))
    x = xyzs[:, 0].reshape(gridshape)
    y = xyzs[:, 1].reshape(gridshape)
    z = xyzs[:, 2].reshape(gridshape)
    return x, y, z


def plot_cylinder_and_plane(r, h, a, b, c, d):
    """Function to plot the cylinder and the plane"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotting the cylinder
    z = np.linspace(0, h, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, z = np.meshgrid(theta, z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot_surface(x, y, z, color='cyan', alpha=0.5)

    # Creating a grid for the plane
    xx, yy = np.meshgrid(np.linspace(-r, r, 100), np.linspace(-r, r, 100))
    zz = (-d - a * xx - b * yy) / c

    # Plotting the plane
    ax.plot_surface(xx, yy, zz, color='orange', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Setting the limits for better visualization
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([0, h])
    
    plt.show()


if __name__ == "__main__":
    # Parameters of the cylinder and plane
    r = 1  # radius
    h = 2  # height
    a, b, c, d = 1, 1, 1, -1  # plane equation ax + by + cz + d = 0

    # Number of random points to sample
    n_points = 1000000

    # Calculate the volume ratio
    volume_ratio = monte_carlo_volume_ratio(n_points, r, h, a, b, c, d)
    print("Estimated Volume Ratio:", volume_ratio)

    # Plotting the cylinder and plane
    plot_cylinder_and_plane(r, h, a, b, c, d)
