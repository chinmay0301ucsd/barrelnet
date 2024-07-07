import torch
from tqdm import tqdm
import roma
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def pad_point_cloud(point_cloud, max_points=1000):
	"""
	Pads the input point cloud with zeros to ensure it has exactly max_points.
	Args:
		point_cloud (torch.Tensor): The input point cloud tensor of shape (N, C).
		max_points (int): The desired number of points in the output tensor.
	Returns:
		torch.Tensor: The padded point cloud tensor of shape (max_points, C).
	"""
	num_points, num_channels = point_cloud.shape
	padded_cloud = torch.zeros((max_points, num_channels), dtype=point_cloud.dtype, device=point_cloud.device)
	length = min(num_points, max_points)
	perm_idx = torch.randperm(point_cloud.shape[0])
	pc = point_cloud[perm_idx]
	padded_cloud[:length, :] = pc[:length, :]
	return padded_cloud

def generate_cylinder_pts(radius, height, sections=64, num_pts=1000):
    """
    Create a capped cylinder using trimesh.
    
    Args:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        sections (int): Number of sections for the cylinder's roundness.
        
    Returns:
       points (torch.tensor) : [num_pts, 3]
    """
    # Create the capped cylinder with cap=True
    capped_cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections, cap=True)
    points, _ = trimesh.sample.sample_surface(capped_cylinder, num_pts)
    points = torch.from_numpy(points).float().cuda()
    return points

def rotate_to_axis(point_cloud, axis, device='cuda'):
    """
    Rotate the point cloud to align its z-axis with the given axis using PyTorch.
    
    Args:
        point_cloud (torch.Tensor): Input point cloud of shape (N, 3).
        axis (torch.Tensor): Target axis to align with, should be of shape (3,).
        device (str): Device to perform the computation ('cpu' or 'cuda').
    Returns:
        torch.Tensor: Rotated point cloud of shape (N, 3).
    """
    ## TODO: make this batched, to rotate the point cloud to different configurations at once. 
    # Normalize the axis
    axis = axis / torch.linalg.norm(axis)
    z_axis = torch.tensor([0., 0., 1.], device='cuda')
    
    # Compute the rotation axis and angle
    rotation_vector = torch.cross(z_axis, axis)
    angle = torch.arccos(torch.dot(z_axis, axis))
    
    # Convert rotation vector and angle to a PyTorch tensor
    rotation_vector = torch.tensor(rotation_vector, dtype=torch.float32, device=device)
    angle = torch.tensor(angle, dtype=torch.float32, device=device)
    
    # Create the rotation matrix using torch-roma
    rotation_matrix = roma.rotvec_to_rotmat(rotation_vector * angle)
    
    # Apply rotation to the point cloud
    point_cloud_rotated = torch.matmul(point_cloud, rotation_matrix.T)
    
    return point_cloud_rotated

def prepare_point_cloud(points, normal, hr_ratio_range=[1/4, 1/2], max_burial_percent=0.7):
    ## TODO : Maybe Subsume this in the point cloud class 
    """
        points (torch.tensor) [N,3] : sampled surface of full cylinder in standard XYZ coordinate system. 
        normal (torch.tensor) [3,]: normal vector to rotate 
        hr_ratio_range (list) : [min, max] range of the height ratio
    Returns:
		valid_pts (torch.tensor) : Rotated point cloud, and clipped so that only points above ocean floor are visible 
		scale (float) - factor by which radius is to be scaled, for easier regression by pointnet.
		radius (float) - randomly sampled radius of the cylinder . It's unscaled 
		burial_z (float) - Relative Displacement of Barrel Center along Z axis inside / outside the ocean floor w.r.t height of the cylinder
    """    
    point_cloud = points.clone().to(points.device)
    radius = hr_ratio_range[0] + torch.rand(1)[0]* (hr_ratio_range[1] - hr_ratio_range[0]) 
    point_cloud[:,:2] *= radius # Scaling height of the cylinder to be 2.5 - 3.5 times the radius 
    point_cloud_rot = rotate_to_axis(point_cloud, normal)
    
    ## TODO: Double Check this
    ## Total Z extent spanned by the barrel after rotation. 
    z_range = 1.0 # z_range is going to dot product of normal_range and Z axis, since height=1.0 
    burial_z = max_burial_percent * (torch.rand(1)[0] - 0.5) # Relative Displacement of Barrel Center along Z axis inside / outside the ocean floor w.r.t height of the cylinder
    burial_offset = burial_z * z_range
    point_cloud_rot[:,-1] += burial_offset
    valid_pts = point_cloud_rot[point_cloud_rot[:,-1] > 0]
    
    ## TODO : Add code here to compute burial fraction, for synthetic data. 
    return valid_pts, radius, burial_z

def normalize_pc(valid_pts):
    """
    Normalize the point cloud to feed in to point net. Also estimate the scaling factor. 
    Args:
		valid_pts (torch.tensor) N,C - Point Cloud 
    Returns:
		pts (torch.tensor) N,C normalized Point Cloud 
		scale (float) - value with which to divide the actual radius, for input to network.
    """
    ## Point Cloud Prep for Feeding it to pointnet 
    ## Estimating the scale that needs to be given to the point cloud as input
    pts = valid_pts - valid_pts.mean(dim=1, keepdim=True)
    scale = torch.max(torch.linalg.norm(pts, dim=-1, keepdim=True), dim=0).values.item()
    pts = pts / scale
    return pts, scale 

class CylinderData(Dataset):
	def __init__(self, num_poses=10000, num_surface_samples=3000, max_points=1000, transform=None, max_burial_percent=0.7):
		"""
		Args:
			num_poses : number of axis vectors to sample 
			num_surface_samples : number of points on the surface of the cylinder to sample
			max_points : maximum number of points in a batch of the point cloud
			transform (callable, optional) : Optional transform to be applied
				on a sample.
		"""
		self.points = generate_cylinder_pts(1.0, 1.0, num_pts = num_surface_samples)
		self.max_points = max_points
		self.normals = self.sample_normals(num_poses)
		self.radii = []
		self.burial_offsets = []
		self.scales = []
		self.pts = []

		## TODO: Later remove this for loop, and make this one shot.
		for i in tqdm(range(num_poses)):
			valid_pts, radius, burial_offset = prepare_point_cloud(self.points, self.normals[i], max_burial_percent=max_burial_percent)
			pts, scale = normalize_pc(valid_pts)
			self.radii.append(radius)
			self.burial_offsets.append(burial_offset)
			self.scales.append(scale)
			self.pts.append(pts)

		self.radii = torch.stack(self.radii)
		self.burial_offsets = torch.stack(self.burial_offsets)
		self.scales = torch.tensor(self.scales)

		self.transform = transform ## Adding noise etc

	def pad_point_cloud(self, point_cloud):
		"""
		Pads the input point cloud with zeros to ensure it has exactly max_points.
		Args:
			point_cloud (torch.Tensor): The input point cloud tensor of shape (N, C).
			max_points (int): The desired number of points in the output tensor.
		Returns:
			torch.Tensor: The padded point cloud tensor of shape (max_points, C).
		"""
		num_points, num_channels = point_cloud.shape
		padded_cloud = torch.zeros((self.max_points, num_channels), dtype=point_cloud.dtype, device=point_cloud.device)
		length = min(num_points, self.max_points)
		perm_idx = torch.randperm(point_cloud.shape[0])
		pc = point_cloud[perm_idx]
		padded_cloud[:length, :] = pc[:length, :]
		return padded_cloud

	def sample_normals(self, num_samples, device='cuda'):
		"""
		Sample random unit vectors uniformly distributed on the upper hemisphere.
		
		Args:
			num_samples (int): Number of unit vectors to sample.
			device (str): Device to perform the computation ('cpu' or 'cuda').
			
		Returns:
			torch.Tensor: Tensor of shape (num_samples, 3) containing the sampled unit vectors.
		"""
		phi = torch.rand(num_samples, device=device) * 2 * torch.pi
		theta = torch.acos(torch.rand(num_samples, device=device))

		x = torch.sin(theta) * torch.cos(phi)
		y = torch.sin(theta) * torch.sin(phi)
		z = torch.cos(theta)

		return torch.stack((x, y, z), dim=1)
	
	def __len__(self):
		"""Returns the total number of samples"""
		return len(self.pts)

	def __getitem__(self, idx):
		"""
		Args:
			idx (int): Index

		Returns:
			sample: (data, label) pair for the given index
		"""
		pts = self.pad_point_cloud(self.pts[idx])
		sample = {'pts': pts.permute(1,0),
            	  'scale_gt': self.scales[idx],
               	  'radius_gt': self.radii[idx],
                  'axis_vec': self.normals[idx],
                  'burial_z': self.burial_offsets[idx]}

		if self.transform:
			sample['pts'] = self.transform(sample['pts'])

		return sample
