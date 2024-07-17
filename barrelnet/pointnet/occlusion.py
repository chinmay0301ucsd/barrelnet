### Creating partially occluded point cloud. 
import mitsuba as mi
import numpy as np 
import roma
import torch
import drjit as dr
from matplotlib import pyplot as plt
import pyrender
import trimesh
import os 

os.environ['PYOPENGL_PLATFORM'] = 'egl'
mi.set_variant('cuda_ad_rgb')

