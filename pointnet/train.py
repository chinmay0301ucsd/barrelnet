import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import uuid
import roma
from data import generate_cylinder_pts, prepare_point_cloud, normalize_pc, CylinderData
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from barrelnet import BarrelNet


def compute_loss(sample, radius_pred, zshift_pred, axis_pred, use_radius_loss=False, use_axis_loss=True, use_loss_zshift=True):
    """ Compute loss on the predictions of pointnet """
    assert use_axis_loss or use_radius_loss, "Atleast one of the losses should be used"
    loss_axis = (1 - F.cosine_similarity(sample['axis_vec'], axis_pred, dim=1)).mean()
    scale = sample['scale_gt']
    loss_radius = F.mse_loss(radius_pred, sample['radius_gt'].cuda() / scale.cuda())
    loss_zshift = F.mse_loss(zshift_pred, sample['burial_z'].cuda())
    loss = 0.0 
    if use_radius_loss:
        loss = loss + loss_radius
    if use_axis_loss:
        loss = loss + loss_axis
    if use_loss_zshift:
        loss = loss + loss_zshift
    return loss, loss_radius, loss_axis, loss_zshift

def train(model, train_loader, optimizer, scheduler, writer, num_epochs=5000, save_epoch=1000, test_epoch=1000, save_dir='weights/r0'):
    """ 
	Train the model for num_epochs
	model : BarrelNet Model instance 
    """
    os.makedirs(save_dir, exist_ok=True)
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_loss_radius = 0.0
        running_loss_axis = 0.0
        running_loss_zshift = 0.0
        for i, sample in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            radius_pred, zshift_pred, axis_pred = model(sample['pts'])
            loss, loss_radius, loss_axis, loss_zshift = compute_loss(sample, radius_pred, zshift_pred, axis_pred, use_radius_loss=False, use_axis_loss=True)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_loss_radius += loss_radius.item()
            running_loss_axis += loss_axis.item()
            running_loss_zshift += loss_zshift.item()
        
        writer.add_scalar('Total_training_loss', running_loss / len(train_loader), epoch * len(train_loader))
        writer.add_scalar('Radius_training_loss', running_loss_radius / len(train_loader), epoch * len(train_loader))
        writer.add_scalar('Axis_training_loss', running_loss_axis / len(train_loader), epoch * len(train_loader))
        writer.add_scalar('Zshift_training_loss', running_loss_zshift / len(train_loader), epoch * len(train_loader))
        
        # Step the scheduler at the end of each epoch
        scheduler.step()
        if epoch % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, LR: {scheduler.get_last_lr()[0]}')

        if epoch % save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'pointnet_iter{epoch}.pth'))
        
        if epoch % test_epoch == 0:
            test(model, test_loader, writer, epoch)
            
def test(model, test_loader, writer, epoch=0):
    model.eval()
    running_acc_axis = 0.0
    accs_axis = []
    criterion_cosine = nn.CosineSimilarity(dim=1)
    for sample in tqdm(test_loader):
        with torch.no_grad():
            radius_pred, zshift_pred, axis_pred = model(sample['pts'])
            radius_pred = radius_pred * sample['scale_gt'].cuda()
            acc_axis = (1 + criterion_cosine(sample['axis_vec'], axis_pred).mean())/2
            running_acc_axis += acc_axis.item()
            accs_axis.append(acc_axis)
    accs = torch.tensor(accs_axis)
    writer.add_scalar("Axis Average Test Accuracy", running_acc_axis / len(test_loader), epoch)
    writer.add_scalar("Axis Best Test Accuracy", torch.max(accs).item(), epoch)
    writer.add_scalar("Axis Worst Test Accuracy", torch.min(accs).item() / len(test_loader), epoch)
      
        

if __name__=="__main__":
	dirname = str(uuid.uuid4()).replace('-', '')[:6]
	save_dir = os.path.join('logs', dirname, 'weights')
	writer = SummaryWriter(f'logs/{dirname}')
	train_data = CylinderData(num_poses=100)
	test_data = CylinderData(num_poses=10)

	train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
	test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

	pointnet = BarrelNet(k=5, normal_channel=False).cuda()
	pointnet.train()

	optimizer = optim.Adam(pointnet.parameters(), lr=0.00005)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)
 
	train(pointnet, train_loader, optimizer, scheduler, writer, save_dir=save_dir)