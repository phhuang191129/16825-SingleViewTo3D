import argparse
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from utils import get_device, get_mesh_renderer, get_points_renderer
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds

#  python fit_data.py --type 'vox'

NUM_ANGLE = 24
DIST = 2
ELEV = 30

def render_voxel_to_360gif(voxels, image_size=256, output_path="voxel.gif", device=None):
    if device is None:
        device = get_device()
    # build voxel grid 

    voxels = voxels.squeeze().detach().cpu().numpy()
    # vertices, faces = mcubes.marching_cubes(grid, isovalue=0)
    vertices, faces = mcubes.marching_cubes(voxels, isovalue=0.5)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    vertices = (vertices / voxels.shape[0]) -0.5 #(max_value - min_value) + min_value
    # textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    # textures = (torch.ones_like(vertices) * torch.tensor([0.7, 0.7, 0.7])).to(device)
    texture_rgb = vertices.clone()
    texture_rgb = (texture_rgb - texture_rgb.min()) / (texture_rgb.max() - texture_rgb.min())
    texture_rgb /= texture_rgb.norm(dim=1, keepdim=True)
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb.unsqueeze(0)).to(device)

    mesh= pytorch3d.structures.Meshes([vertices], [faces], textures=textures)

    num_angle = NUM_ANGLE
    dist = DIST
    angles = np.linspace(-180, 180, num_angle, endpoint=True)

    rotations, translations = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=ELEV, azim=angles, device=device)

    full_view_cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=rotations, T=translations, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -dist]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    images = renderer(mesh.to(device).extend(num_angle), cameras=full_view_cameras, lights=lights).detach().cpu().numpy()[..., :3].clip(0, 1)
    images = [(img * 255).astype(np.uint8) for img in images]
    imageio.mimsave(output_path, images, duration=1000//15,loop=0)
    print(f'Saved gif to {output_path}')




def render_pointcloud_360gif(point_cloud, output_path, image_size,color=[0.7,0.7,1], background_color=[1, 1, 1], device=None):
    if device is None :
        device = get_device()
    if isinstance(color, list):
        feats = torch.tensor(color,device=device).expand((point_cloud.squeeze(0).shape[0], 3))
    else:
        feats = color.to(device).squeeze(0)
    point_cloud = Pointclouds(point_cloud, features = [feats])
    num_angle = NUM_ANGLE
    dist = DIST
    angles = np.linspace(-180, 180, num_angle, endpoint=True)

    rotations, translations = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=ELEV, azim=angles, device=device)

    full_view_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=rotations, T=translations, device=device
    )

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -dist]], device=device)
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color, device=device
    )
    images = renderer(point_cloud.to(device).extend(num_angle), cameras=full_view_cameras, lights=lights).detach().cpu().numpy()[..., :3]
    images = [(img * 255).astype(np.uint8) for img in images]
    imageio.mimsave(output_path, images, duration=1000//15,loop=0)
    print(f'Saved gif to {output_path}')





def render_mesh360gif(mesh,output_path, image_size,color=[0.7, 0.7, 1], device=None):
    if device is None:
        device = get_device()
    mesh.to(device)

    vertices = mesh.verts_packed().clone()
    textures = (torch.ones_like(vertices) * torch.tensor(color)).to(device)
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))
    mesh.textures = textures

    num_angle = NUM_ANGLE
    dist = DIST
    angles = np.linspace(-180, 180, num_angle, endpoint=True)

    rotations, translations = pytorch3d.renderer.look_at_view_transform(dist=dist, elev=ELEV, azim=angles, device=device)

    full_view_cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=rotations, T=translations, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -dist]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    images = renderer(mesh.to(device).extend(num_angle), cameras=full_view_cameras, lights=lights).detach().cpu().numpy()[..., :3].clip(0, 1)
    images = [(img * 255).astype(np.uint8) for img in images]
    imageio.mimsave(output_path, images, duration=1000//15,loop=0)
    print(f'Saved gif to {output_path}')    


def distance_map(source, target,kind):
    # reference to figure 4 in https://arxiv.org/pdf/1811.10943#page=11.51
    if kind =='mesh':
        source = sample_points_from_meshes(source, 5000)
        target = sample_points_from_meshes(target, 5000)
    
    knn_dist_p1, knn_idx_p1, knn_in_p2 = knn_points(source, target, K=1, return_nn=True)
    knn_dist_p1 = knn_dist_p1**0.5
    normalized_dist = (knn_dist_p1-knn_dist_p1.mean())/(knn_dist_p1.max()-knn_dist_p1.min()+1e-8)
    print('normalized_dist',normalized_dist.shape)
    return normalized_dist
    
    


