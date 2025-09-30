import torch
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	# if voxel_tgt.dim() == 5:
	# 	voxel_tgt = voxel_tgt.squeeze(1)
	loss = torch.nn.functional.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	knn_dist_p1, knn_idx_p1, knn_in_p2 = knn_points(point_cloud_src, point_cloud_tgt, K=1, return_nn=True)
	knn_dist_p2, knn_idx_p2, knn_in_p1 = knn_points(point_cloud_tgt, point_cloud_src, K=1, return_nn=True)
	loss_chamfer = (sum(knn_dist_p1.squeeze(2)) +sum(knn_dist_p2.squeeze(2))).mean()
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")
	return loss_laplacian


def implicit_loss(pred, target):
	loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
	return loss