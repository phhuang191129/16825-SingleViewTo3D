from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            pass
            # TODO:
            self.decoder =  nn.Sequential(
                nn.Linear(512, 32*4*4*4),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (32, 4, 4, 4)),
                nn.Conv3d(32, 16, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(16, 8, kernel_size=4, stride=4, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1, bias=True),
            )


        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                # nn.Linear(128, 128),
                # nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
                # nn.ReLU(inplace=True),
                nn.Linear(128, args.n_points * 3),
            )
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # use the arch similar to the sample architecture from the L06 slides
            self.decoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(256, mesh_pred.verts_packed().shape[0] * 3), #verts_packed: tensor of vertices of shape (sum(V_n), 3).
                nn.Tanh()
            )
            self.offset_scale = 0.2
        elif args.type == "implicit":
            self.grid_size = 32
            min_value = -1
            max_value = 1
            X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, self.grid_size)] * 3)
            self.voxel_coords = torch.stack([X, Y, Z], dim=-1).view(-1, 3).expand(args.batch_size,-1,3)
            self.register_buffer("voxel_coords", self.voxel_coords)
            # use simple MLP
            self.cbn_layer1 = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
            )

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat).view(B, 1,32, 32, 32)
            if not self.training:
                # we use bce loss with logits for loss calculation due to numerical stability so when eval, we'll add sigmoid here since there's no
                # sigmoid in evaluate function 
                voxels_pred = torch.sigmoid(voxels_pred)
            # From AtlasNet paper : The architecture of our decoder is 4 fully-connected layers
            # of size 1024, 512, 256, 128 with ReLU non-linearities on
            # the first three layers and tanh on the final output layer.
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat).view(B, self.n_point, 3)
            # make the bound around ground_truth_3d.reshape(-1, 3).abs().quantile(0.999, dim=0).max() ~= 0.57
            pointclouds_pred = 1.05*0.55 * torch.tanh(pointclouds_pred)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred =  self.decoder(encoded_feat).view(B, -1, 3)*self.offset_scale  
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred

        elif args.type == "implicit":
            # B,512 -> B, N, 512
            expanded_encoded_feat = encoded_feat.expand(encoded_feat.shape[0], self.voxel_coords.shape[1], encoded_feat.shape[1])
            encoded_feat_xyz = torch.cat([expanded_encoded_feat, self.voxel_coords],dim=2)
            implicit_pred = self.decoder(encoded_feat_xyz).view(B, 1,self.grid_size, self.grid_size, self.grid_size)
            if not self.training:
                implicit_pred = torch.sigmoid(implicit_pred)
            return implicit_pred

