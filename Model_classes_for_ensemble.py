import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from vit_pytorch.vit import Transformer
from einops import repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        
        return out

class ConvTransformer(nn.Module):

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100, embed_dim=256, depth=8, num_heads=4):
        super(ConvTransformer, self).__init__()
        
        self.conv_layers = nn.Sequential(
            ResidualBlock(in_chans, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2)
        )
        
        self.num_patches = (img_size // (patch_size * 2)) ** 2
        self.patch_dim = 256
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        self.patch_embeddings = nn.Linear(self.patch_dim, self.embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_layers(x)
        
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, self.patch_dim)
        x = self.patch_embeddings(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings[:, :(self.num_patches + 1), :]
        
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x[0]
        
        x = self.mlp_head(x)
        
        return x
    
class CNNModel(nn.Module):
    def __init__(self, num_classes=100):
        super(CNNModel, self).__init__()
        # Initial Convolutional Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Residual Block 1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.residual_conv1 = nn.Conv2d(64, 128, kernel_size=1)  # Match the number of channels

        # Residual Block 2
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.residual_conv2 = nn.Conv2d(128, 256, kernel_size=1)  # Match the number of channels

        # Residual Block 3
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.residual_conv3 = nn.Conv2d(256, 512, kernel_size=1)  # Match the number of channels

        # Final Convolutional Block
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(1024)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Initial Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Residual Block 1
        identity = self.residual_conv1(x)  # Adjust the number of channels
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + identity  # Residual Connection
        x = self.pool2(x)

        # Residual Block 2
        identity = self.residual_conv2(x)  # Adjust the number of channels
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x + identity  # Residual Connection
        x = self.pool3(x)

        # Residual Block 3
        identity = self.residual_conv3(x)  # Adjust the number of channels
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = x + identity  # Residual Connection
        x = self.pool4(x)

        # Final Convolutional Block
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.pool5(x)

        # Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class ViT_modified(nn.Module):
    dim = 192
    depth = 14
    heads= 8
    mlp_dim = 2048
    num_channels = 3
    dim_head = 64
    dropout = 0.0
    emb_dropout = 0.0
    patch_size = 4
    img_size = 32
    num_classes = 100
    
    def __init__(self, *,
                        dim,
                        depth,
                        heads,
                        mlp_dim,
                        pool = 'cls',
                        num_classes = num_classes,
                        num_channels = num_channels,
                        dim_head = 64,
                        dropout = dropout,
                        emb_dropout = emb_dropout,
                        img_size  = img_size,
                        patch_size = patch_size
                        ):

        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_embed = PatchAndFlatten()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim)) 

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) 
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x, _ = self.patch_embed(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding 
        #x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        return self.mlp_head(x)
    
class PatchAndFlatten(nn.Module):
    dim = 192
    depth = 14 
    heads= 8 
    mlp_dim = 2048 
    num_channels = 3
    dim_head = 64 
    dropout = 0.0
    emb_dropout = 0.0
    patch_size = 4
    img_size = 32
    num_classes = 100

    def __init__(self, patch_dim=patch_size, dim=dim, c = num_channels):
        super().__init__()
        self.p = patch_dim
        self.unfold = torch.nn.Unfold(kernel_size = patch_dim, stride = patch_dim)
        self.linear_proj = nn.Linear(patch_dim*patch_dim*c, dim)

    def forward(self, img):

        bs, c, h, w = img.shape

        patches_unfold = self.unfold(img)

        patches_unfold = patches_unfold.permute(0,2,1)
        a_proj = self.linear_proj(patches_unfold)

        return a_proj, patches_unfold