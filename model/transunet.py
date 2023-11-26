import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.ViViT.module import Attention, PreNorm, FeedForward

class TransAnomaly(nn.Module):
    def __init__(self, batch_size, num_frames):
        super(TransAnomaly, self).__init__()
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.channels_1 = 64 
        self.channels_2 = 128
        self.channels_3 = 256
        self.channels_4 = 512

        self.contracting_11 = self.conv_block(in_channels=3, out_channels=self.channels_1)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=self.channels_1, out_channels=self.channels_2)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.contracting_31 = self.conv_block(in_channels=self.channels_2, out_channels=self.channels_3)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.contracting_41 = self.conv_block(in_channels=self.channels_3, out_channels=self.channels_4)

        self.residual_14 = nn.Conv2d(in_channels=self.channels_1*self.num_frames, out_channels=self.channels_1, kernel_size=3, stride=1, padding=1)
        self.residual_23 = nn.Conv2d(in_channels=self.channels_2*self.num_frames, out_channels=self.channels_2, kernel_size=3, stride=1, padding=1)
        self.residual_32 = nn.Conv2d(in_channels=self.channels_3*self.num_frames, out_channels=self.channels_3, kernel_size=3, stride=1, padding=1)
        self.residual_41 = nn.Conv2d(in_channels=self.channels_4*self.num_frames, out_channels=self.channels_4, kernel_size=3, stride=1, padding=1)

        self.middle = ViViT(image_size=32, patch_size=2, num_frames=self.num_frames, in_channels=512)
        
        self.expansive_11 = nn.ConvTranspose2d(in_channels=self.channels_4, out_channels=self.channels_4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=self.channels_4*2, out_channels=self.channels_4) #(512,32,32)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=self.channels_4, out_channels=self.channels_3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=self.channels_3*2, out_channels=self.channels_3)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=self.channels_3, out_channels=self.channels_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=self.channels_2*2, out_channels=self.channels_2)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=self.channels_2, out_channels=self.channels_1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=self.channels_1*2, out_channels=self.channels_1)
        self.output = nn.Conv2d(in_channels=self.channels_1, out_channels=3, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block


    def forward(self, frames):

        tmp_frames = rearrange(frames, 'b t c h w -> (b t) c h w')
        
        contracting_11_out = self.contracting_11(tmp_frames)

        contracting_12_out = self.contracting_12(contracting_11_out)
        contracting_21_out = self.contracting_21(contracting_12_out)
        contracting_22_out = self.contracting_22(contracting_21_out)
        contracting_31_out = self.contracting_31(contracting_22_out)
        contracting_32_out = self.contracting_32(contracting_31_out)
        contracting_41_out = self.contracting_41(contracting_32_out)

        vivit_input = rearrange(contracting_41_out, '(b t) c h w -> b t c h w', b=self.batch_size)

        middle_out = self.middle(vivit_input)

        residual_14_out = rearrange(contracting_11_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_14_out = self.residual_14(residual_14_out)
        residual_23_out = rearrange(contracting_21_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_23_out = self.residual_23(residual_23_out)
        residual_32_out = rearrange(contracting_31_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_32_out = self.residual_32(residual_32_out)
        residual_41_out = rearrange(contracting_41_out, '(b t) c h w -> b (t c) h w', b=self.batch_size)
        residual_41_out = self.residual_41(residual_41_out)

        expansive_11_out = self.expansive_11(middle_out)
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, residual_41_out), dim=1)) 
        expansive_21_out = self.expansive_21(expansive_12_out) 
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, residual_32_out), dim=1)) 
        expansive_31_out = self.expansive_31(expansive_22_out) 
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, residual_23_out), dim=1)) 
        expansive_41_out = self.expansive_41(expansive_32_out)
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, residual_14_out), dim=1)) 
        output_out = self.output(expansive_42_out) 

        return torch.tanh(output_out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, dim = 512, depth = 4, heads = 3, pool = 'cls', in_channels = 512, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()

        self.image_size = image_size 
        self.patch_size = patch_size
        self.num_frames = num_frames 
        self.in_channels = in_channels 
        self.dim = dim 

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)' 


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_size // patch_size) ** 2 
        self.patch_dim = in_channels * patch_size ** 2 

        self.to_patch_embedding = nn.Sequential(

            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(self.patch_dim, self.dim),
            Rearrange('b t n d -> b n (d t)') 
        )

        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, (self.num_frames + 1)*self.dim))

        self.temporal_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.temporal_transformer = Transformer(dim * (self.num_frames + 1), depth, heads, dim_head, dim*scale_dim, dropout) 

        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))

        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout) 

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool


    def forward(self, x):

        x = self.to_patch_embedding(x)

        b, n, d = x.shape

        pred_temporal_tokens = repeat(self.temporal_token, '() () d -> b n d', b=b, n=n)

        x = torch.cat((pred_temporal_tokens, x), dim=2)
        x += self.temporal_pos_embedding[:,:,:(d+self.dim)]
        x = self.dropout(x)

        x = self.temporal_transformer(x)

        x = x[:,:,:self.dim]

        x += self.spatial_pos_embedding


        x = self.space_transformer(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.image_size//2,w=self.image_size//2)

        return x