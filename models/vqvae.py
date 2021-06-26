# The code is mostly borrow from the class 
# Berkeley's CS294-158 Deep Unsupervised Learning (https://github.com/rll/deepul)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)


class Quantize(nn.Module):
    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size,1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        b, c, h, w = z.shape
        weight = self.embedding.weight

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.max(-distances, dim=1)[1]
        encoding_indices = encoding_indices.view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()

        return quantized, (quantized - z).detach() + z, encoding_indices


class LargeVectorQuantizedVAE(nn.Module):
    def __init__(self, code_dim=256, code_size=128):
        super().__init__()
        self.code_size = code_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.codebook = Quantize(code_size, code_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode_code(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

    def decode_code(self, latents):
        with torch.no_grad():
            latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents).permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5

    def forward(self, x):
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)

        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)
        recon_loss = F.mse_loss(x_tilde, x)
        diff = diff1 + diff2
        loss = recon_loss + diff
        return x_tilde, OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)


class VectorQuantizedVAE_Encode(nn.Module):
    def __init__(self, code_dim, code_size):
        super().__init__()
        self.code_size = code_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.codebook = Quantize(code_size, code_dim)    

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices


class LargeVectorQuantizedVAE_Encode(nn.Module):
    def __init__(self, code_dim, code_size):
        super().__init__()
        self.code_size = code_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.codebook = Quantize(code_size, code_dim)

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

# class VectorQuantizedVAE(nn.Module):
#     def __init__(self, code_dim=256, code_size=128):
#         super().__init__()
#         self.code_size = code_size

#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 256, 4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.Conv2d(256, 256, 4, stride=2, padding=1),
#             ResidualBlock(256),
#             ResidualBlock(256),
#         )

#         self.codebook = Quantize(code_size, code_dim)

#         self.decoder = nn.Sequential(
#             ResidualBlock(256),
#             ResidualBlock(256),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
#             nn.Tanh(),
#         )
        
#     def encode_code(self, x):
#         with torch.no_grad():
#             #x = 2 * x - 1
#             z = self.encoder(x)
#             indices = self.codebook(z)[2]
#             return indices

#     def decode_code(self, latents):
#         with torch.no_grad():
#             latents = self.codebook.embedding(latents).permute(0, 3, 1, 2).contiguous()
#             return self.decoder(latents).permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5

#     def forward(self, x):
#         #x = 2 * x - 1
#         z = self.encoder(x)
#         e, e_st, _ = self.codebook(z)
#         x_tilde = self.decoder(e_st)

#         diff1 = torch.mean((z - e.detach()) ** 2)
#         diff2 = torch.mean((e - z.detach()) ** 2)
#         recon_loss = F.mse_loss(x_tilde, x)
#         diff = diff1 + diff2
#         loss = recon_loss + diff
#         return x_tilde, OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)