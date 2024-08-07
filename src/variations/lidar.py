import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.embedding_size = mapping_size

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, in_dim, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs
        self.embedding_size = multires*in_dim*2 + in_dim

    def forward(self, x):
        # x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class Same(nn.Module):
    def __init__(self, in_dim) -> None:
        super().__init__()
        self.embedding_size = in_dim

    def forward(self, x):
        return x

class FFEncoder(nn.Module):
    def __init__(self, scale, embedding_size, dim=3):
        super().__init__()
        self.scale = scale
        self.embedding_size = embedding_size
        self.dim = dim
        self.bval = (torch.randn(self.embedding_size, 1) * self.scale * torch.pi * 2).to(device=torch.device('cuda'))
        self.bval = self.bval.squeeze()

    def forward(self, x):
        x = x[..., :self.dim, None]
        bval = self.bval[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
        x = x * bval
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = x.flatten(-2, -1)
        return x

class Decoder(nn.Module):
    def __init__(self,
                 depth=8,
                 width=258,
                 in_dim=3,
                 sdf_dim=128,
                 skips=[4],
                 multires=6,
                 embedder='none',
                 point_dim=3,
                 level = 2,
                 local_coord=False,
                 **kwargs) -> None:
        super().__init__()
        self.D = depth
        self.W = width
        self.skips = skips
        self.point_dim = point_dim
        gaussian_scale = 50.0
        embedding_size = 8
        self.embedder_name = embedder
        self.in_dim = in_dim
        self.level = level
        print('use ', self.level, 'level!!!')
        self.in_dim = self.in_dim * self.level
        print('in_dim = ', self.in_dim)
        if embedder == 'nerf':
            multires = 10
            self.pe = Nerf_positional_embedding(3, multires)
            self.embedding_size = multires * 3 * 2 + self.in_dim
        elif embedder == 'none':
            self.pe = Same(3)
            self.embedding_size = 3
        elif embedder == 'ffe':
            self.pe = FFEncoder(gaussian_scale, embedding_size, 3)
            self.embedding_size = self.in_dim + 3 * embedding_size * 2
        else:
            raise NotImplementedError("unknown positional encoder")
        
        print('embedding_size = ', self.embedding_size)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.embedding_size, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + self.pe.embedding_size, width) for i in range(depth-1)])
        self.sdf_out = nn.Linear(width, 1)

    def get_values(self, input):
        embeddings, xyz = torch.split(input, [self.in_dim, 3], dim=1)
        # 只对坐标编码
        x = self.pe(xyz)
        if self.embedder_name == 'none':
            h = embeddings
        else:
            h = torch.cat((embeddings, x), dim=1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # outputs = self.output_linear(h)
        # outputs[:, :3] = torch.sigmoid(outputs[:, :3])
        sdf_out = self.sdf_out(h)

        return sdf_out

    def forward(self, inputs):
        outputs = self.get_values(inputs)

        return {
            'sdf': outputs,
            # 'depth': outputs[:, 1]
        }
