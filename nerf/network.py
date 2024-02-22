import torch
import torch.nn as nn
import torch.nn.functional as F


def getPositionalEncoder(config):
    dim = config.model.encoding_dim
    return _Encoder(dim).func

class _Encoder():
    def __init__(self, dim):
        self.dim = dim
    
    def func(self, x):
        embedding = []
        embedding.append(x)
        for w in 2**torch.arange(self.dim, dtype=torch.float):
            embedding.append(torch.sin(x * w))
            embedding.append(torch.cos(x * w))
        return torch.cat(embedding, axis=-1)

def getModel(config):
    in_ch = config.model.encoding_dim * 2 * 3 + 3
    out_ch = 1
    depth = config.model.depth
    width = config.model.width
    activation = config.model.activation.lower()
    if activation == 'relu':
        return _Model(in_ch, out_ch, depth, width, connections=[depth//2])
    elif activation == 'sin':
        return SIREN(in_ch, out_ch, depth, width)



class _Model(nn.Module):
    def __init__(self, in_ch, out_ch, depth, width, connections):
        super(_Model, self).__init__()
        self.in_ch = in_ch
        self.depth = depth
        self.width = width
        self.connections = connections
        # create model
        self.linears = nn.ModuleList([nn.Linear(self.in_ch, self.width)] + [nn.Linear(self.width, self.width) if i not in self.connections else nn.Linear(self.width + self.in_ch, self.width) for i in range(self.depth-1)])
        self.last_linear = nn.Linear(self.width, out_ch)
        
    def forward(self, x):
        h = x
        for idx, func in enumerate(self.linears):
            h = func(h)
            h = F.relu(h)
            if idx in self.connections:
                h = torch.cat((x, h), dim=-1)
        x = self.last_linear(h)
        return x



class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else (6 / self.in_f)**.5 / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):

    def __init__(self, in_ch, out_ch, depth, width):
        super(SIREN, self).__init__()
        layers = [SirenLayer(in_ch, width, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(width, width))
        layers.append(SirenLayer(width, out_ch, is_last=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out