import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.layers.mesh_conv import MeshConv
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool

##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """
    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))


        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.leaky_relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.leaky_relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.leaky_relu(x)
        return x


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """
    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        self.decoder = MeshDecoder(unrolls, up_convs, blocks=blocks, transfer_data=transfer_data)

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        return fe

    def __call__(self, x, meshes):
        return self.forward(x, meshes)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0):
        super(DownConv, self).__init__()
        self.bn = []
        self.pool = None
        self.conv1 = MeshConv(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if pool:
            self.pool = MeshPool(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.leaky_relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            x2 = x2 + x1
            x2 = F.leaky_relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        before_pool = None
        if self.pool:
            before_pool = x2
            x2 = self.pool(x2, meshes)
        return x2, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,
                 batch_norm=True, transfer_data=True):
        super(UpConv, self).__init__()
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = MeshConv(in_channels, out_channels)
        if transfer_data:
            self.conv1 = MeshConv(2 * out_channels, out_channels)
        else:
            self.conv1 = MeshConv(out_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(MeshConv(out_channels, out_channels))
            self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if unroll:
            self.unroll = MeshUnpool(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        if self.unroll:
            x1 = self.unroll(x1, meshes)
        if self.transfer_data:
            x1 = torch.cat((x1, from_down), 1)
        x1 = self.conv1(x1, meshes)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = F.leaky_relu(x1)
        x2 = x1
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1, meshes)
            if self.bn:
                x2 = self.bn[idx + 1](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = F.leaky_relu(x2)
            x1 = x2
        x2 = x2.squeeze(3)
        return x2


class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()
        self.fcs = None
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool))
        self.global_pool = None
        if fcs is not None:
            self.fcs = []
            self.fcs_bn = []
            last_length = convs[-1]
            if global_pool is not None:
                if global_pool == 'max':
                    self.global_pool = nn.MaxPool1d(pools[-1])
                elif global_pool == 'avg':
                    self.global_pool = nn.AvgPool1d(pools[-1])
                else:
                    assert False, 'global_pool %s is not defined' % global_pool
            else:
                last_length *= pools[-1]
            if fcs[0] == last_length:
                fcs = fcs[1:]
            for length in fcs:
                self.fcs.append(nn.Linear(last_length, length))
                self.fcs_bn.append(nn.InstanceNorm1d(length))
                last_length = length
            self.fcs = nn.ModuleList(self.fcs)
            self.fcs_bn = nn.ModuleList(self.fcs_bn)
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        encoder_outs = []
        for conv in self.convs:
            fe, before_pool = conv((fe, meshes))
            encoder_outs.append(before_pool)
        if self.fcs is not None:
            if self.global_pool is not None:
                fe = self.global_pool(fe)
            fe = fe.contiguous().view(fe.size()[0], -1)
            for i in range(len(self.fcs)):
                fe = self.fcs[i](fe)
                if self.fcs_bn:
                    x = fe.unsqueeze(1)
                    fe = self.fcs_bn[i](x).squeeze(1)
                if i < len(self.fcs) - 1:
                    fe = F.leaky_relu(fe)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i], convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data))
        self.final_conv = UpConv(convs[-2], convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i+2)]
            fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

def reset_params(model): # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)