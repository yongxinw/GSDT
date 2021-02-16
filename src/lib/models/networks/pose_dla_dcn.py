from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
from os.path import join

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn

from torch_geometric.nn import GATConv, GraphConv, GCNConv, AGNNConv, EdgeConv
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch

from .DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))



class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out



class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc
#

def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model



def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out



class DLASegGNN(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, num_gnn_layers=1, gnn_func=GraphConv,
                 use_residual=False, return_pre_gnn_layer_outputs=False, heads_share_params=False,
                 freeze_bn=False, trainable_modules=None, omit_gnn=False, use_roi_align=False,
                 gatconv_heads=4):
        super(DLASegGNN, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        self.create_heads(channels, final_kernel, head_conv, heads)

        if return_pre_gnn_layer_outputs:
            if not heads_share_params:
                for i in range(num_gnn_layers):
                    heads_i = {f"{k}_{i}": v for k, v in heads.items()}
                    setattr(self, f"heads_{i}", heads_i)
                    self.create_heads(channels, final_kernel, head_conv, heads_i)

        self.gnn_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.gnn = nn.ModuleList()
        self.omit_gnn = omit_gnn
        self.use_roi_align = use_roi_align
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn
        self.trainable_modules = trainable_modules

        self.use_residual = use_residual
        self.return_pre_gnn_layer_outputs = return_pre_gnn_layer_outputs
        self.heads_share_params = heads_share_params
        self.num_gnn_layers = num_gnn_layers

        for _ in range(num_gnn_layers):
            if gnn_func == EdgeConv:
                self.fc = nn.Sequential(
                    nn.Linear(channels[self.first_level] * 2, channels[self.first_level]),
                    nn.ReLU(),
                    nn.Linear(channels[self.first_level], channels[self.first_level])
                )

                edge_conv = EdgeConv(nn=self.fc, aggr='mean')
                self.gnn.append(edge_conv)
            elif gnn_func == GATConv:
                # TODO: implement for GATConv
                gat_conv = GATConv(channels[self.first_level], channels[self.first_level] // gatconv_heads, heads=gatconv_heads)
                self.gnn.append(gat_conv)
            else:
                self.gnn.append(gnn_func(channels[self.first_level], channels[self.first_level]))

        if self.use_roi_align:
            import torchvision
            self.roi_align = torchvision.ops.RoIAlign(output_size=1, spatial_scale=1.0, sampling_ratio=-1)

    def create_heads(self, channels, final_kernel, head_conv, heads):
        # self.heads = heads
        for head in heads:
            classes = heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def backbone_forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return y

    def build_graph_batch(self, y, y_p_crops_list, edge_index):
        """
        build the graph nodes and edge
        :param y: features of current input image (N, C, H, W)
        :param y_p_crops_list: list of previous crops, length of N, each with shape (n_crops_i, 64)
        :param edge_index: list with length (N) of tensors, each element of which has a shape of (2, n_edges_i)
        :return:
            graph: torch_geometric.data.Batch object containing the batched graph
            center_inds: list with length (N) of tensors, each element containing the corresponding indices for centers
        """
        data_list = []
        center_inds = []
        offset = 0
        _, C, _, _ = y.shape
        for yi, y_p_crops_i, ei in zip(y, y_p_crops_list, edge_index):
            graph_nodes = torch.cat((y_p_crops_i, yi.reshape(C, -1).T.contiguous()), dim=0)
            data_list.append(gData(x=graph_nodes, edge_index=ei))
            center_inds_i = offset + len(y_p_crops_i) + torch.arange(len(yi.reshape(C, -1).T))
            center_inds.append(center_inds_i)
            offset += len(graph_nodes)
        graph = Batch.from_data_list(data_list)
        center_inds = torch.cat(center_inds)
        return graph, center_inds

    def forward(self, x, p_crops, p_crops_lengths, edge_index, p_imgs=None):
        """
        forward function of the GNN detTrack module
        :param x: input image of (N, 3, im_h, im_w)
        :param p_crops: input image crops of previous frame corresponding to each input image, (∑_i n_crops_i, 64)
        :param p_crops_lengths: lengths of the number of previous crops for each batch image (N)
        :param edge_index: list of tensors with length (N), each element of which has a shape of (2, n_edges_i)
        :return:
        """
        # Get the current image features (N, C, H, W)
        y = self.backbone_forward(x)[-1]

        if not self.omit_gnn:
            N, C, H, W = y.shape
            y_p_crops_list = self.crop_features_forward(p_crops=p_crops, p_crops_lengths=p_crops_lengths, p_imgs=p_imgs)

            # build the batched graph
            graph, center_inds = self.build_graph_batch(y, y_p_crops_list, edge_index)

            # pass through gnn
            gnn_feat = graph.x
            if self.return_pre_gnn_layer_outputs:
                cached_feats = [y]
            for gnn in self.gnn:
                gnn_out = gnn(gnn_feat, graph.edge_index)
                if self.use_residual:
                    gnn_feat = gnn_feat + gnn_out
                else:
                    gnn_feat = gnn_out
                if self.return_pre_gnn_layer_outputs:
                    cached_feats.append(gnn_feat[center_inds].reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous())
            # slice the features corresponding to the centers of each image in the batch
            center_feats = gnn_feat[center_inds].reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            center_feats = y

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(center_feats)

        if not self.return_pre_gnn_layer_outputs:
            return [z]

        # Returning the values at all previous GNN layers (i.e. before the last GNN layer)
        zs = [[z]]
        for i in range(self.num_gnn_layers):
            z_i = {}
            if not self.heads_share_params:
                for head0 in getattr(self, f"heads_{i}"):
                    name = head0.split("_")[0]
                    z_i[name] = self.__getattr__(head0)(cached_feats[i])
            else:
                for head in self.heads:
                    z_i[head] = self.__getattr__(head)(cached_feats[i])

            zs.append([z_i])

        return zs

    def crop_features_forward(self, p_crops, p_imgs, p_crops_lengths):
        if self.use_roi_align:
            assert p_imgs is not None
            p_features = self.backbone_forward(p_imgs)[-1]
            N, C, H, W = p_features.shape
            p_crops = p_crops * torch.tensor([[W, H, W, H]]).to(p_crops.device)
            p_rois = []
            for i in range(len(p_crops_lengths)):
                start = 0 if i - 1 < 0 else p_crops_lengths[i - 1]
                p_rois.append(p_crops[start: p_crops_lengths[i]])
            y_crops = self.roi_align(p_features, p_rois)
            y_crops = y_crops.squeeze(-1).squeeze(-1)
        else:
            # Get the previous crops features (∑_i n_crops_i, 64)
            y_crops = self.backbone_forward(p_crops)[-1]
            y_crops = self.gnn_pool(y_crops).squeeze(-1).squeeze(-1)
        # split the previous crops features as a list of tensors
        y_p_crops_list = []
        for i in range(len(p_crops_lengths)):
            start = 0 if i - 1 < 0 else p_crops_lengths[i - 1]
            y_p_crops_list.append(y_crops[start: p_crops_lengths[i]])
        return y_p_crops_list

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(DLASegGNN, self).train(mode)
        if self.freeze_bn:
            for c, _ in self.named_children():
                if c not in self.trainable_modules:
                    # for m in self.base.modules():
                    for name, m in self.__getattr__(c).named_modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            print(f"Freezing BN for module {c}:{name}")
                            m.eval()
                            if self.freeze_bn_affine:
                                m.weight.requires_grad = False
                                m.bias.requires_grad = False





gnn_type_2_func = {
    "GraphConv": GraphConv,
    "GATConv": GATConv,
    "GCNConv": GCNConv,
    "AGNNConv": AGNNConv,
    "EdgeConv": EdgeConv

}


def get_pose_net_with_gnn(num_layers, heads, head_conv=256, down_ratio=4, num_gnn_layers=1, gnn_type="GraphConv",
                          use_residual=False, edge_regression=False, motion_model=False, viz_attention=False, **kwargs):
    
    model = DLASegGNN('dla{}'.format(num_layers), heads,
                      pretrained=True,
                      down_ratio=down_ratio,
                      final_kernel=1,
                      last_level=5,
                      head_conv=head_conv,
                      num_gnn_layers=num_gnn_layers,
                      gnn_func=gnn_type_2_func[gnn_type],
                      use_residual=use_residual,
                      **kwargs)
    return model
