import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
EPS = 1e-10


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


class DDC(nn.Module):
    def __init__(self, input_dim, n_clusters):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()
        hidden_layers = [nn.Linear(input_dim[0], 100), nn.ReLU(), nn.BatchNorm1d(num_features=100)]
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(100, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden


class WeightedMean(nn.Module):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self, n_views, input_sizes):
        super().__init__()
        self.n_views = n_views
        self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def get_weighted_sum_output_size(self, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        return [flat_sizes[0]]

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)


def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = F.softmax(weights, dim=0)
    out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
    return out


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, feature_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, kernel_size=5, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=5, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layer(x)


class BaseMVC(nn.Module):
    def __init__(self, input_size, feature_dim, class_num):
        super(BaseMVC, self).__init__()
        self.encoder = Encoder(input_size, feature_dim)
        self.cluster_module = DDC([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, x):
        z = self.encoder(x)
        output, hidden = self.cluster_module(z)
        return output, hidden


class SiMVC(nn.Module):
    def __init__(self, view, input_size, feature_dim):
        super(SiMVC, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        return zs, fused


class SiMVCLarge(nn.Module):
    def __init__(self, view, feature_dim):
        super(SiMVCLarge, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(ConvNet())
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        return zs, fused


class MVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num):
        super(MVC, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)
        self.cluster_module = DDC(self.fusion_module.output_size, class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        output, hidden = self.cluster_module(fused)
        return output, hidden


class MVCLarge(nn.Module):
    def __init__(self, view, feature_dim, class_num):
        super(MVCLarge, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(ConvNet())
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)
        self.cluster_module = DDC(self.fusion_module.output_size, class_num)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        output, hidden = self.cluster_module(fused)
        return output, hidden


class DSMVC(nn.Module):
    def __init__(self, view_old, view_new, input_size, feature_dim, class_num):
        super(DSMVC, self).__init__()
        self.view = view_new
        self.old_model = SiMVC(view_old, input_size, feature_dim)
        self.new_model = SiMVC(view_new, input_size, feature_dim)
        self.single = Encoder(input_size[view_new-1], feature_dim)
        self.gate = WeightedMean(3, [[feature_dim], [feature_dim], [feature_dim]])
        self.cluster_module = DDC([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs_old, fused_old = self.old_model(xs)
        zs_new, fused_new = self.new_model(xs)
        single = self.single(xs[self.view-1])
        fused = self.gate([fused_old, fused_new, single])
        output, hidden = self.cluster_module(fused)
        return zs_old, zs_new, output, hidden


class DSMVCLarge(nn.Module):
    def __init__(self, view_old, view_new, input_size, feature_dim, class_num):
        super(DSMVCLarge, self).__init__()
        self.view = view_new
        self.old_model = SiMVCLarge(view_old, feature_dim)
        self.new_model = SiMVCLarge(view_new, feature_dim)
        self.single = ConvNet()
        self.gate = WeightedMean(3, [[feature_dim], [feature_dim], [feature_dim]])
        self.cluster_module = DDC([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs_old, fused_old = self.old_model(xs)
        zs_new, fused_new = self.new_model(xs)
        single = self.single(xs[self.view-1])
        fused = self.gate([fused_old, fused_new, single])
        output, hidden = self.cluster_module(fused)
        return zs_old, zs_new, output, hidden
