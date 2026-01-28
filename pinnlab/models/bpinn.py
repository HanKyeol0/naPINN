import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pinnlab.models.mlp import get_act


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def _sigma(self, rho):
        return F.softplus(rho) + 1e-8

    def _sample(self, mu, rho):
        sigma = self._sigma(rho)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x):
        if self.training:
            weight = self._sample(self.weight_mu, self.weight_rho)
            bias = self._sample(self.bias_mu, self.bias_rho)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)

        prior_var = self.prior_sigma ** 2
        weight_kl = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma ** 2 + (self.weight_mu - self.prior_mu) ** 2) / (2.0 * prior_var)
            - 0.5
        ).sum()
        bias_kl = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma ** 2 + (self.bias_mu - self.prior_mu) ** 2) / (2.0 * prior_var)
            - 0.5
        ).sum()
        return weight_kl + bias_kl


class BPINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_f, out_f = cfg["in_features"], cfg["out_features"]
        H, W = cfg["hidden_layers"], cfg["hidden_width"]
        act = get_act(cfg.get("activation", "tanh"))

        prior_mu = float(cfg.get("prior_mu", 0.0))
        prior_sigma = float(cfg.get("prior_sigma", 1.0))
        self.kl_weight = float(cfg.get("kl_weight", 1.0))

        layers = []
        fin = in_f
        for _ in range(H):
            layers.append(BayesianLinear(fin, W, prior_mu=prior_mu, prior_sigma=prior_sigma))
            layers.append(act)
            fin = W
        layers.append(BayesianLinear(fin, out_f, prior_mu=prior_mu, prior_sigma=prior_sigma))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def kl_loss(self):
        kl = 0.0
        for module in self.net.modules():
            if isinstance(module, BayesianLinear):
                kl = kl + module.kl_divergence()
        return kl
