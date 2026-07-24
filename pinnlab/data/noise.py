"""Measurement-noise distributions used by the paper experiments."""

import torch
import torch.distributions as D


def get_noise(kind, f, pars=0, par_list=None):
    if kind == "G":
        return GaussianNoise(f, _parameters(pars, [0.0, 4.0]))
    if kind == "Laplace":
        return LaplaceNoise(f, _parameters(pars, [0.0, 1.0]))
    if kind == "StudentT":
        return StudentTNoise(f, _parameters(pars, [3.0, 0.0, 1.0]))
    if kind == "4G":
        if not par_list:
            raise ValueError("4G noise requires four [mean, std] components")
        return GaussianMixtureNoise(f, par_list)
    raise ValueError(f"Unsupported noise distribution: {kind}")


def _parameters(parameters, default):
    return default if isinstance(parameters, int) else parameters


class GaussianNoise:
    def __init__(self, scale, parameters):
        mean, std = parameters
        self.distribution = D.Normal(mean * scale, std * scale)

    def sample(self, count):
        return self.distribution.sample((count,))

    def pdf(self, value):
        return torch.exp(self.distribution.log_prob(value))


class LaplaceNoise:
    def __init__(self, scale, parameters):
        location, width = parameters
        self.distribution = D.Laplace(location * scale, width * scale)

    def sample(self, count):
        return self.distribution.sample((count,))

    def pdf(self, value):
        return torch.exp(self.distribution.log_prob(value))


class StudentTNoise:
    def __init__(self, scale, parameters):
        degrees_of_freedom, location, width = parameters
        self.distribution = D.StudentT(
            df=degrees_of_freedom,
            loc=location * scale,
            scale=width * scale,
        )

    def sample(self, count):
        return self.distribution.sample((count,))

    def pdf(self, value):
        return torch.exp(self.distribution.log_prob(value))


class GaussianMixtureNoise:
    def __init__(self, scale, components):
        parameters = torch.as_tensor(components, dtype=torch.float32)
        if parameters.shape != (4, 2):
            raise ValueError("4G noise requires exactly four [mean, std] pairs")

        mixture = D.Categorical(logits=torch.zeros(4))
        component_distribution = D.Normal(
            parameters[:, 0] * scale,
            parameters[:, 1] * scale,
        )
        self.distribution = D.MixtureSameFamily(mixture, component_distribution)

    def sample(self, count):
        return self.distribution.sample((count,))

    def pdf(self, value):
        return torch.exp(self.distribution.log_prob(value))
