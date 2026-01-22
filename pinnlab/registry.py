# models
from pinnlab.models.mlp import MLP
from pinnlab.models.fourier_mlp import FourierMLP
from pinnlab.models.residual_network import ResidualNetwork

# experiments
from pinnlab.experiments.burgers2d import Burgers2D
from pinnlab.experiments.helmholtz2d import Helmholtz2D
from pinnlab.experiments.navierstokes2d_cylinder import NavierStokesCylinder
from pinnlab.experiments.allencahn2d import AllenCahn2D
from pinnlab.experiments.lambdaomega2d import LambdaOmega2D
from pinnlab.experiments.fitzhugh_nagumo_rd2d import FitzHughNagumoRD2D

_MODEL_REG = {
    "mlp": MLP,
    "fourier_mlp": FourierMLP,
    "residual_network": ResidualNetwork,
}

_EXP_REG = {
    "burgers2d": Burgers2D,
    "helmholtz2d": Helmholtz2D,
    "navierstokes2d_cylinder": NavierStokesCylinder,
    "allencahn2d": AllenCahn2D,
    "lambdaomega2d": LambdaOmega2D,
    "fitzhugh_nagumo_rd2d": FitzHughNagumoRD2D,
}

def get_model(name):     return _MODEL_REG[name]
def get_experiment(name):return _EXP_REG[name]
