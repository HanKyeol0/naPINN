# models
from pinnlab.models.mlp import MLP
from pinnlab.models.bpinn import BPINN

# experiments
from pinnlab.experiments.burgers2d import Burgers2D
from pinnlab.experiments.allencahn2d import AllenCahn2D
from pinnlab.experiments.lambdaomega2d import LambdaOmega2D

_MODEL_REG = {
    "mlp": MLP,
    "bpinn": BPINN,
}

_EXP_REG = {
    "burgers2d": Burgers2D,
    "allencahn2d": AllenCahn2D,
    "lambdaomega2d": LambdaOmega2D,
}

def get_model(name):     return _MODEL_REG[name]
def get_experiment(name):return _EXP_REG[name]
