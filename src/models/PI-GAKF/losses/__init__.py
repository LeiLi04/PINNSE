from .innovation_nll import innovation_nll
from .physics_residual import physics_residual_loss
from .wgan_gp import critic_loss, generator_loss, gradient_penalty

__all__ = [
    "innovation_nll",
    "physics_residual_loss",
    "critic_loss",
    "generator_loss",
    "gradient_penalty",
]
