# 
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpepx.typing import (
    Array,
    ScalarFloat,
)

tfd = tfp.distributions


def build_student_t_distribution(nu: int) -> tfd.Distribution:
    r"""Build a Student's t distribution with a fixed smoothness parameter.

    For a fixed half-integer smoothness parameter, compute the spectral density of a
    Matérn kernel; a Student's t distribution.

    Args:
        nu (int): The smoothness parameter of the Matérn kernel.

    Returns
    -------
        tfp.Distribution: A Student's t distribution with the same smoothness parameter.
    """
    dist = tfd.StudentT(df=nu, loc=0.0, scale=1.0)
    return dist


def squared_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Compute the squared distance between a pair of inputs.

    Args:
        x (Float[Array, " D"]): First input.
        y (Float[Array, " D"]): Second input.

    Returns
    -------
        ScalarFloat: The squared distance between the inputs.
    """
    return jnp.sum(jnp.square(x - y))


def euclidean_distance(x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
    r"""Compute the euclidean distance between a pair of inputs.

    Args:
        x (Float[Array, " D"]): First input.
        y (Float[Array, " D"]): Second input.

    Returns
    -------
        ScalarFloat: The euclidean distance between the inputs.
    """
    return jnp.sqrt(jnp.maximum(squared_distance(x, y), 1e-36))
