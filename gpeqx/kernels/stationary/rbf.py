# 

import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax as tfp

from gpeqx.kernels.stationary.base import StationaryKernel
from gpeqx.kernels.stationary.utils import squared_distance
from gpeqx.typing import (
    Array,
    ScalarFloat,
)


class RBF(StationaryKernel):
    r"""The Radial Basis Function (RBF) kernel.

    Computes the covariance for pair of inputs $(x, y)$ with lengthscale parameter
    $\ell$ and variance $\sigma^2$:
    $$
    k(x,y)=\sigma^2\exp\Bigg(- \frac{\lVert x - y \rVert^2_2}{2 \ell^2} \Bigg)
    $$
    """

    name: str = "RBF"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        K = self.variance.value * jnp.exp(-0.5 * squared_distance(x, y))
        return K.squeeze()

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)
