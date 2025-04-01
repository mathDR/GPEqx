# 

import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.distributions as tfd

from gpeqx.kernels.stationary.base import StationaryKernel
from gpeqx.kernels.stationary.utils import (
    build_student_t_distribution,
    euclidean_distance,
)
from gpeqx.typing import (
    Array,
    ScalarFloat,
)


class Matern12(StationaryKernel):
    r"""The Matérn kernel with smoothness parameter fixed at 0.5.

    Computes the covariance on a pair of inputs $(x, y)$ with
    lengthscale parameter $\ell$ and variance $\sigma^2$.

    $$
    k(x, y) = \sigma^2\exp\Bigg(-\frac{\lvert x-y \rvert}{2\ell^2}\Bigg)
    $$
    """

    name: str = "Matérn12"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        K = self.variance.value * jnp.exp(-euclidean_distance(x, y))
        return K.squeeze()

    @property
    def spectral_density(self) -> tfd.Distribution:
        return build_student_t_distribution(nu=1)
