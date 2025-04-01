# 

import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.distributions as tfd

from gpeqx.kernels.stationary.base import StationaryKernel
from gpeqx.kernels.stationary.utils import (
    build_student_t_distribution,
    euclidean_distance,
)
from gpeqx.typing import Array


class Matern52(StationaryKernel):
    r"""The Matérn kernel with smoothness parameter fixed at 2.5.


    Computes the covariance for pairs of inputs $(x, y)$ with
    lengthscale parameter $\ell$ and variance $\sigma^2$.

    $$
    k(x, y) = \sigma^2 \exp \Bigg(1+ \frac{\sqrt{5}\lvert x-y \rvert}{\ell^2} + \frac{5\lvert x - y \rvert^2}{3\ell^2} \Bigg)\exp\Bigg(-\frac{\sqrt{5}\lvert x-y\rvert}{\ell^2} \Bigg)
    $$
    """

    name: str = "Matérn52"

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        tau = euclidean_distance(x, y)
        K = (
            self.variance.value
            * (1.0 + jnp.sqrt(5.0) * tau + 5.0 / 3.0 * jnp.square(tau))
            * jnp.exp(-jnp.sqrt(5.0) * tau)
        )
        return K.squeeze()

    @property
    def spectral_density(self) -> tfd.Distribution:
        return build_student_t_distribution(nu=5)
