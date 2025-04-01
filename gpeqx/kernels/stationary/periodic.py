# 

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import Float

from gpeqx.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpeqx.kernels.stationary.base import StationaryKernel
from gpeqx.parameters import PositiveReal
from gpeqx.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class Periodic(StationaryKernel):
    r"""The periodic kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with length-scale
    parameter $\ell$, variance $\sigma^2$ and period $p$.
    $$
    k(x, y) = \sigma^2 \exp \left( -\frac{1}{2} \sum_{i=1}^{D} \left(\frac{\sin (\pi (x_i - y_i)/p)}{\ell}\right)^2 \right)
    $$
    Key reference is MacKay 1998 - "Introduction to Gaussian processes".
    """

    name: str = "Periodic"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, Lengthscale] = 1.0,
        variance: tp.Union[ScalarFloat, ScalarArray] = 1.0,
        period: tp.Union[ScalarFloat, ScalarArray] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: the indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            period: the period of the kernel p.
            n_dims: the number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: the computation engine that the kernel uses to compute the
                covariance matrix.
        """

        self.period = period

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        x = self.slice_input(x)
        y = self.slice_input(y)
        sine_squared = jnp.square(
            jnp.sin(jnp.pi * (x - y) / self.period.value) / self.lengthscale.value
        )
        K = self.variance.value * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
        return K.squeeze()
