# 
import typing as tp

import jax.numpy as jnp
from jaxtyping import Float

from gpeqx.kernels.computations import (
    AbstractKernelComputation,
    ConstantDiagonalKernelComputation,
)
from gpeqx.kernels.stationary.base import StationaryKernel
from gpeqx.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)


class White(StationaryKernel):
    r"""The White noise kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with variance $\sigma^2$:
    $$
    k(x, y) = \sigma^2 \delta(x-y)
    $$
    """

    name: str = "White"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        variance: tp.Union[ScalarFloat, ScalarArray] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = ConstantDiagonalKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            variance: the variance of the kernel σ.
            n_dims: The number of input dimensions.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix
        """
        super().__init__(active_dims, 1.0, variance, n_dims, compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        K = jnp.all(jnp.equal(x, y)) * self.variance.value
        return K.squeeze()
