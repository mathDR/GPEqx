# 

import beartype.typing as tp
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.parameters import PositiveReal
from gpjax.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)


class Polynomial(AbstractKernel):
    r"""The Polynomial kernel with variable degree.

    Computes the covariance for pairs of inputs $(x, y)$ with variance $\sigma^2$:
    $$
    k(x, y) = (\alpha + \sigma^2 x y)^d
    $$
    where $\sigma^\in \mathbb{R}_{>0}$ is the kernel's variance parameter, shift
    parameter $\alpha$ and integer degree $d$.
    """

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        degree: int = 2,
        shift: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 0.0,
        variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            degree: The degree of the polynomial.
            shift: The shift parameter of the kernel.
            variance: The variance of the kernel.
            n_dims: The number of input dimensions.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        super().__init__(active_dims, n_dims, compute_engine)

        self.degree = degree
        self.shift = shift
        self.variance = variance
        self.name = f"Polynomial (degree {self.degree})"

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x)
        y = self.slice_input(y)
        K = jnp.power(
            self.shift.value + self.variance.value * jnp.dot(x, y), self.degree
        )
        return K.squeeze()
