# 

import beartype.typing as tp
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from gpepx.kernels.base import AbstractKernel
from gpepx.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpepx.parameters import PositiveReal
from gpepx.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

WeightVariance = tp.Union[Float[Array, "D"], ScalarArray]
WeightVarianceCompatible = tp.Union[ScalarFloat, list[float], WeightVariance]


class ArcCosine(AbstractKernel):
    r"""The ArCosine kernel.

    This kernel is non-stationary and resembles the behavior of neural networks.
    See Section 3.1 of
    [Cho and Saul (2011)](https://arxiv.org/abs/1112.3712) for
    additional details.
    """

    variance: ScalarArray
    weight_variance: WeightVariance
    bias_variance: ScalarArray
    name = "ArcCosine"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        order: tp.Literal[0, 1, 2] = 0,
        variance: tp.Union[ScalarFloat, ScalarArray] = 1.0,
        weight_variance: tp.Union[
            WeightVarianceCompatible, WeightVariance
        ] = 1.0,
        bias_variance: tp.Union[ScalarFloat, nnx.Variable[ScalarArray]] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            order: The order of the kernel. Must be 0, 1 or 2.
            variance: The variance of the kernel σ.
            weight_variance: The weight variance of the kernel.
            bias_variance: The bias variance of the kernel.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """

        if order not in [0, 1, 2]:
            raise ValueError("ArcCosine kernel only implemented for orders 0, 1 and 2.")

        self.order = order
        self.weight_variance = weight_variance
        self.variance = variance
        self.bias_variance = bias_variance
        self.name = f"ArcCosine (order {self.order})"
        super().__init__(active_dims, n_dims, compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarArray:
        x = self.slice_input(x)
        y = self.slice_input(y)

        x_x = self._weighted_prod(x, x)
        x_y = self._weighted_prod(x, y)
        y_y = self._weighted_prod(y, y)

        cos_theta = x_y / jnp.sqrt(x_x * y_y)
        jitter = 1e-15  # improve numerical stability
        theta = jnp.arccos(jitter + (1 - 2 * jitter) * cos_theta)

        K = self._J(theta)
        K *= jnp.sqrt(x_x) ** self.order
        K *= jnp.sqrt(y_y) ** self.order
        K *= self.variance.value / jnp.pi

        return K.squeeze()

    def _weighted_prod(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> ScalarFloat:
        r"""Calculate the weighted product between two arguments.

        Args:
            x (Float[Array, "D"]): The left hand argument.
            y (Float[Array, "D"]): The right hand argument.
        Returns:
            ScalarFloat: The value of the weighted product between the two arguments``.
        """
        return jnp.inner(self.weight_variance.value * x, y) + self.bias_variance.value

    def _J(self, theta: ScalarFloat) -> ScalarFloat:
        r"""Evaluate the angular dependency function corresponding to the desired order.

        Args:
            theta (Float[Array, "1"]): The weighted angle between inputs.

        Returns:
            Float[Array, "1"]: The value of the angular dependency function`.
        """
        return jnp.select(
            condlist=[self.order == 0,self.order == 1],
            choicelist=[jnp.pi - theta,jnp.sin(theta) + (jnp.pi - theta) * jnp.cos(theta)],
            default=3.0 * jnp.sin(theta) * jnp.cos(theta) + (jnp.pi - theta) * (1.0 + 2.0 * jnp.square(jnp.cos(theta)))
        )

