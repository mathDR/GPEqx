# 

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import Float

from gpeqx.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpeqx.kernels.stationary.base import StationaryKernel
from gpeqx.kernels.stationary.utils import euclidean_distance
from gpeqx.parameters import SigmoidBounded
from gpeqx.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class PoweredExponential(StationaryKernel):
    r"""The powered exponential family of kernels.

    Computes the covariance for pairs of inputs $(x, y)$ with length-scale parameter
    $\ell$, $\sigma$ and power $\kappa$.
    $$
    k(x, y)=\sigma^2\exp\Bigg(-\Big(\frac{\lVert x-y\rVert^2}{\ell^2}\Big)^\kappa\Bigg)
    $$

    This also equivalent to the symmetric generalized normal distribution.
    See Diggle and Ribeiro (2007) - "Model-based Geostatistics".
    and
    https://en.wikipedia.org/wiki/Generalized_normal_distribution#Symmetric_version
    """

    name: str = "Powered Exponential"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, Lengthscale] = 1.0,
        variance: tp.Union[ScalarFloat, ScalarArray]]= 1.0,
        power: tp.Union[ScalarFloat, ScalarArray]]= 1.0,
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
            power: the power of the kernel κ.
            n_dims: the number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: the computation engine that the kernel uses to compute the
                covariance matrix.
        """
        self.power = power

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, ""]:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        K = self.variance.value * jnp.exp(
            -(euclidean_distance(x, y) ** self.power.value)
        )
        return K.squeeze()
