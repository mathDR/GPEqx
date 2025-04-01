# 

import beartype.typing as tp
from jaxtyping import Float

from gpeqx.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpeqx.kernels.stationary.base import StationaryKernel
from gpeqx.kernels.stationary.utils import squared_distance
from gpeqx.parameters import PositiveReal
from gpeqx.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class RationalQuadratic(StationaryKernel):
    r"""The Rational Quadratic kernel.

    Computes the covariance for pairs of inputs $(x, y)$ with lengthscale parameter
    $\ell$ and variance $\sigma^2$.
    $$
    k(x,y)=\sigma^2\exp\Bigg(1+\frac{\lVert x-y\rVert^2_2}{2\alpha\ell^2}\Bigg)
    $$
    """

    name: str = "Rational Quadratic"

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, Lengthscale] = 1.0,
        variance: tp.Union[ScalarFloat, ScalarArray] = 1.0,
        alpha: tp.Union[ScalarFloat, ScalarArray] = 1.0,
        n_dims: tp.Union[int, None] = None,
        compute_engine: AbstractKernelComputation = DenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            alpha: the alpha parameter of the kernel α.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """
        self.alpha = alpha

        super().__init__(active_dims, lengthscale, variance, n_dims, compute_engine)

    def __call__(self, x: Float[Array, " D"], y: Float[Array, " D"]) -> ScalarFloat:
        x = self.slice_input(x) / self.lengthscale.value
        y = self.slice_input(y) / self.lengthscale.value
        K = self.variance.value * (
            1 + 0.5 * squared_distance(x, y) / self.alpha.value
        ) ** (-self.alpha.value)
        return K.squeeze()
