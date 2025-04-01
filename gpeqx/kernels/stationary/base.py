# 

import beartype.typing as tp
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
import tensorflow_probability.substrates.jax.distributions as tfd

from gpeqx.kernels.base import AbstractKernel
from gpeqx.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpeqx.parameters import PositiveReal
from gpeqx.typing import (
    Array,
    ScalarArray,
    ScalarFloat,
)

Lengthscale = tp.Union[Float[Array, "D"], ScalarArray]
LengthscaleCompatible = tp.Union[ScalarFloat, list[float], Lengthscale]


class StationaryKernel(AbstractKernel):
    """Base class for stationary kernels.

    Stationary kernels are a class of kernels that are invariant to translations
    in the input space. They can be isotropic or anisotropic, meaning that they
    can have a single lengthscale for all input dimensions or a different lengthscale
    for each input dimension.
    """

    lengthscale: nnx.Variable[Lengthscale]
    variance: nnx.Variable[ScalarArray]

    def __init__(
        self,
        active_dims: tp.Union[list[int], slice, None] = None,
        lengthscale: tp.Union[LengthscaleCompatible, Lengthscale] = 1.0,
        variance: tp.Union[ScalarFloat, ScalarArray] = 1.0,
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
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """

        super().__init__(active_dims, n_dims, compute_engine)
        self.lengthscale = lengthscale
        self.variance = variance

    @property
    def spectral_density(self) -> tfd.Distribution:
        r"""The spectral density of the kernel.

        Returns:
            Callable[[Float[Array, "D"]], Float[Array, "D"]]: The spectral density function.
        """
        raise NotImplementedError(
            f"Kernel {self.name} does not have a spectral density."
        )
