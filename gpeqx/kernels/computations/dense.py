# 

import beartype.typing as tp
from jax import vmap
from jaxtyping import Float

import gpepx  # noqa: F401
from gpepx.kernels.computations.base import AbstractKernelComputation
from gpepx.typing import Array

K = tp.TypeVar("K", bound="gpepx.kernels.base.AbstractKernel")  # noqa: F821


class DenseKernelComputation(AbstractKernelComputation):
    r"""Dense kernel computation class. Operations with the kernel assume
    a dense gram matrix structure.
    """

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
