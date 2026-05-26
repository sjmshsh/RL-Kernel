# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

try:
    from torch.utils.cpp_extension import ROCMExtension
except ImportError:
    ROCMExtension = None


def get_extensions():
    extensions = []
    is_rocm = torch.version.hip is not None

    if is_rocm:
        extensions.append(
            ROCMExtension(
                name="rl_engine._C",
                sources=[
                    "csrc/ops.cpp",
                    "csrc/fused_logp_kernel.cpp",
                ],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "hipcc": ["-O3", "--use_fast_math", "-Xhipcc", "-compress-all"],
                },
            )
        )
    elif torch.cuda.is_available():
        cuda_sources = [
            "csrc/ops.cpp",
            "csrc/fused_logp_kernel.cu",
        ]

        cc_major, _ = torch.cuda.get_device_capability()
        nvcc_flags = ["-O3", "--use_fast_math", "-Xfatbin", "-compress-all"]
        extra_link_args = []

        tma_src = "csrc/cuda/fused_logp_sm90.cu"
        if cc_major >= 9 or os.path.exists(tma_src):
            cuda_sources.append(tma_src)
            nvcc_flags.append("-gencode=arch=compute_90a,code=sm_90a")
            extra_link_args.append("-lcuda")

        extensions.append(
            CUDAExtension(
                name="rl_engine._C",
                sources=cuda_sources,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": nvcc_flags,
                },
                extra_link_args=extra_link_args,
            )
        )
    return extensions


setup(
    name="rl-engine",
    version="0.1.0",
    packages=find_packages(include=["rl_engine", "rl_engine.*"]),
    install_requires=[
        "torch>=2.4.0",
        "tabulate",
        "numpy",
        "accelerate",
        "transformers",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    extras_require={
        "cuda": ["flashinfer"],
        "rocm": ["aiter"],
    },
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
