#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

setup(
    name="gaussian_hierarchy",
    install_requires=["ninja", "jaxtyping", "torch", "numpy"],
    ext_modules=[
        CUDAExtension(
            name="gaussian_hierarchy._C",
            sources=[
                os.path.join(os.path.join(BASE_PATH, "src/"), f)
                for f in os.listdir(os.path.join(BASE_PATH, "src/"))
                if (f.endswith(".cpp") or f.endswith(".cu")) and not f.startswith("mainHierarchy")
            ],
            extra_compile_args=[
                "-I" + os.path.join(BASE_PATH, "include/"),
                "-I" + os.path.join(BASE_PATH, "dependencies/"),
                "-I" + os.path.join(BASE_PATH, "dependencies/eigen/"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
