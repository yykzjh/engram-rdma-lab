"""Build the independent WRITE extension and device PTX; no DeepEP/MPI dependency."""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
NCCL = Path(os.environ["EP_NCCL_ROOT_DIR"] if "EP_NCCL_ROOT_DIR" in os.environ else os.environ["NCCL_ROOT"])


class BuildWrite(BuildExtension):
    def run(self):
        subprocess.run(
            [
                str(Path(CUDA_HOME) / "bin/nvcc"),
                "-std=c++17",
                "-O3",
                "-lineinfo",
                "-arch=" + os.getenv("ENGRAM_GPU_ARCH", "compute_100"),
                "-ptx",
                "-I" + str(NCCL / "include"),
                str(ROOT / "csrc/gpu_request.cu"),
                "-o",
                str(ROOT / "engram_sge_write/gpu_request.ptx"),
            ],
            check=True,
        )
        super().run()
        # wheel的build_py可能先于build_ext运行；把刚生成的PTX同时放入打包目录，
        # 避免源码树能运行、安装后的包却缺少device module。
        if not self.inplace:
            destination = Path(self.build_lib) / "engram_sge_write/gpu_request.ptx"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "engram_sge_write/gpu_request.ptx", destination)


setup(
    name="engram_sge_write",
    version="0.1.0",
    packages=find_packages(),
    package_data={"engram_sge_write": ["*.ptx"]},
    ext_modules=[
        CUDAExtension(
            "engram_sge_write._C",
            ["csrc/runtime.cpp"],
            include_dirs=[str(NCCL / "include"), str(Path(CUDA_HOME) / "include/cccl")],
            library_dirs=[str(NCCL / "lib")],
            libraries=["nccl", "ibverbs", "mlx5", "cuda"],
            extra_compile_args={"cxx": ["-std=c++20", "-O3", "-g0", "-fvisibility=hidden", "-Wno-unused-function"]},
            extra_link_args=["-Wl,-rpath," + str(NCCL / "lib")],
        )
    ],
    cmdclass={"build_ext": BuildWrite},
)
