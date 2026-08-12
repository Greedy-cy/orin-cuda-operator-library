import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "python" / "build" / "operatorlib"
BUILD.mkdir(parents=True, exist_ok=True)

os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get(
    "OPERATORLIB_CUDA_ARCH_LIST", "8.7"
)
os.environ.setdefault("MAX_JOBS", "2")


def load_operatorlib(verbose: bool = False) -> None:
    load(
        name="operatorlib_ext",
        sources=[
            str(ROOT / "python" / "operatorlib_binding.cpp"),
            str(ROOT / "src" / "reduce.cu"),
            str(ROOT / "src" / "softmax.cu"),
            str(ROOT / "src" / "transpose.cu"),
            str(ROOT / "src" / "rmsnorm.cu"),
            str(ROOT / "src" / "gemm.cu"),
            str(ROOT / "src" / "attention.cu"),
        ],
        extra_include_paths=[str(ROOT / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"],
        build_directory=str(BUILD),
        with_cuda=True,
        is_python_module=False,
        verbose=verbose,
    )


if __name__ == "__main__":
    load_operatorlib(verbose=True)
