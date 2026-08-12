#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NVCC=${NVCC:-nvcc}
MODE=${1:-smoke}
CUDA_FLAGS=(-O3 -lineinfo -std=c++17 -arch=sm_87)

compile_all()
{
    mkdir -p "$ROOT"/{reduce,softmax,transpose,rmsnorm,gemm,attention}/build

    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/reduce/reduce_v3.cu" \
        -o "$ROOT/reduce/build/reduce_v3"
    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/softmax/softmax_v4.cu" \
        -o "$ROOT/softmax/build/softmax_v4"
    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/transpose/transpose_v2.cu" \
        -o "$ROOT/transpose/build/transpose_v2"
    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/rmsnorm/rmsnorm_v3.cu" \
        -o "$ROOT/rmsnorm/build/rmsnorm_v3"
    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/gemm/gemm_v6.cu" \
        -o "$ROOT/gemm/build/gemm_v6" -lcublas
    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/gemm/gemm_v8.cu" \
        -o "$ROOT/gemm/build/gemm_v8" -lcublas
    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/attention/attention_v5.cu" \
        -o "$ROOT/attention/build/attention_v5"
    "$NVCC" "${CUDA_FLAGS[@]}" "$ROOT/attention/attention_v6.cu" \
        -o "$ROOT/attention/build/attention_v6"
}

check_clocks()
{
    local gpu_min gpu_max emc_cap
    gpu_min=$(</sys/devices/platform/17000000.gpu/devfreq_dev/min_freq)
    gpu_max=$(</sys/devices/platform/17000000.gpu/devfreq_dev/max_freq)
    emc_cap=$(</sys/kernel/nvpmodel_clk_cap/emc)
    if [[ "$gpu_min" != 1020000000 || "$gpu_max" != 1020000000 ||
          "$emc_cap" != 3199000000 ]]; then
        printf 'Expected GPU=1020 MHz and EMC=3199 MHz; got %s/%s and %s Hz.\n' \
            "$gpu_min" "$gpu_max" "$emc_cap" >&2
        printf 'Run sudo tools/lock_orin_clocks.sh before formal benchmarks.\n' >&2
        exit 1
    fi
}

run_smoke()
{
    "$ROOT/reduce/build/reduce_v3" --n=1000 --warmup=2 --repeat=5
    "$ROOT/softmax/build/softmax_v4" \
        --rows=17 --cols=1003 --extreme --warmup=2 --repeat=5
    "$ROOT/transpose/build/transpose_v2" \
        --rows=1000 --cols=1023 --warmup=2 --repeat=5
    "$ROOT/rmsnorm/build/rmsnorm_v3" \
        --tokens=17 --hidden=1003 --warmup=2 --repeat=5
    "$ROOT/gemm/build/gemm_v6" \
        --m=37 --n=65 --k=1003 --warmup=2 --repeat=5
    "$ROOT/gemm/build/gemm_v8" \
        --m=37 --n=65 --k=1003 --warmup=2 --repeat=5
    "$ROOT/attention/build/attention_v5" \
        --b=1 --h=2 --s=37 --d=71 --causal=1 --extreme=1 \
        --warmup=2 --repeat=5
    "$ROOT/attention/build/attention_v6" \
        --b=1 --h=2 --s=37 --d=71 --causal=1 --extreme=1 \
        --warmup=2 --repeat=5
}

run_benchmark()
{
    check_clocks
    printf 'source_commit=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
    "$ROOT/reduce/build/reduce_v3" \
        --n=16777216 --warmup=10 --repeat=30
    "$ROOT/softmax/build/softmax_v4" \
        --rows=4096 --cols=1024 --warmup=20 --repeat=50
    "$ROOT/transpose/build/transpose_v2" \
        --rows=4096 --cols=4096 --warmup=10 --repeat=30
    "$ROOT/rmsnorm/build/rmsnorm_v3" \
        --tokens=2048 --hidden=4096 --warmup=10 --repeat=30
    "$ROOT/gemm/build/gemm_v6" \
        --m=4096 --n=4096 --k=4096 --warmup=5 --repeat=15
    "$ROOT/gemm/build/gemm_v8" \
        --m=4096 --n=4096 --k=4096 --warmup=5 --repeat=15
    "$ROOT/attention/build/attention_v5" \
        --b=1 --h=12 --s=1024 --d=128 --causal=0 --warmup=5 --repeat=15
    "$ROOT/attention/build/attention_v6" \
        --b=1 --h=12 --s=1024 --d=128 --causal=0 --warmup=5 --repeat=15
}

case "$MODE" in
    compile)
        compile_all
        ;;
    smoke)
        compile_all
        run_smoke
        ;;
    benchmark)
        compile_all
        run_benchmark
        ;;
    all)
        compile_all
        run_smoke
        run_benchmark
        ;;
    *)
        printf 'Usage: %s [compile|smoke|benchmark|all]\n' "$0" >&2
        exit 2
        ;;
esac
