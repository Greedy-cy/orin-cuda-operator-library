#!/usr/bin/env bash
set -euo pipefail

# Reproduce the clock state used by the standalone benchmark reports.
# This script is intentionally Orin-specific and must be run as root.

if [[ ${EUID} -ne 0 ]]; then
    printf 'Run with sudo: sudo %s\n' "$0" >&2
    exit 1
fi

GPU_MIN=/sys/devices/platform/17000000.gpu/devfreq_dev/min_freq
GPU_MAX=/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq
EMC_CAP=/sys/kernel/nvpmodel_clk_cap/emc
EMC_DEBUG=/sys/kernel/debug/bpmp/debug/clk/emc

for path in "$GPU_MIN" "$GPU_MAX" "$EMC_CAP" \
            "$EMC_DEBUG/rate" "$EMC_DEBUG/mrq_rate_locked"; do
    if [[ ! -e "$path" ]]; then
        printf 'Required Orin clock path is missing: %s\n' "$path" >&2
        exit 1
    fi
done

nvpmodel -m 2
jetson_clocks

# On this JetPack image, jetson_clocks may leave the EMC cap at 2.133 GHz
# after a reboot even in MAXN_SUPER. The historical reports used 3.199 GHz.
printf '3199000000\n' > "$EMC_CAP"
printf '0\n' > "$EMC_DEBUG/mrq_rate_locked"
printf '3199000000\n' > "$EMC_DEBUG/rate"
printf '1\n' > "$EMC_DEBUG/mrq_rate_locked"

gpu_min=$(<"$GPU_MIN")
gpu_max=$(<"$GPU_MAX")
emc_cap=$(<"$EMC_CAP")
emc_rate=$(<"$EMC_DEBUG/rate")

printf 'GPU min/max: %s/%s Hz\n' "$gpu_min" "$gpu_max"
printf 'EMC cap/rate: %s/%s Hz\n' "$emc_cap" "$emc_rate"

if [[ "$gpu_min" != 1020000000 || "$gpu_max" != 1020000000 ||
      "$emc_cap" != 3199000000 || "$emc_rate" != 3199000000 ]]; then
    printf 'Clock verification failed; do not record benchmark results.\n' >&2
    exit 1
fi

printf 'Clock state matches the frozen standalone benchmark environment.\n'
