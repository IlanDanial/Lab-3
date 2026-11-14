#!/bin/bash
# Usage (on a GPU node): bash energy_compare.sh <mlp binary> [more binaries ...]   env: ENERGY_ITERS=2000 ROUNDS=2
# Runs each binary's student kernel at sustained load, alternating binaries each round, and reports
# median SM clock and power over busy samples, energy per pass, and GFLOP per joule.
BENCH="4093 1000 1990 1010 500 7"
ITERS=${ENERGY_ITERS:-2000}
ROUNDS=${ROUNDS:-2}
GPU_IDX=${CUDA_VISIBLE_DEVICES%%,*}
GPU_IDX=${GPU_IDX:-0}
TMP=$(mktemp -d)
for r in $(seq 1 "$ROUNDS"); do
  for bin in "$@"; do
    nvidia-smi -i "$GPU_IDX" --query-gpu=clocks.sm,power.draw,utilization.gpu \
      --format=csv,noheader,nounits -lms 100 > "$TMP/samples.csv" 2>&1 &
    smi=$!
    per=$("$bin" $BENCH --iters "$ITERS" --skip-ref --skip-baseline | sed -nE 's/^\[student\].*per_iter_ms=([0-9.]+).*/\1/p')
    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    awk -F', *' -v per="$per" -v bin="$bin" -v r="$r" -v bench="$BENCH" '
      function med(a, n,   i, j, t) {
        for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
        return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
      }
      $3 + 0 >= 90 { n++; c[n] = $1 + 0; p[n] = $2 + 0 }   # busy samples only
      END {
        if (!n || per == "") { printf "round %d  %s: no data\n", r, bin; exit }
        split(bench, d, " "); f = 2 * d[1] * (d[2] * d[3] + d[3] * d[4] + d[4] * d[5])
        mc = med(c, n); mp = med(p, n)
        printf "round %d  %-28s %.3f ms/pass | %4.0f MHz | %4.1f W | %5.1f mJ/pass | %3.0f GFLOP/J\n",
          r, bin, per, mc, mp, mp * per, f / 1e9 / (mp * per / 1000)
      }' "$TMP/samples.csv"
  done
done
rm -rf "$TMP"
