#!/bin/bash -l
#SBATCH --job-name=gpu_lab3_mlp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --account=eel4930
#SBATCH --qos=eel4930

# Lab 3, Option B (MLP): build, correctness, Compute Sanitizer, timing, ncu check.
#
# Submit from lab3/mlp:
#   sbatch run.slurm                         # logs go to logs/run_<jobid>/
#   sbatch run.slurm v1_tiled                # tag a version: logs/v1_tiled_<jobid>/
#   REPEATS=7 ITERS=200 sbatch run.slurm v2  # more timing repeats or iterations
#
# Every job saves a copy of src/*.cu next to its logs, so each number in the
# report can be traced back to the exact code that produced it.

LABEL=${1:-run}
LABEL=${LABEL//[^A-Za-z0-9_.-]/_}
REPEATS=${REPEATS:-5}
ITERS=${ITERS:-100}
BENCH="4093 1000 1990 1010 500 7"   # benchmark size from sizes.txt

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [[ ! -f Makefile || ! -f src/student.cu ]]; then
  echo "ERROR: submit this from lab3/mlp (no Makefile or src/student.cu in $PWD)" >&2
  exit 1
fi

LOGDIR="logs/${LABEL}_${SLURM_JOB_ID:-local}"
mkdir -p "$LOGDIR/src_snapshot"
cp src/*.cu "$LOGDIR/src_snapshot/"
cp src/*.cuh "$LOGDIR/src_snapshot/" 2>/dev/null

FAILS=0
section() { echo; echo "=== $* ==="; }
# run <logfile> <command...>: print the command, run it, append output to logfile,
# and return the command's own exit code (not tee's).
run() {
  local log=$1; shift
  echo "\$ $*" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo >> "$log"
  return "$rc"
}

# 1. Modules and environment
module purge
module load gcc cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # the CPU reference uses OpenMP

GPU_IDX=${CUDA_VISIBLE_DEVICES%%,*}
GPU_IDX=${GPU_IDX:-0}
GPU_NAME=$(nvidia-smi -i "$GPU_IDX" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)

section "Environment"
{
  echo "Job ID:  ${SLURM_JOB_ID:-n/a}"
  echo "Label:   $LABEL"
  echo "Host:    $(hostname)"
  echo "Date:    $(date)"
  echo "OpenMP threads for the CPU reference: $OMP_NUM_THREADS"
  nvidia-smi -i "$GPU_IDX" --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  nvcc --version | tail -2
  gcc --version | head -1
} 2>&1 | tee "$LOGDIR/env.txt"

if [[ "$GPU_NAME" != *L4* ]]; then
  echo "WARNING: this GPU is '$GPU_NAME', not an L4. Don't compare these timings with L4 runs." | tee -a "$LOGDIR/env.txt"
fi

# Device limits for the occupancy and peak calculations (SM count, L2, registers, shared memory).
# Built outside src/ so the lab Makefile never picks it up.
cat > "$LOGDIR/devinfo.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("cudaGetDeviceProperties failed\n"); return 1; }
  printf("device: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
  printf("SMs: %d\n", p.multiProcessorCount);
  printf("L2 cache: %d KB\n", p.l2CacheSize / 1024);
  printf("registers per SM: %d\n", p.regsPerMultiprocessor);
  printf("max threads per SM: %d\n", p.maxThreadsPerMultiProcessor);
  printf("max blocks per SM: %d\n", p.maxBlocksPerMultiProcessor);
  printf("shared memory per SM: %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
  printf("shared memory per block, default / opt-in: %zu / %zu KB\n",
         p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024);
  return 0;
}
EOF
if nvcc -o "$LOGDIR/devinfo" "$LOGDIR/devinfo.cu" > /dev/null 2>&1; then
  "$LOGDIR/devinfo" | tee -a "$LOGDIR/env.txt"
else
  echo "(could not build the device-info helper)" | tee -a "$LOGDIR/env.txt"
fi

# 2. Build
section "Build"
make clean > /dev/null
if ! run "$LOGDIR/build.log" make; then
  echo "BUILD FAILED, see $LOGDIR/build.log"
  exit 1
fi

# 3. Correctness against the CPU reference
PASSED=0
TOTAL=0
check_case() {
  TOTAL=$((TOTAL + 1))
  # $1 is left unquoted on purpose so it splits into separate arguments.
  if run "$LOGDIR/correctness.log" ./mlp $1; then
    PASSED=$((PASSED + 1))
  else
    FAILS=$((FAILS + 1))
    echo ">>> FAIL: ./mlp $1" | tee -a "$LOGDIR/correctness.log"
  fi
}

section "Correctness: sizes.txt cases"
for c in "100 37 53 29 10 1" \
         "257 784 500 300 10 2" \
         "1000 100 70 40 10 3" \
         "33 17 19 23 5 4" \
         "64 64 64 64 64 5"; do
  check_case "$c"
done

# Not in sizes.txt. Grading uses a hidden size, so also cover all-ones,
# width-1 layers, a single output, and the slide-example shape.
section "Correctness: extra edge cases"
for c in "1 1 1 1 1 11" \
         "3 5 1 7 1 12" \
         "129 65 33 17 2 13" \
         "4093 784 500 300 10 14"; do
  check_case "$c"
done

section "Correctness: benchmark size (CPU reference, 1 iteration)"
check_case "$BENCH --iters 1"

# 4. Compute Sanitizer. memcheck is required; racecheck and synccheck are extra,
#    but they catch missing __syncthreads() once shared-memory tiles are in use.
section "Compute Sanitizer"
san() {   # san <result variable> <tool> <mlp args>
  if run "$LOGDIR/sanitizer.log" compute-sanitizer --tool "$2" --error-exitcode 1 ./mlp $3; then
    printf -v "$1" PASS
  else
    printf -v "$1" FAIL
    FAILS=$((FAILS + 1))
  fi
}
san MEM1 memcheck  "100 37 53 29 10 1"
san MEM2 memcheck  "33 17 19 23 5 4"
san RACE racecheck "100 37 53 29 10 1"
san SYNC synccheck "100 37 53 29 10 1"

# 5. Timing. Each repeat times baseline and student back to back on the same node,
#    while the GPU's clocks and power are sampled in the background.
section "Timing: $REPEATS repeats of ./mlp $BENCH --iters $ITERS --skip-ref"
nvidia-smi -i "$GPU_IDX" \
  --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -lms 250 > "$LOGDIR/gpu_clocks.csv" 2>&1 &
SMI_PID=$!
for r in $(seq 1 "$REPEATS"); do
  echo "--- repeat $r/$REPEATS ---" | tee -a "$LOGDIR/timing.log"
  run "$LOGDIR/timing.log" ./mlp $BENCH --iters "$ITERS" --skip-ref || FAILS=$((FAILS + 1))
done
kill "$SMI_PID" 2>/dev/null

per_iter() { grep "^\[$1\]" "$LOGDIR/timing.log" | sed -E 's/.*per_iter_ms=([0-9.]+).*/\1/'; }
median() {
  sort -g | awk '{v[NR] = $1}
    END { if (NR == 0) print "n/a";
          else if (NR % 2) print v[(NR + 1) / 2];
          else printf "%.4f\n", (v[NR / 2] + v[NR / 2 + 1]) / 2 }'
}
paste -d, <(per_iter baseline) <(per_iter student) |
  awk -F, 'BEGIN { print "repeat,baseline_per_iter_ms,student_per_iter_ms" } { print NR "," $1 "," $2 }' \
  > "$LOGDIR/timing_runs.csv"
BASE_MED=$(per_iter baseline | median)
STU_MED=$(per_iter student | median)
if [[ "$BASE_MED" == n/a || "$STU_MED" == n/a ]]; then
  SPEEDUP=n/a
else
  SPEEDUP=$(awk -v b="$BASE_MED" -v s="$STU_MED" 'BEGIN { printf "%.2fx", b / s }')
fi

# 6. Can this account read GPU performance counters with Nsight Compute?
section "Nsight Compute permission check"
NCU_STATUS="ncu not found on PATH"
if command -v ncu > /dev/null; then
  run "$LOGDIR/ncu_check.log" ncu --launch-count 1 --section SpeedOfLight \
    ./mlp 100 37 53 29 10 1 --skip-ref --skip-baseline
  if grep -q "ERR_NVGPUCTRPERM" "$LOGDIR/ncu_check.log"; then
    NCU_STATUS="BLOCKED: no permission to read GPU performance counters"
  elif grep -qi "speed of light" "$LOGDIR/ncu_check.log"; then
    NCU_STATUS="works"
  else
    NCU_STATUS="unclear, see ncu_check.log"
  fi
fi

# 7. Summary
section "Summary"
{
  echo "Label:        $LABEL"
  echo "GPU:          $GPU_NAME"
  echo "Correctness:  $PASSED/$TOTAL cases pass"
  echo "Sanitizer:    memcheck=$MEM1 (ragged), memcheck=$MEM2 (all dims < 32), racecheck=$RACE, synccheck=$SYNC"
  echo "Timing:       $REPEATS repeats, --iters $ITERS, ./mlp $BENCH"
  echo "  median baseline per_iter_ms: $BASE_MED"
  echo "  median student  per_iter_ms: $STU_MED"
  echo "  speedup (median / median):   $SPEEDUP"
  echo "ncu:          $NCU_STATUS"
  echo "Logs:         $LOGDIR/"
  if (( FAILS > 0 )); then echo "STATUS:       $FAILS FAILURE(S), check the logs above"; else echo "STATUS:       all checks passed"; fi
} | tee "$LOGDIR/summary.txt"

exit $(( FAILS > 0 ? 1 : 0 ))
