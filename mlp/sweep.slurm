#!/bin/bash -l
#SBATCH --job-name=gpu_lab3_sweep
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --account=eel4930
#SBATCH --qos=eel4930

# Bonus sweep: R (outputs per thread side) x BK (tile depth) of the final kernel.
# Usage (from mlp/): sbatch sweep.slurm   env: RS="1 2 4 8" BKS="8 16 32" REPEATS=5 ITERS=100
# Output: logs/sweep_<jobid>/sweep.csv, raw logs per config

RS=${RS:-"1 2 4 8"}
BKS=${BKS:-"8 16 32"}
REPEATS=${REPEATS:-5}
ITERS=${ITERS:-100}
BENCH="4093 1000 1990 1010 500 7"   # sizes.txt benchmark

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [[ ! -f Makefile || ! -f src/student.cu ]]; then
  echo "ERROR: submit from lab3/mlp" >&2
  exit 1
fi

module purge
module load gcc cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

LOGDIR="logs/sweep_${SLURM_JOB_ID:-local}"
BUILD="${TMPDIR:-/tmp}/sweep_${SLURM_JOB_ID:-local}"   # binaries stay out of the repo
mkdir -p "$LOGDIR" "$BUILD"
{
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv
  nvcc --version | tail -2
} > "$LOGDIR/env.txt" 2>&1
cp src/student.cu "$LOGDIR/student_template.cu"

CONFIGS=()
for r in $RS; do for bk in $BKS; do CONFIGS+=("R${r}_BK${bk}"); done; done
key_r()  { local r=${1#R}; echo "${r%%_*}"; }
key_bk() { echo "${1##*BK}"; }

# build <key>: copy the tree, set R and BK in the copy only, make
build() {
  local r bk d="$BUILD/$1" log="$LOGDIR/$1"
  r=$(key_r "$1"); bk=$(key_bk "$1")
  mkdir -p "$d" "$log"
  cp -r Makefile include src "$d/"
  sed -i -E "s/^constexpr int R = [0-9]+;/constexpr int R = $r;/; s/^constexpr int BK = [0-9]+;/constexpr int BK = $bk;/" "$d/src/student.cu"
  if ! grep -q "^constexpr int R = $r;" "$d/src/student.cu" || ! grep -q "^constexpr int BK = $bk;" "$d/src/student.cu"; then
    echo "could not set R=$r BK=$bk" > "$log/build.log"
    return 1
  fi
  make -C "$d" > "$log/build.log" 2>&1
}

echo "=== Build ${#CONFIGS[@]} configs ==="
i=0
for key in "${CONFIGS[@]}"; do
  build "$key" &
  i=$((i + 1)); (( i % ${SLURM_CPUS_PER_TASK:-4} == 0 )) && wait
done
wait

declare -A CHECK REGS LOCAL SMEM
echo "=== Correctness (CPU reference) and resources ==="
for key in "${CONFIGS[@]}"; do
  d="$BUILD/$key"; log="$LOGDIR/$key"
  if [[ ! -x "$d/mlp" ]]; then CHECK[$key]=BUILD_FAIL; echo "$key: BUILD_FAIL"; continue; fi
  res=$(cuobjdump -res-usage "$d/mlp" 2>/dev/null | grep -A1 'Function.*matmul_student' | tail -1)
  echo "$res" > "$log/resources.txt"
  REGS[$key]=$(grep -oE 'REG:[0-9]+' <<< "$res" | cut -d: -f2)
  LOCAL[$key]=$(grep -oE 'LOCAL:[0-9]+' <<< "$res" | cut -d: -f2)
  SMEM[$key]=$(grep -oE 'SHARED:[0-9]+' <<< "$res" | cut -d: -f2)
  CHECK[$key]=PASS
  for s in "100 37 53 29 10 1" "33 17 19 23 5 4" "$BENCH --iters 1"; do
    "$d/mlp" $s >> "$log/check.log" 2>&1 || CHECK[$key]=FAIL   # $s unquoted: split into args
  done
  echo "$key: ${CHECK[$key]}  regs=${REGS[$key]} local=${LOCAL[$key]} smem=${SMEM[$key]}"
done

# Round-robin so GPU drift spreads evenly over configs
echo "=== Timing: $REPEATS rounds x ${#CONFIGS[@]} configs, --iters $ITERS ==="
for rep in $(seq 1 "$REPEATS"); do
  for key in "${CONFIGS[@]}"; do
    [[ ${CHECK[$key]} == PASS ]] || continue
    "$BUILD/$key/mlp" $BENCH --iters "$ITERS" --skip-ref --skip-baseline >> "$LOGDIR/$key/timing.log" 2>&1
  done
  echo "round $rep done"
done

echo "=== ncu: layer 1 multiply per config ==="
for key in "${CONFIGS[@]}"; do
  [[ ${CHECK[$key]} == PASS ]] || continue
  ncu --kernel-name regex:matmul --launch-count 1 \
    --section SpeedOfLight --section Occupancy --section LaunchStats \
    --metrics smsp__inst_executed_op_shared_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    "$BUILD/$key/mlp" $BENCH --iters 1 --skip-ref --skip-baseline > "$LOGDIR/$key/ncu.txt" 2>&1
done

median() {
  sort -g | awk '{v[NR] = $1}
    END { if (NR % 2) print v[(NR + 1) / 2]; else printf "%.4f\n", (v[NR / 2] + v[NR / 2 + 1]) / 2 }'
}
ncu_fields() {
  awk '/^ *Duration/ { v = $NF; if ($(NF-1) == "us") v /= 1000; d = v }
       /L1\/TEX Cache Throughput/  { l1 = $NF }
       /Compute \(SM\) Throughput/ { sm = $NF }
       /Theoretical Occupancy/     { th = $NF }
       /Achieved Occupancy/        { ac = $NF }
       /inst_executed_op_shared_ld\.sum/               { v = $NF; gsub(/,/, "", v); si = v }
       /wavefronts_mem_shared_op_ld\.sum/              { v = $NF; gsub(/,/, "", v); sw = v }
       /bank_conflicts_pipe_lsu_mem_shared_op_ld\.sum/ { v = $NF; gsub(/,/, "", v); sc = v }
       END { printf "%s,%s,%s,%s,%s,%s,%s,%s", d, l1, sm, th, ac, si, sw, sc }' "$1"
}

CSV="$LOGDIR/sweep.csv"
echo "R,BK,check,median_ms,min_ms,max_ms,runs,regs,local_bytes,smem_bytes,layer1_ms_ncu,L1_pct,SM_pct,theor_occ_pct,achieved_occ_pct,shld_inst,shld_wavefronts,shld_conflicts" > "$CSV"
for key in "${CONFIGS[@]}"; do
  t=$(grep '^\[student\]' "$LOGDIR/$key/timing.log" 2>/dev/null | sed -E 's/.*per_iter_ms=([0-9.]+).*/\1/')
  if [[ -z "$t" ]]; then med=n/a; mn=n/a; mx=n/a; runs=0
  else med=$(median <<< "$t"); mn=$(sort -g <<< "$t" | head -1); mx=$(sort -g <<< "$t" | tail -1); runs=$(grep -c . <<< "$t"); fi
  if [[ -f "$LOGDIR/$key/ncu.txt" ]]; then nf=$(ncu_fields "$LOGDIR/$key/ncu.txt"); else nf=",,,,,,,"; fi
  echo "$(key_r "$key"),$(key_bk "$key"),${CHECK[$key]},$med,$mn,$mx,$runs,${REGS[$key]},${LOCAL[$key]},${SMEM[$key]},$nf" >> "$CSV"
done

echo "=== Results ==="
column -s, -t < "$CSV"
best=$(awk -F, 'NR > 1 && $3 == "PASS" && $4 != "n/a" { print $4, $1, $2 }' "$CSV" | sort -g | head -1)
read -r best_ms best_r best_bk <<< "$best"
echo "Best: R=$best_r BK=$best_bk at $best_ms ms (run run.slurm on it for the full tests and sanitizer)"
echo "Logs: $LOGDIR/"
rm -rf "$BUILD"
