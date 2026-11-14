#!/bin/bash -l
#SBATCH --job-name=gpu_lab3_sweep
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gpus=1
#SBATCH --time=01:45:00
#SBATCH --account=eel4930
#SBATCH --qos=eel4930

# Tile sweep: per-thread tile RM x RN, tile depth BK, pipeline stages NBUF, against a reference commit.
# Usage (from mlp/): sbatch sweep.slurm
#   env: SHAPES="8x8 8x16" DEPTHS="16:2 16:3" (BK:NBUF) REF=5a35b86 REPEATS=5 ITERS=100 TOPN=4
# Output: logs/sweep_<jobid>/sweep.csv, energy.txt, raw logs per config

SHAPES=${SHAPES:-"8x8 8x12 8x16 16x8 12x12 12x16"}
DEPTHS=${DEPTHS:-"8:2 8:3 8:4 16:2 16:3 16:4 32:2"}
REF=${REF:-5a35b86}   # v13
REPEATS=${REPEATS:-5}
ITERS=${ITERS:-100}
TOPN=${TOPN:-4}
BENCH="4093 1000 1990 1010 500 7"   # sizes.txt benchmark

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [[ ! -f Makefile || ! -f src/student.cu ]]; then
  echo "ERROR: submit from lab3/mlp" >&2
  exit 1
fi
HERE=$PWD

module purge
module load gcc cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

LOGDIR="logs/sweep_${SLURM_JOB_ID:-local}"
BUILD="${TMPDIR:-/tmp}/sweep_${SLURM_JOB_ID:-local}"   # binaries stay out of the repo
mkdir -p "$LOGDIR" "$BUILD"
{
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv
  nvcc --version | tail -2
  echo "ref: $(git log -1 --format='%h %s' "$REF" 2>&1)"
} > "$LOGDIR/env.txt" 2>&1
cp src/student.cu "$LOGDIR/student_template.cu"

srcparam() { grep -oE "^constexpr int $2 = [0-9]+" "$1" | grep -oE '[0-9]+$'; }
params() { [[ $1 =~ ^RM([0-9]+)_RN([0-9]+)_BK([0-9]+)_NB([0-9]+)$ ]] && echo "${BASH_REMATCH[*]:1}"; }

# Shared memory per block = 4 B x NBUF x BK x TILE x (RM + RN). Skip configs over the 99 KB limit,
# and 2-block configs (RM*RN <= 64) over 48 KB, which would fall to 1 block per SM.
TILE=$(srcparam src/student.cu TILE)
CONFIGS=(ref); SKIPPED=()
for s in $SHAPES; do
  m=${s%x*}; n=${s#*x}
  for dn in $DEPTHS; do
    bk=${dn%:*}; nb=${dn#*:}
    smem=$((4 * nb * bk * TILE * (m + n)))
    if (( smem > 99 * 1024 || (m * n <= 64 && smem > 48 * 1024) )); then SKIPPED+=("${s}:${bk}:${nb}"); continue; fi
    CONFIGS+=("RM${m}_RN${n}_BK${bk}_NB${nb}")
  done
done
echo "configs: ${#CONFIGS[@]}  skipped (shared memory): ${SKIPPED[*]:-none}"

# build <key>: copy the tree, set the parameters (or check out the reference kernel), make
build() {
  local key=$1 d="$BUILD/$1" log="$LOGDIR/$1" m n bk nb v name val
  mkdir -p "$d" "$log"
  cp -r Makefile include src "$d/"
  if [[ $key == ref ]]; then
    git show "$REF:./src/student.cu" > "$d/src/student.cu" 2> "$log/build.log" || return 1
  else
    read -r m n bk nb <<< "$(params "$key")"
    sed -i -E "s/^constexpr int RM = [0-9]+;/constexpr int RM = $m;/; s/^constexpr int RN = [0-9]+;/constexpr int RN = $n;/;
               s/^constexpr int BK = [0-9]+;/constexpr int BK = $bk;/; s/^constexpr int NBUF = [0-9]+;/constexpr int NBUF = $nb;/" "$d/src/student.cu"
    for v in "RM $m" "RN $n" "BK $bk" "NBUF $nb"; do
      read -r name val <<< "$v"
      grep -q "^constexpr int $name = $val;" "$d/src/student.cu" || { echo "could not set $name=$val" > "$log/build.log"; return 1; }
    done
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

declare -A CHECK REGS LOCAL SMEM P
echo "=== Correctness (CPU reference) and resources ==="
for key in "${CONFIGS[@]}"; do
  d="$BUILD/$key"; log="$LOGDIR/$key"
  if [[ $key == ref ]]; then
    r=$(srcparam "$d/src/student.cu" R)
    P[$key]="${r:-?},${r:-?},$(srcparam "$d/src/student.cu" BK),$(srcparam "$d/src/student.cu" NBUF)"
  else
    P[$key]=$(params "$key" | tr ' ' ,)
  fi
  if [[ ! -x "$d/mlp" ]]; then CHECK[$key]=BUILD_FAIL; echo "$key: BUILD_FAIL"; continue; fi
  res=$(cuobjdump -res-usage "$d/mlp" 2>/dev/null | grep -A1 'Function.*matmul_student' | grep 'REG:')
  echo "$res" > "$log/resources.txt"
  REGS[$key]=$(grep -oE 'REG:[0-9]+' <<< "$res" | cut -d: -f2 | sort -g | tail -1)        # max over both kernels
  LOCAL[$key]=$(grep -oE 'LOCAL:[0-9]+' <<< "$res" | cut -d: -f2 | sort -g | tail -1)
  if [[ $key == ref ]]; then
    SMEM[$key]=$(grep -oE 'SHARED:[0-9]+' <<< "$res" | cut -d: -f2 | sort -g | tail -1)  # static
  else
    IFS=, read -r m n bk nb <<< "${P[$key]}"; SMEM[$key]=$((4 * nb * bk * TILE * (m + n)))  # dynamic
  fi
  CHECK[$key]=PASS
  for s in "100 37 53 29 10 1" "33 17 19 23 5 4" "517 129 260 513 37 2" "$BENCH --iters 1"; do
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

declare -A MED MIN MAX RUNS
for key in "${CONFIGS[@]}"; do
  t=$(grep '^\[student\]' "$LOGDIR/$key/timing.log" 2>/dev/null | sed -E 's/.*per_iter_ms=([0-9.]+).*/\1/')
  if [[ -z "$t" ]]; then MED[$key]=n/a; MIN[$key]=n/a; MAX[$key]=n/a; RUNS[$key]=0
  else MED[$key]=$(median <<< "$t"); MIN[$key]=$(sort -g <<< "$t" | head -1); MAX[$key]=$(sort -g <<< "$t" | tail -1); RUNS[$key]=$(grep -c . <<< "$t"); fi
done

CSV="$LOGDIR/sweep.csv"
echo "config,RM,RN,BK,NBUF,check,median_ms,min_ms,max_ms,runs,vs_ref,regs,local_bytes,smem_bytes,layer1_ms_ncu,L1_pct,SM_pct,theor_occ_pct,achieved_occ_pct,shld_inst,shld_wavefronts,shld_conflicts" > "$CSV"
for key in "${CONFIGS[@]}"; do
  vs=$(awk -v r="${MED[ref]}" -v m="${MED[$key]}" 'BEGIN { if (r + 0 > 0 && m + 0 > 0) printf "%.3f", r / m; else print "n/a" }')
  if [[ -f "$LOGDIR/$key/ncu.txt" ]]; then nf=$(ncu_fields "$LOGDIR/$key/ncu.txt"); else nf=",,,,,,,"; fi
  echo "$key,${P[$key]},${CHECK[$key]},${MED[$key]},${MIN[$key]},${MAX[$key]},${RUNS[$key]},$vs,${REGS[$key]},${LOCAL[$key]},${SMEM[$key]},$nf" >> "$CSV"
done

echo "=== Results (fastest first; vs_ref = ref ms / config ms) ==="
{ head -1 "$CSV"
  awk -F, 'NR > 1 && $7 != "n/a"' "$CSV" | sort -t, -k7 -g
  awk -F, 'NR > 1 && $7 == "n/a"' "$CSV"; } | cut -d, -f1-14 | column -s, -t

# Timed runs sit at the power cap, so energy per pass decides; check the fastest at sustained load
mapfile -t TOP < <(awk -F, 'NR > 1 && $1 != "ref" && $6 == "PASS" && $7 != "n/a" { print $7, $1 }' "$CSV" | sort -g | head -"$TOPN" | cut -d' ' -f2)
BINS=()
[[ ${CHECK[ref]} == PASS ]] && BINS+=(ref)
BINS+=("${TOP[@]}")
if (( ${#BINS[@]} )); then
  echo "=== Energy at sustained load: ${BINS[*]} ==="
  mkdir -p "$BUILD/bins"
  for key in "${BINS[@]}"; do cp "$BUILD/$key/mlp" "$BUILD/bins/$key"; done
  (cd "$BUILD/bins" && bash "$HERE/energy_compare.sh" $(printf './%s ' "${BINS[@]}")) | tee "$LOGDIR/energy.txt"
fi

echo "Logs: $LOGDIR/ (run run.slurm on the winner for the full tests and sanitizer)"
rm -rf "$BUILD"
