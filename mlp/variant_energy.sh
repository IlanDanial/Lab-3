#!/bin/bash -l
#SBATCH --job-name=gpu_lab3_variant_energy
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

# Energy at sustained load for kernel variants, with each variant's FFMA register-read stats.
# Each argument is one variant: comma-separated constexpr settings from student.cu.
# Usage (from mlp/): sbatch variant_energy.sh ORDER=0 ORDER=1 ORDER=2   or, on a GPU node: bash variant_energy.sh ...
#   env: ENERGY_ITERS=2000 ROUNDS=2
(( $# )) || { echo "usage: variant_energy.sh NAME=V[,NAME=V...] ..." >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
[[ -f Makefile && -f src/student.cu ]] || { echo "ERROR: run from lab3/mlp" >&2; exit 1; }
HERE=$PWD
module load gcc cuda 2>/dev/null
BUILD=$(mktemp -d)
mkdir -p "$BUILD/bins"

KEYS=()
for v in "$@"; do
  key=$(tr ',=' '_-' <<< "$v"); d="$BUILD/$key"; KEYS+=("$key")
  mkdir -p "$d"; cp -r Makefile include src sass_ffma.sh "$d/"
  IFS=, read -ra sets <<< "$v"
  for kv in "${sets[@]}"; do
    name=${kv%%=*}; val=${kv#*=}
    sed -i -E "s/^constexpr int $name = [0-9]+;/constexpr int $name = $val;/" "$d/src/student.cu"
    grep -q "^constexpr int $name = $val;" "$d/src/student.cu" || { echo "could not set $kv" >&2; exit 1; }
  done
  (make -C "$d" > "$d/build.log" 2>&1 && cp "$d/mlp" "$BUILD/bins/$key" || echo "BUILD FAILED: $key") &
  (( ${#KEYS[@]} % 4 == 0 )) && wait
done
wait

for key in "${KEYS[@]}"; do
  [[ -x $BUILD/bins/$key ]] && echo "$key: $(cd "$BUILD/$key" && bash sass_ffma.sh mlp)"
done
cd "$BUILD/bins" && bash "$HERE/energy_compare.sh" $(for k in "${KEYS[@]}"; do [[ -x $k ]] && printf './%s ' "$k"; done)
cd "$HERE" && rm -rf "$BUILD"
