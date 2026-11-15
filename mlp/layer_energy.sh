#!/bin/bash -l
#SBATCH --job-name=gpu_lab3_layer_energy
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

# Energy per layer at sustained load, per tile config. Each test binary runs one full pass (real
# inputs for every layer), then only layer L on every later call. Full-pass binaries check the sum.
# Usage (from mlp/): sbatch layer_energy.sh   or, on a GPU node: bash layer_energy.sh
#   env: CONFIGS="8:8:16:3 16:8:32:2" (RM:RN:BK:NBUF) ENERGY_ITERS=6000 ROUNDS=2
CONFIGS=${CONFIGS:-"8:8:16:3 16:8:32:2"}
export ENERGY_ITERS=${ENERGY_ITERS:-6000}
export ROUNDS=${ROUNDS:-2}

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
[[ -f Makefile && -f src/student.cu ]] || { echo "ERROR: run from lab3/mlp" >&2; exit 1; }
HERE=$PWD
module load gcc cuda 2>/dev/null
BUILD=$(mktemp -d)
mkdir -p "$BUILD/bins"

KEYS=()
for c in $CONFIGS; do
  IFS=: read -r m n bk nb <<< "$c"
  for L in all 1 2 3; do
    key="${m}x${n}_${L}"; d="$BUILD/$key"; KEYS+=("$key")
    mkdir -p "$d"; cp -r Makefile include src "$d/"
    sed -i -E "s/^constexpr int RM = [0-9]+;/constexpr int RM = $m;/; s/^constexpr int RN = [0-9]+;/constexpr int RN = $n;/;
               s/^constexpr int BK = [0-9]+;/constexpr int BK = $bk;/; s/^constexpr int NBUF = [0-9]+;/constexpr int NBUF = $nb;/" "$d/src/student.cu"
    if [[ $L != all ]]; then
      sed -i "/^void mlp_student_compute/,/^}/ s|^  for (int l = 0; l < MLP_LAYERS; l++) {\$|&\n    { static int calls = 0; if (calls++ >= MLP_LAYERS \&\& l != $((L - 1))) continue; }  // layer_energy|" "$d/src/student.cu"
      grep -q "// layer_energy" "$d/src/student.cu" || { echo "could not patch $key" >&2; exit 1; }
    fi
    (make -C "$d" > "$d/build.log" 2>&1 && cp "$d/mlp" "$BUILD/bins/$key" || echo "BUILD FAILED: $key (see $d/build.log)") &
    (( ${#KEYS[@]} % 4 == 0 )) && wait
  done
done
wait

echo "ms/pass and mJ/pass are per call: one layer for _1/_2/_3, the full pass for _all (GFLOP/J only valid for _all)"
cd "$BUILD/bins" && ls -1 | grep -q . && bash "$HERE/energy_compare.sh" $(for k in "${KEYS[@]}"; do [[ -x $k ]] && printf './%s ' "$k"; done)
cd "$HERE" && rm -rf "$BUILD"
