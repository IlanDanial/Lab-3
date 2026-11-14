#!/bin/bash
# Usage (from mlp/, after module load cuda): bash sass_variants.sh [orders]   default: "0 1 2"
# Compiles each FFMA ORDER in a temp copy (no GPU needed) and prints its register-read stats.
ORDERS=${1:-"0 1 2"}
TMP="${TMPDIR:-/tmp}/sassvar_$$"
fail=0
for o in $ORDERS; do
  d="$TMP/order$o"
  mkdir -p "$d" && cp -r Makefile include src sass_ffma.sh "$d/"
  sed -i -E "s/^constexpr int ORDER = [0-9]+;/constexpr int ORDER = $o;/" "$d/src/student.cu"
  if ! grep -q "^constexpr int ORDER = $o;" "$d/src/student.cu"; then echo "ORDER=$o: could not set"; continue; fi
  if make -C "$d" > "$d/build.log" 2>&1; then
    echo "ORDER=$o: $(cd "$d" && bash sass_ffma.sh mlp)"
  else
    echo "ORDER=$o: build failed, see $d/build.log"; fail=1
  fi
done
(( fail )) || rm -rf "$TMP"   # keep logs if a build failed
