#!/bin/bash
# Usage: bash sass_ffma.sh [binary]   (default: mlp)
# Register reads of every FFMA in matmul_student's compiled code (SASS):
# reuse-cache hits per operand slot, bank reads per FFMA, and FFMAs whose bank reads
# share a bank under a 2-bank (register parity) model, an estimate since Ada's banking is undocumented.
cuobjdump -sass "${1:-mlp}" | awk '
  /Function : / { infn = ($0 ~ /matmul_student/) }
  infn && /[ \t]FFMA[ .]/ {
    s = $0; sub(/.*FFMA[^ ]* +/, "", s); sub(/ *;.*/, "", s)
    if (split(s, op, ",") < 4) next
    f++; nb = 0
    for (k = 2; k <= 4; k++) {
      o = op[k]; gsub(/[ |-]/, "", o)
      if (o ~ /^R[0-9]+\.reuse/) { hit[k]++; continue }
      if (o ~ /^R[0-9]+/) { r = o; sub(/^R/, "", r); sub(/[^0-9].*/, "", r); par[++nb] = r % 2; reads++ }
    }
    same = 0
    for (a = 1; a <= nb; a++) for (b = a + 1; b <= nb; b++) if (par[a] == par[b]) same = 1
    conf += same
  }
  END {
    if (!f) { print "no FFMA found in matmul_student"; exit }
    printf "%d FFMA | reuse hits A %.1f%% B %.1f%% C %.1f%% | bank reads/FFMA %.2f | same-bank pairs %.1f%% (2-bank model, est.)\n",
      f, 100 * hit[2] / f, 100 * hit[3] / f, 100 * hit[4] / f, reads / f, 100 * conf / f
  }'
