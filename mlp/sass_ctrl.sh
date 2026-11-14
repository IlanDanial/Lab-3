#!/bin/bash
# Usage: bash sass_ctrl.sh [binary]   (default: mlp)
# Control-flow and sync instructions in matmul_student's compiled code (static counts).
cuobjdump -sass "${1:-mlp}" | awk '
  /Function : / { infn = 0; if ($0 ~ /matmul_student/ && !seen) { infn = 1; seen = 1 } }
  infn && /\/\*[0-9a-f]+\*\// {
    s = $0; sub(/^ *\/\*[0-9a-f]+\*\/ +/, "", s); sub(/^@!?U?P[0-9T]+ +/, "", s); split(s, w, /[ .;]/)
    if (w[1] ~ /^(BRA|BSSY|BSYNC|CALL|RET|BAR|EXIT)$/) n[w[1]]++
    if (w[1] != "") tot++
  }
  END {
    printf "%d instructions | BRA %d  BSSY %d  BSYNC %d  CALL %d  BAR %d\n", tot, n["BRA"], n["BSSY"], n["BSYNC"], n["CALL"], n["BAR"]
  }'
