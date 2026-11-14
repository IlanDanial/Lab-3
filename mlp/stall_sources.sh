#!/bin/bash
# Usage: bash stall_sources.sh logs/<run>/source_sass.csv
# Where warps stall, by SASS instruction (ncu SourceCounters warp-state samples).
gawk -v FPAT='([^,]*)|("([^"]|"")*")' '
  function unq(s) { gsub(/^"|"$/, "", s); gsub(/""/, "\"", s); return s }
  !hdr && /Source/ && /Address/ {
    for (i = 1; i <= NF; i++) {
      h = unq($i)
      if (h == "Source") cs = i
      if (h ~ /Warp Stall Sampling \(All/) ca = i
    }
    hdr = 1
    if (!cs || !ca) { print "stall columns not found; header was:"; print; exit 1 }
    next
  }
  hdr && NF >= ca {
    v = unq($ca) + 0
    if (v <= 0) next
    src = unq($cs); op = src
    sub(/^@!?U?P[0-9T]+ +/, "", op); split(op, w, /[ .;]/); op = w[1]
    tot += v; byop[op] += v; samp[src] += v
  }
  END {
    if (!hdr) { print "no source table in the CSV (see ncu_source.log)"; exit }
    if (!tot) { print "no stall samples found"; exit }
    PROCINFO["sorted_in"] = "@val_num_desc"
    printf "stall samples by instruction type (%d samples):", tot
    k = 0; for (o in byop) { if (++k > 8) break; printf "  %s %.1f%%", o, 100 * byop[o] / tot }
    printf "\ntop instructions:\n"
    k = 0; for (s in samp) { if (++k > 5) break; printf "  %5.1f%%  %s\n", 100 * samp[s] / tot, s }
  }' "$1"
