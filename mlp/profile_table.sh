#!/bin/bash
# Usage: bash profile_table.sh logs/<run>/profile.txt
# One row per kernel, first forward pass only (driver runs warm-up + timed pass).
awk '
  /\)x\(/ {
    n++; s = $1; sub(/\(.*/, "", s); name[n] = s
    match($0, /\([0-9, ]+\)x\([0-9, ]+\)/); g = substr($0, RSTART, RLENGTH); gsub(/ /, "", g); grid[n] = g
  }
  /^ *Duration/ { v = $NF; if ($(NF-1) == "us") v /= 1000; if ($(NF-1) == "ns") v /= 1e6; dur[n] = v }
  /op_ld\.sum/ && /requests/ { v = $NF; gsub(/,/, "", v); req[n] = v }
  /op_ld\.sum/ && /sectors/  { v = $NF; gsub(/,/, "", v); sec[n] = v }
  /L1\/TEX Hit Rate/          { l1hit[n] = $NF }
  /L1\/TEX Cache Throughput/  { l1[n] = $NF }
  /L2 Cache Throughput/       { l2[n] = $NF }
  /DRAM Throughput/ && $(NF-1) == "%" { dram[n] = $NF }
  /Compute \(SM\) Throughput/ { sm[n] = $NF }
  END {
    half = (n % 2 == 0) ? n / 2 : n
    printf "%-22s %-22s %8s %8s %7s %6s %6s %6s %6s\n", "kernel", "grid x block", "ms", "sect/req", "L1hit%", "L1%", "L2%", "DRAM%", "SM%"
    for (i = 1; i <= half; i++) {
      spr = (req[i] > 0) ? sprintf("%.2f", sec[i] / req[i]) : "-"
      printf "%-22s %-22s %8.3f %8s %7.1f %6.1f %6.1f %6.1f %6.1f\n", name[i], grid[i], dur[i], spr, l1hit[i], l1[i], l2[i], dram[i], sm[i]
      total += dur[i]
    }
    printf "total kernel time, one pass: %.3f ms (ncu clocks, not comparable to timed runs)\n", total
  }
' "$1"
