#!/bin/bash
# Usage: bash profile_table.sh logs/<run>/profile.txt [logs/<run>/profile_raw.txt]
# One row per kernel, first forward pass only (driver runs warm-up + timed pass).
awk '
  # Skip ncu rule prose (WRN/OPT/INF paragraphs) so it cannot be parsed as metrics
  /^ *(WRN|OPT|INF) / { inrule = 1; next }
  inrule && (/^ *$/ || /^ *-----/) { inrule = 0; next }
  inrule { next }

  /\)x\(/ {
    n++; s = $0; sub(/\(.*/, "", s); k = split(s, a, " "); name[n] = a[k]   # last word before "(", skips "void"
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
  /Registers Per Thread/      { regs[n] = $NF }
  /Achieved Occupancy/        { occ[n] = $NF }
  /Warp Cycles Per Issued Instruction/ { wc[n] = $NF }
  /Eligible Warps Per Scheduler/       { el[n] = $NF }
  /No Eligible/                        { ne[n] = $NF }
  /Issue Slots Busy/                   { isb[n] = $NF }
  /smsp__inst_executed\.sum/           { v = $NF; gsub(/,/, "", v); ins[n] = sprintf("%.1f", v / 1e6) }
  END {
    half = (n % 2 == 0) ? n / 2 : n
    printf "%-22s %-22s %8s %8s %7s %6s %6s %6s %6s %5s %5s\n", "kernel", "grid x block", "ms", "sect/req", "L1hit%", "L1%", "L2%", "DRAM%", "SM%", "regs", "occ%"
    for (i = 1; i <= half; i++) {
      spr = (req[i] > 0) ? sprintf("%.2f", sec[i] / req[i]) : "-"
      printf "%-22s %-22s %8.3f %8s %7.1f %6.1f %6.1f %6.1f %6.1f %5d %5.1f\n", name[i], grid[i], dur[i], spr, l1hit[i], l1[i], l2[i], dram[i], sm[i], regs[i], occ[i]
      total += dur[i]
    }
    printf "total kernel time, one pass: %.3f ms (ncu clocks, not comparable to timed runs)\n", total
    if (wc[1] != "") {
      printf "\n%-22s %-22s %10s %9s %9s %8s %10s\n", "kernel", "grid x block", "cyc/issue", "eligible", "noElig%", "issue%", "instr(M)"
      for (i = 1; i <= half; i++)
        printf "%-22s %-22s %10s %9s %9s %8s %10s\n", name[i], grid[i], wc[i], el[i], ne[i], isb[i], (ins[i] != "" ? ins[i] : "-")
    }
  }
' "$1"

# Stall breakdown per multiply kernel (warps per issued instruction), from the raw page
[[ -f "$2" ]] || exit 0
awk '
  FNR == NR { if (/\)x\(/) { n++; s = $0; sub(/\(.*/, "", s); k = split(s, a, " "); name[n] = a[k] } next }
  /smsp__average_warps_issue_stalled_barrier_per_issue_active\.ratio/ { b++ }   # first reason = new kernel
  /smsp__average_warps_issue_stalled_[a-z_]+_per_issue_active\.ratio/ {
    r = $1; sub(/^smsp__average_warps_issue_stalled_/, "", r); sub(/_per_issue_active\.ratio$/, "", r)
    st[b, r] = $NF; tot[b] += $NF
  }
  END {
    split("selected not_selected dispatch_stall long_scoreboard math_pipe_throttle barrier wait lg_throttle short_scoreboard mio_throttle", key, " ")
    split("sel notsel dispatch longsb maththr barrier wait lgthr shortsb mio", lab, " ")
    half = (n % 2 == 0) ? n / 2 : n
    printf "\n%-32s %6s", "stalls (warps per issued instr)", "total"
    for (c = 1; c <= 10; c++) printf " %8s", lab[c]
    printf "\n"
    for (i = 1; i <= half && i <= b; i++) {
      if (name[i] !~ /matmul/) continue
      printf "%-24s layer %d %6.2f", name[i], ++layer, tot[i]
      for (c = 1; c <= 10; c++) printf " %8.2f", st[i, key[c]]
      printf "\n"
    }
  }
' "$1" "$2"
