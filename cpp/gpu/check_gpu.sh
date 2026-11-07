#!/bin/bash
# Reads a GPU kernel's output on stdin, echoes it, and exits nonzero if any line
# reports MISMATCH, FAIL or a CUDA error, or if any printed relative error
# ("rel=X" or "rel X") exceeds the tolerance given as $1 (default 1e-8).
tol=${1:-1e-8}
awk -v tol="$tol" '
  { print }
  /MISMATCH|FAIL|CUDA error|cuFFT error/ { bad = 1 }
  {
    line = $0
    while (match(line, /rel[ =]+[-+0-9.eE]+/)) {
      tok = substr(line, RSTART, RLENGTH); sub(/rel[ =]+/, "", tok)
      seen = 1; if (tok + 0 > tol + 0) bad = 1
      line = substr(line, RSTART + RLENGTH)
    }
  }
  END {
    if (!seen) { print "  no relative error reported"; exit 1 }
    if (bad)   { print "  CHECK FAILED"; exit 1 }
    print "  ok"
  }'
