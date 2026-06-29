#!/bin/bash
# Full CPU regression: build every program, then run the MMS suite and flag any
# program whose output contains fail/nan/inf/error. Writes full_regression.log.
cd "$(dirname "$0")/.." || exit 1
NPROC=${NPROC:-$(nproc)}
{
  echo "=== full build -j$NPROC ==="; make -j"$NPROC" all 2>&1 | grep -E 'Error|error:' | head
  echo "=== full MMS regression ==="
  for t in mms_check mms_ns mms_bmf mms_poisson mms_compressible mms_burgers mms_wall2d mms_infsup mms_wallns2d mms_wallns3d mms_tgdecay mms_cut1d mms_cut2d mms_cut3d mms_comp_cut3d mms_hill2d mms_hill3d mms_hill_metric; do
    R=$(OMP_NUM_THREADS=$NPROC timeout 300 ./bin/$t 2>&1 | grep -ciE 'fail|nan|inf|error' )
    echo "  $t: $([ "$R" = "0" ] && echo GREEN || echo "CHECK ($R flags)")"
  done
  echo "=== DONE full_regression ==="
} > full_regression.log 2>&1
