#!/bin/bash
# Simple zeta sweep for ONE PDE
# Usage: bash scripts/sweep_pde.sh <pde> <device>
PDE=$1
DEV=${2:-cuda}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="outputs/zeta_tuning/sweep_${PDE}.log"
rm -f "$LOG"

run() {
  local task=$1 za=$2 zu=$3 zp=$4 tag=$5 dp=$6
  echo "=== $PDE $task a=$za u=$zu p=$zp $tag ===" | tee -a "$LOG"
  python -u -m sampling.runner --config "configs/main/${PDE}.yaml" \
    --override task="$task" --override zeta_obs_a="$za" --override zeta_obs_u="$zu" --override zeta_pde="$zp" \
    --override data_path="$dp" --override device="$DEV" --override num_obs=500 --override num_steps=1000 \
    --override sample_seed=42 --override save_plots=false --override output_dir=outputs/zeta_tuning \
    ${CHECKPOINT:+--override checkpoint_path="$CHECKPOINT"} \
    ${LOADBY:+--override loadby="$LOADBY"} \
    2>&1 | grep "'rel_l2_a\|'rel_l2_u" | tee -a "$LOG"
}

case $PDE in
  darcy)
    CHECKPOINT="outputs/pretrained/formal/darcy/260625-150746-darcy-batch4-epoch300-accum8-float32/fm4darcy.pth"
    LOADBY="h5py"
    TRAIN=/large_storage/zhangxf/PDEdata/darcy/darcy_10000-128-128_1.mat
    TEST=/large_storage/zhangxf/PDEdata/darcy/darcy_test_10000-128-128.mat
    A_SWEEP="50 100 200"
    U_SWEEP="50000 100000 200000"
    P_SWEEP="1 2.5 5"
    ;;
  helmholtz)
    CHECKPOINT="outputs/pretrained/formal/helmholtz/260625-150653-helmholtz-batch4-epoch300-accum8-float32/fm4helmholtz.pth"
    LOADBY="scipy"
    TRAIN=/large_storage/zhangxf/PDEdata/helmholtz/helmholtz_10000-128-128_1.mat
    TEST=/large_storage/zhangxf/PDEdata/helmholtz/helmholtz_test_10000-128-128.mat
    A_SWEEP="5000 20000 100000"
    U_SWEEP="5000000 20000000 100000000"
    P_SWEEP="0.01 0.1 1"
    ;;
  nsnonbounded)
    CHECKPOINT="outputs/pretrained/formal/nsnonbounded/260712-121023-nsnonbounded-batch4-epoch300-accum8-float32/fm4nsnonbounded.pth"
    LOADBY="h5py"
    TRAIN=/large_storage/zhangxf/PDEdata/nsnonbounded/nsnonbounded_10000-128-128-10_1_new.mat
    TEST=/large_storage/zhangxf/PDEdata/nsnonbounded/nsnonbounded_test_10000-128-128-10.mat
    A_SWEEP="5000 30000 100000"
    U_SWEEP="50000 300000 1000000"
    P_SWEEP="0.1 1 10"
    ;;
  burger)
    CHECKPOINT="outputs/pretrained/formal/burger/260625-150439-burger-batch4-epoch300-accum8-float32/fm4burger.pth"
    LOADBY="scipy"
    TRAIN=/large_storage/zhangxf/PDEdata/burgers/burger_10000-128-128_1.mat
    TEST=/large_storage/zhangxf/PDEdata/burgers/burger_test_10000-128-128.mat
    A_SWEEP="0.01 0.1 10"
    U_SWEEP="100 10000 100000"
    P_SWEEP="0.1 1 10"
    ;;
esac

echo "SWEEP $PDE on $DEV" | tee -a "$LOG"
echo "A: $A_SWEEP  U: $U_SWEEP  P: $P_SWEEP" | tee -a "$LOG"

# both task
for za in $A_SWEEP; do for zu in $U_SWEEP; do for zp in $P_SWEEP; do
  run both $za $zu $zp train "$TRAIN"
done; done; done

# forward task
for zu in $U_SWEEP; do for zp in $P_SWEEP; do
  run forward 0 $zu $zp train "$TRAIN"
done; done

# inverse task (sweep u)
for zu in $U_SWEEP; do for zp in $P_SWEEP; do
  run inverse 0 $zu $zp train "$TRAIN"
done; done

echo "$PDE DONE" | tee -a "$LOG"
