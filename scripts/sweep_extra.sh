#!/bin/bash
# Minimal: NS probe + FWD/INV for Darcy/Helmholtz/Burger
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/outputs/zeta_tuning/sweep_extra.log"
rm -f "$LOG"

run() {
  local pde=$1 task=$2 za=$3 zu=$4 zp=$5 dp=$6 ckpt=$7 ldby=$8
  echo "=== $pde $task a=$za u=$zu p=$zp ===" | tee -a "$LOG"
  python -u -m sampling.runner --config "configs/main/${pde}.yaml" \
    --override task="$task" --override zeta_obs_a="$za" --override zeta_obs_u="$zu" --override zeta_pde="$zp" \
    --override data_path="$dp" --override device=cuda --override num_obs=500 --override num_steps=1000 \
    --override sample_seed=42 --override save_plots=false --override output_dir=outputs/zeta_tuning \
    --override checkpoint_path="$ckpt" --override loadby="$ldby" \
    2>&1 | grep "'rel_l2_a\|'rel_l2_u" | tee -a "$LOG"
}

# === NS probe (sparse: tiny to moderate) ===
echo "=== NS PROBE ===" | tee -a "$LOG"
N="/large_storage/zhangxf/PDEdata/nsnonbounded/nsnonbounded_10000-128-128-10_1_new.mat"
CK="outputs/pretrained/formal/nsnonbounded/260712-121023-nsnonbounded-batch4-epoch300-accum8-float32/fm4nsnonbounded.pth"
for za in 1e-8 1e-4 1 10000; do
for zu in 1e-8 1e-4 1 10000; do
  run nsnonbounded both $za $zu 1 "$N" "$CK" h5py
done; done

# === FWD/INV Darcy ===
echo "=== DARCY FWD/INV ===" | tee -a "$LOG"
D="/large_storage/zhangxf/PDEdata/darcy/darcy_10000-128-128_1.mat"
C="outputs/pretrained/formal/darcy/260625-150746-darcy-batch4-epoch300-accum8-float32/fm4darcy.pth"
for zu in 50000 200000 500000 2000000 5000000; do
  run darcy forward 0 $zu 5 "$D" "$C" h5py
  run darcy inverse 0 $zu 5 "$D" "$C" h5py
done

# === FWD/INV Helmholtz ===
echo "=== HELMHOLTZ FWD/INV ===" | tee -a "$LOG"
H="/large_storage/zhangxf/PDEdata/helmholtz/helmholtz_10000-128-128_1.mat"
K="outputs/pretrained/formal/helmholtz/260625-150653-helmholtz-batch4-epoch300-accum8-float32/fm4helmholtz.pth"
for zu in 100000 1000000 5000000 20000000 50000000; do
  run helmholtz forward 0 $zu 0.01 "$H" "$K" scipy
  run helmholtz inverse 0 $zu 0.01 "$H" "$K" scipy
done

# === FWD/INV Burger ===
echo "=== BURGER FWD/INV ===" | tee -a "$LOG"
B="/large_storage/zhangxf/PDEdata/burgers/burger_10000-128-128_1.mat"
L="outputs/pretrained/formal/burger/260625-150439-burger-batch4-epoch300-accum8-float32/fm4burger.pth"
for zu in 100 1000 10000 100000 1000000; do
  run burger forward 0 $zu 1 "$B" "$L" scipy
  run burger inverse 0 $zu 1 "$B" "$L" scipy
done

echo "ALL DONE" | tee -a "$LOG"
