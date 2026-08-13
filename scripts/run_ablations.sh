#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# FM4PDE 消融实验脚本 — 支持运行全部或单个消融组
#
# 用法:
#   bash scripts/run_ablations.sh                         # 运行全部 9 组
#   bash scripts/run_ablations.sh guidance_components     # 只运行 guidance_components
#   bash scripts/run_ablations.sh guidance_components loss_state  # 多个
#   bash scripts/run_ablations.sh --help                  # 列出所有组
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT="outputs/ablations"
mkdir -p "${OUTPUT}"

# ── ablation group registry ───────────────────────────────────────────────────
declare -A ABLATION_GRIDS=(
    ["guidance_components"]="configs/ablations/main_guidance_components.yaml"
    ["loss_state"]="configs/ablations/main_loss_state.yaml"
    ["sampler_phase"]="configs/ablations/main_sampler_phase.yaml"
    ["guidance_schedule"]="configs/ablations/main_guidance_schedule.yaml"
    ["clipping"]="configs/ablations/main_clipping.yaml"
    ["pde_residual_region"]="configs/ablations/main_pde_residual_region.yaml"
    ["sensor_noise"]="configs/ablations/main_sensor_noise.yaml"
    ["steps_timegrid"]="configs/ablations/main_steps_timegrid.yaml"
    ["zeta_sensitivity"]="configs/ablations/main_zeta_sensitivity.yaml"
)

ALL_GROUPS=(
    guidance_components
    loss_state
    sampler_phase
    guidance_schedule
    clipping
    pde_residual_region
    sensor_noise
    steps_timegrid
    zeta_sensitivity
)

# ── base config ──────────────────────────────────────────────────────────────
# 直接使用 configs/ablations/base/poisson_both.yaml 作为基线
BASE_CFG="configs/ablations/base/poisson_both.yaml"
if [[ ! -f "${BASE_CFG}" ]]; then
    echo "Base config not found: ${BASE_CFG}" >&2
    exit 2
fi

# ── run one ablation group ────────────────────────────────────────────────────
run_ablation() {
    local name="$1"
    local grid_file="${ABLATION_GRIDS[${name}]}"
    echo ""
    echo "=== Ablation: ${name} (${grid_file}) ==="

    local modified="${OUTPUT}/grid_${name}.yaml"
    python3 -c "
import yaml
with open('${grid_file}') as f:
    grid = yaml.safe_load(f)
grid['base_config'] = '${BASE_CFG}'
with open('${modified}', 'w') as f:
    yaml.dump(grid, f)
"

    python -u -m sampling.sweep --grid "${modified}" 2>&1
    echo "--- ${name} done ---"
}

# ── main ──────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Available ablation groups:"
    for g in "${ALL_GROUPS[@]}"; do
        echo "  ${g}  →  ${ABLATION_GRIDS[${g}]}"
    done
    exit 0
fi

if [[ $# -eq 0 ]]; then
    GROUPS=("${ALL_GROUPS[@]}")
else
    GROUPS=("$@")
fi

for name in "${GROUPS[@]}"; do
    if [[ -z "${ABLATION_GRIDS[${name}]:-}" ]]; then
        echo "Unknown ablation group: ${name}" >&2
        echo "Run --help to see available groups." >&2
        exit 2
    fi
    run_ablation "${name}"
done

# ── aggregate ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Aggregating ablation results ==="
python -u -m sampling.aggregate "${OUTPUT}" --output-dir "${OUTPUT}" 2>&1

echo ""
echo "=== Ablations complete ==="
echo "Results: ${OUTPUT}/summary_all_raw.csv"
