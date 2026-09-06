from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
import torch

from sampling.config import load_config
from sampling.legacy_guidance import legacy_residual
from sampling.losses import compute_guidance_losses
from sampling.masks import PairMasks
from sampling.state import SplitState
from scripts.compare_bak import (
    cell_key, load_archived_batch, make_manifest, read_results, result_key,
    select_groups, summarize, worker_for_entry,
)


@pytest.mark.parametrize("mode", ["random", "sensor_column"])
def test_burgers_historical_stencil_and_loss_denominators(mode):
    # For a constant field, only zero-padding edge artifacts remain in the old
    # stencil. The physical periodic residual is zero; guidance must stay separate.
    pred = torch.ones(2, 1, 8, 8, dtype=torch.float64, requires_grad=True)
    expected = torch.zeros_like(pred)
    expected[..., 0, :] += 0.5
    expected[..., -1, :] -= 0.5
    expected[..., :, 0] += 0.5
    expected[..., :, -1] -= 0.5
    expected[..., :, 1] += 0.0025
    expected[..., :, -2] += 0.0025
    torch.testing.assert_close(legacy_residual("burger", pred, pred), expected)
    cfg = load_config("configs/bak/both/burger.yaml")
    cfg.sensor_mode = mode
    cfg.num_sensor_columns = 2
    gt = SimpleNamespace(coef=torch.zeros_like(pred), sol=torch.zeros_like(pred), pde_params={})
    mask = torch.zeros_like(pred)
    mask[..., :2] = 1
    out = compute_guidance_losses(SplitState(pred, pred), gt, PairMasks(mask, mask, {}), cfg)
    assert out.guidance_L_obs_u.item() == pytest.approx(4 / 500)
    assert out.L_obs_u.item() == pytest.approx(1.0)
    denominator = 16 if mode == "sensor_column" else 64
    assert out.guidance_L_pde.item() == pytest.approx(expected[0].norm().item() / denominator)
    (out.guidance_L_obs_u + out.guidance_L_pde).backward()
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad[0], pred.grad[1])


def write_baseline_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_burgers_selects_six_complete_cells_and_uses_both_workers(tmp_path):
    for distribution in ("id", "smooth", "rough"):
        rows = [dict(pde="burger", task="both", sample_id=i,
                     sensor_mode=mode, num_sensor_columns=5, ablation_group_key=mode,
                     num_obs=500, num_steps=100, sampler_phase="stochastic")
                for mode in ("random", "sensor_column") for i in range(1000)]
        write_baseline_csv(tmp_path / f"MAIN1000_100_TEST_{distribution}/metrics_per_sample_all.csv", rows)
        # Partial tuned1 runs must not replace complete original baseline cells.
        write_baseline_csv(tmp_path / f"MAIN1000_100_TEST_{distribution}_tuned1/metrics_per_sample_all.csv", rows[:3])
    manifest = make_manifest(tmp_path, 2, 3, 20260905, "burgers")
    assert len(manifest) == 14
    assert len({result_key(row) for row in manifest}) == 14
    assert len({cell_key(row) for row in manifest}) == 6
    assert all(row["test_type"] == "rough" for row in manifest[:6])
    for worker in (0, 1):
        assigned = [row for row in manifest if worker_for_entry(row, 2) == worker]
        assert len(assigned) == 7
        assert len({row["sensor_mode"] for row in assigned}) == 1
    # Two complete groups with different column budgets require an explicit choice.
    rows += [dict(row, num_sensor_columns=16, ablation_group_key="column16")
             for row in rows if row["sensor_mode"] == "sensor_column"]
    write_baseline_csv(tmp_path / "MAIN1000_100_TEST_rough/metrics_per_sample_all.csv", rows)
    with pytest.raises(ValueError, match="Ambiguous"):
        select_groups(tmp_path, "burgers")


@pytest.mark.parametrize("mode", ["random", "sensor_column"])
def test_burgers_archive_replay_has_one_channel_and_validates_masks(tmp_path, mode):
    field = torch.ones(3, 1, 128, 128)
    mask = torch.zeros_like(field)
    if mode == "random":
        mask.flatten(1)[:, :500] = 1
    else:
        mask[..., :5] = 1
    payload = dict(
        config=dict(pde="burger", task="both", num_obs=500, num_steps=100, sampler_phase="stochastic",
                    sensor_mode=mode, num_sensor_columns=5, checkpoint_path="formal/burger/260625-150439/fm4burger.pth"),
        coef_ground_truth=field, sol_ground_truth=field,
        coef_final=field * 0.9, sol_final=field * 0.9,
        ground_truth_metadata=dict(sample_ids=["40", "41", "42"], channel_names_coef=["u"], channel_names_sol=["u"]),
        masks=dict(coef=mask, sol=mask, metadata={}),
        pde_params={"nu": torch.tensor([0.01, 0.02, 0.03])},
    )
    torch.save(payload, tmp_path / "result.pt")
    entries = [dict(pde="burger", task="both", test_type="rough", sample_id=40+i,
                    sensor_mode=mode, num_sensor_columns=5 if mode == "sensor_column" else None,
                    baseline=dict(run_dir=str(tmp_path), sample_index=str(i), rel_l2_a="0.1", rel_l2_u="0.1"))
               for i in (2, 0)]
    gt, masks, _, indices, batch_size = load_archived_batch(entries, "cpu")
    assert gt.pair.shape == (2, 1, 128, 128)
    assert indices == [2, 0] and batch_size == 3
    assert gt.metadata["sample_ids"] == ["42", "40"]
    torch.testing.assert_close(gt.pde_params["nu"], torch.tensor([0.03, 0.01]))
    assert masks.sol[0].sum().item() == (640 if mode == "sensor_column" else 500)
    # A corrupted budget must stop comparison rather than silently change guidance.
    payload["masks"]["sol"] = mask.clone()
    payload["masks"]["sol"][..., 0, 0] = 0
    torch.save(payload, tmp_path / "result.pt")
    with pytest.raises(ValueError, match="observation budget"):
        load_archived_batch(entries, "cpu")


def test_burgers_resume_and_statistics_keep_sensor_layouts_separate(tmp_path):
    rows = [dict(pde="burger", task="both", test_type="rough", sample_id=i,
                 sensor_mode=mode, num_sensor_columns=5 if mode == "sensor_column" else None,
                 current_a=0.5, current_u=0.5, bak_a=0.4, bak_u=0.4, status="ok")
            for mode in ("random", "sensor_column") for i in range(8)]
    failure = dict(rows[0], status="failed", error="temporary failure")
    (tmp_path / "paired_results.jsonl").write_text(json.dumps(failure) + "\n")
    (tmp_path / "paired_results.worker0.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    assert len(read_results(tmp_path)) == 16
    summaries = summarize(tmp_path)
    assert len(summaries) == 2
    assert all(row["n"] == 8 and row["failures"] == 0 for row in summaries)
    assert {row["metric"] for row in summaries} == {"rel_l2_u"}
    assert all("p_holm_6" in row and "p_holm_72" not in row for row in summaries)
    assert all(row["mean_difference"] == pytest.approx(-0.1) for row in summaries)
