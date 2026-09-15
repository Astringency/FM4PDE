import copy
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ns44_replay", ROOT / "revision_ns44m_0915/run_ablations.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_preserves_batch_seed_and_non_model_scientific_settings(tmp_path):
    from sampling.config import AblationConfig
    original = AblationConfig(pde="nsnonbounded", task="both", batch_size=4,
        sample_seed=20261332, mask_seed=20261332, clip_threshold=50.0,
        zeta_obs_a=60000., zeta_obs_u=60000., zeta_pde=.1,
        pde_guidance_start_ratio=.8, residual_mode="endpoint_secant",
        model_profile="recommended").asdict()
    job = {"config": original, "sample_ids": [425, 215, 830, 756]}
    cfg = replay.effective_config(job, tmp_path / "model.pth", tmp_path / "out", "cpu")
    assert cfg.model_profile == "light"
    assert replay.scientific_config(cfg.asdict()) == replay.scientific_config(original)
    assert job["config"] == original
    wrong = copy.deepcopy(job)
    wrong["sample_ids"] = [425]
    with pytest.raises(AssertionError):
        replay.effective_config(wrong, tmp_path / "model.pth", tmp_path / "out", "cpu")


def test_saved_input_checks_reject_changed_conditioning():
    a, u = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)
    masks = {"coef": torch.zeros_like(a), "sol": torch.ones_like(u)}
    packet = {"inputs": {"coef_ground_truth": a, "sol_ground_truth": u,
                          "masks": masks, "pde_params": {"T": torch.tensor([1.])}}}
    payload = copy.deepcopy(packet["inputs"])
    payload.update(coef_final=a.clone(), sol_final=u.clone())
    replay.validate_payload(packet, payload)
    payload["masks"]["coef"][0, 0, 0, 0] = 1
    with pytest.raises(AssertionError):
        replay.validate_payload(packet, payload)
    payload = copy.deepcopy(packet["inputs"])
    payload.update(coef_final=a.clone(), sol_final=u.clone())
    payload["pde_params"]["T"] = torch.tensor([2.])
    with pytest.raises(AssertionError):
        replay.validate_payload(packet, payload)


def test_saved_temporal_auxiliary_is_reused_without_raw_data(tmp_path):
    from sampling.config import AblationConfig
    from sampling.data import PDEGroundTruth
    from sampling.masks import PairMasks
    import sampling.runner as runner
    a = torch.zeros(1, 1, 8, 8)
    aux = {"near_endpoint_temporal": {"q_dt_obs": a.clone(), "q_T_minus_dt_obs": a.clone(),
           "mask_0": torch.ones_like(a), "mask_T": torch.ones_like(a), "dt": torch.tensor([.1])}}
    masks = PairMasks(torch.ones_like(a), torch.ones_like(a), {})
    packet = {"inputs": {"pde_params": copy.deepcopy(aux), "masks": {"coef": masks.coef, "sol": masks.sol}}}
    gt = PDEGroundTruth("nsnonbounded", a, a, torch.cat([a, a], 1), aux, ["w0"], ["wT"], {})
    cfg = AblationConfig(pde="nsnonbounded", residual_mode="near_endpoint_temporal", data_path="/absent/source.mat")
    before = runner.attach_near_endpoint_observations
    with replay.record_actual_inputs(packet, {}, tmp_path) as observed:
        assert runner.attach_near_endpoint_observations(cfg, gt, masks) is gt
        assert observed["temporal_auxiliary_source"] == "historical_saved_sparse_observations"
        from sampling.losses import prediction_only_pde_params
        params, _ = prediction_only_pde_params(gt.pde_params, allow_sparse_near_endpoint=True)
        assert torch.equal(params["near_endpoint_temporal"]["q_dt"], packet["inputs"]["pde_params"]["near_endpoint_temporal"]["q_dt_obs"])
    assert runner.attach_near_endpoint_observations is before


def test_weight_selection_preserves_field_guard_and_no_selection_case():
    from revision_ns44m_0915.run_weight_selection import choose
    dev = [dict(multiplier=0, mean_pde=1., mean_errors={"a":1., "u":1.}),
           dict(multiplier=1, mean_pde=.1, mean_errors={"a":1., "u":1.}),
           dict(multiplier=10, mean_pde=.05, mean_errors={"a":1.03, "u":1.}),
           dict(multiplier=100, mean_pde=.5, mean_errors={"a":1.02, "u":1.01})]
    assert choose(dev)["selected_multiplier"] == 100
    dev[-1]["mean_pde"] = 1.0
    assert choose(dev)["selected_multiplier"] is None
