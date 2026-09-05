import os
import subprocess

from scripts.tuning.run_rough_inverse_refine import (
    PDES,
    apply_weighted_scores,
    candidates_for,
    parse_distribution_weights,
)


def test_rough_refine_grids_are_inverse_only_and_deduplicated():
    for profile in ("quick", "standard"):
        for pde in PDES:
            candidates = candidates_for(pde, profile=profile)
            signatures = {
                tuple(row[field] for field in ("zeta_obs_a", "zeta_obs_u", "zeta_pde", "clip_threshold"))
                for row in candidates
            }
            assert candidates[0]["candidate"] == "baseline"
            assert len(signatures) == len(candidates)
            assert all(row["zeta_obs_a"] == 0.0 for row in candidates)
    assert len(candidates_for("poisson", profile="quick")) == 6
    assert len(candidates_for("poisson", profile="standard")) == 9
    assert len(candidates_for("darcy", profile="standard")) == 27


def test_rough_weighting_prefers_the_candidate_that_is_better_on_rough():
    def summary(name: str, id_ratio: float, rough_ratio: float):
        row = {"candidate": name, "selection_score": 1.0}
        for test_type, ratio in (("id", id_ratio), ("smooth", id_ratio), ("rough", rough_ratio)):
            row.update(
                {
                    f"hard_primary_mean_ratio_{test_type}": ratio,
                    f"hard_primary_p90_ratio_{test_type}": ratio,
                    f"good_primary_mean_ratio_{test_type}": ratio,
                    f"random_primary_mean_ratio_{test_type}": ratio,
                    f"pde_residual_ratio_{test_type}": 1.0,
                }
            )
        return row

    summaries = [summary("id_favored", 0.7, 1.0), summary("rough_favored", 1.0, 0.6)]
    weighted = apply_weighted_scores(
        summaries,
        parse_distribution_weights("id=1,smooth=1,rough=2"),
        residual_penalty=0.1,
    )
    by_name = {row["candidate"]: row for row in weighted}
    assert by_name["rough_favored"]["selection_score"] < by_name["id_favored"]["selection_score"]


def test_rough_refine_wrapper_plan_does_not_require_gpu_or_artifacts(tmp_path):
    result = subprocess.run(
        ["bash", "scripts/tuning/run_rough_inverse_refine_a100.sh"],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PDE_LIST": "poisson",
            "PLAN_ONLY": "true",
            "SOURCE_ROOT": str(tmp_path / "missing-source-is-fine-for-plan"),
            "ARTIFACT_ROOT": str(tmp_path / "unused"),
        },
    )
    assert "PLAN pde=poisson task=inverse candidates=6 tune_jobs=36" in result.stdout
    assert "distribution_weights={'id': 1.0, 'smooth': 1.0, 'rough': 2.0}" in result.stdout
    assert not (tmp_path / "unused").exists()


def test_rough_refine_wrapper_can_plan_a_baseline_only_step_screen(tmp_path):
    result = subprocess.run(
        ["bash", "scripts/tuning/run_rough_inverse_refine_a100.sh"],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PDE_LIST": "darcy",
            "PLAN_ONLY": "true",
            "BASELINE_ONLY": "true",
            "NUM_STEPS": "1000",
            "SOURCE_ROOT": str(tmp_path / "missing-source-is-fine-for-plan"),
            "ARTIFACT_ROOT": str(tmp_path / "unused"),
        },
    )
    assert "candidates=1 tune_jobs=6" in result.stdout
    assert "steps=1000" in result.stdout
