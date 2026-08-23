import os
import subprocess


def test_single_sample_script_routes_task_to_nested_main_config(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf 'PYTHON_ARGS %s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PDE": "poisson",
            "TASK": "inverse",
            "DEVICE": "cpu",
            "VIS": "false",
            "PATH": f"{tmp_path}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/sample/run_sample.sh"],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert "config:   configs/main/inverse/poisson.yaml" in result.stdout
    assert "--config configs/main/inverse/poisson.yaml" in result.stdout


def test_single_sample_script_rejects_unsupported_burger_inverse():
    env = os.environ.copy()
    env.update({"PDE": "burger", "TASK": "inverse"})

    result = subprocess.run(
        ["bash", "scripts/sample/run_sample.sh"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 2
    assert "configs/main/inverse/burger.yaml" in result.stderr
