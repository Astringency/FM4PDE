from pathlib import Path


def test_fourier_features_value_vs_coordinate_documented():
    text = Path("docs/model_architectures.md").read_text(encoding="utf-8")

    assert "value Fourier" in text
    assert "coordinate Fourier" in text
    assert "input field values" in text
    assert "grid coordinates" in text
    assert "official Flow Matching image example" in text
