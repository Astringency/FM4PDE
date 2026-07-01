import pytest

torch = pytest.importorskip("torch")

from models.unet import (
    UNetModel,
    base2_fourier_features,
    coordinate_fourier_feature_channel_count,
    coordinate_fourier_features,
    fourier_feature_channel_count,
)


def _tiny_unet(channels: int, **overrides) -> UNetModel:
    cfg = {
        "in_channels": channels,
        "model_channels": 32,
        "out_channels": channels,
        "num_res_blocks": 1,
        "attention_resolutions": (),
        "dropout": 0.0,
        "channel_mult": (1,),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": False,
        "resblock_updown": False,
        "use_new_attention_order": False,
    }
    cfg.update(overrides)
    return UNetModel(**cfg)


def test_unet_forward_extra_default_is_not_shared():
    assert UNetModel.forward.__defaults__ == (None,)
    model = _tiny_unet(1, num_classes=3)

    x1 = torch.randn(1, 1, 8, 8)
    t1 = torch.tensor([0.25])
    out1 = model(x1, t1)

    x2 = torch.randn(2, 1, 8, 8)
    t2 = torch.tensor([0.25, 0.75])
    out2 = model(x2, t2)

    assert out1.shape == x1.shape
    assert out2.shape == x2.shape


@pytest.mark.parametrize("channels", [1, 2, 4])
def test_coordinate_fourier_feature_channel_count_matches_forward(channels):
    model = _tiny_unet(channels, with_coordinate_fourier_features=True)
    coordinate_extra = coordinate_fourier_feature_channel_count()
    expected = channels + coordinate_extra
    assert model.input_blocks[0][0].weight.shape[1] == expected

    x = torch.randn(1, channels, 8, 8)
    t = torch.tensor([0.5])
    out = model(x, t)

    assert out.shape == x.shape


def test_value_and_coordinate_fourier_can_coexist():
    channels = 2
    model = _tiny_unet(
        channels,
        with_value_fourier_features=True,
        with_coordinate_fourier_features=True,
    )
    value_extra = fourier_feature_channel_count(channels, start=6, stop=8, step=1)
    coord_extra = coordinate_fourier_feature_channel_count()
    expected = channels + value_extra + coord_extra
    assert model.input_blocks[0][0].weight.shape[1] == expected

    x = torch.randn(1, channels, 8, 8)
    t = torch.tensor([0.5])
    out = model(x, t)

    assert out.shape == x.shape


def test_coordinate_fourier_features_are_input_value_independent():
    x1 = torch.zeros(2, 3, 5, 7)
    x2 = torch.randn(2, 3, 5, 7)

    assert torch.allclose(coordinate_fourier_features(x1), coordinate_fourier_features(x2))


def test_value_fourier_features_are_input_value_dependent():
    x1 = torch.zeros(2, 3, 5, 7)
    x2 = torch.full_like(x1, 0.123)

    assert not torch.allclose(
        base2_fourier_features(x1, start=0, stop=2, step=1),
        base2_fourier_features(x2, start=0, stop=2, step=1),
    )


def test_concat_conditioning_without_config_raises_clear_error():
    model = _tiny_unet(1, with_value_fourier_features=True, with_coordinate_fourier_features=True)
    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([0.5])

    with pytest.raises(ValueError, match="concat_conditioning requires explicit concat_conditioning_channels"):
        model(x, t, extra={"concat_conditioning": torch.randn(1, 1, 8, 8)})


def test_unet_scalar_conditioning_forward_shape():
    model = _tiny_unet(2, scalar_conditioning=True, scalar_conditioning_dim=2)
    x = torch.randn(3, 2, 8, 8)
    t = torch.tensor([0.25, 0.5, 0.75])
    scalar = torch.randn(3, 2)

    out = model(x, t, extra={"scalar_conditioning": scalar})

    assert out.shape == x.shape


def test_unet_scalar_conditioning_requires_extra():
    model = _tiny_unet(1, scalar_conditioning=True, scalar_conditioning_dim=1)
    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([0.5])

    with pytest.raises(ValueError, match="requires extra\\['scalar_conditioning'\\]"):
        model(x, t)


def test_unet_scalar_conditioning_dim_mismatch_raises():
    model = _tiny_unet(1, scalar_conditioning=True, scalar_conditioning_dim=2)
    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([0.5])

    with pytest.raises(ValueError, match="feature dimension must match"):
        model(x, t, extra={"scalar_conditioning": torch.randn(1, 1)})
