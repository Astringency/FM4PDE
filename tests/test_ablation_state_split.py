import pytest

torch = pytest.importorskip("torch")

from sampling.state import compose_pair_state, split_pair_state


def test_split_even_pair():
    x = torch.zeros(2, 4, 8, 8)
    split = split_pair_state(x, "reaction_diffusion")
    assert split.coef.shape[1] == 2
    assert split.sol.shape[1] == 2


def test_split_burger_single_channel():
    x = torch.zeros(2, 1, 8, 8)
    split = split_pair_state(x, "burger")
    assert split.coef.shape == split.sol.shape


def test_compose_pair_state():
    a = torch.zeros(1, 1, 4, 4)
    u = torch.ones(1, 1, 4, 4)
    pair = compose_pair_state(a, u)
    assert pair.shape == (1, 2, 4, 4)
