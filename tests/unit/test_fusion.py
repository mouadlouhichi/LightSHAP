from __future__ import annotations

import pytest
import torch

from lightshap.scoring.fusion import game_a_slots, mask_and_mean


@pytest.mark.unit
def test_game_a_always_keeps_e0() -> None:
    assert game_a_slots(()) == (0,)
    assert game_a_slots((2, 3)) == (0, 2, 3)
    assert game_a_slots((1, 2, 3)) == (0, 1, 2, 3)


@pytest.mark.unit
def test_mask_and_mean() -> None:
    layers = [torch.ones(3, 2) * k for k in range(4)]
    fused = mask_and_mean(layers, (0, 2))
    assert torch.allclose(fused, torch.ones(3, 2))  # mean(0, 2) = 1
