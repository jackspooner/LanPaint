import importlib
import sys
import types

import pytest
import torch


def _repeat_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))
    repeats = (batch_size + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))[:batch_size]


def _import_nodes(monkeypatch, comfyui_version: str):
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.__path__ = []

    comfy_utils_mod = types.ModuleType("comfy.utils")
    comfy_utils_mod.repeat_to_batch_size = _repeat_to_batch_size

    comfy_samplers_mod = types.ModuleType("comfy.samplers")
    class DummyKSAMPLER: ...
    comfy_samplers_mod.KSAMPLER = DummyKSAMPLER

    comfy_model_base_mod = types.ModuleType("comfy.model_base")
    class ModelType:
        FLUX = "FLUX"
        FLOW = "FLOW"

    class WAN22: ...
    comfy_model_base_mod.ModelType = ModelType
    comfy_model_base_mod.WAN22 = WAN22

    comfyui_version_mod = types.ModuleType("comfyui_version")
    comfyui_version_mod.__version__ = comfyui_version

    comfy_mod.utils = comfy_utils_mod
    comfy_mod.samplers = comfy_samplers_mod
    comfy_mod.model_base = comfy_model_base_mod

    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_mod)
    monkeypatch.setitem(sys.modules, "comfy.samplers", comfy_samplers_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_base", comfy_model_base_mod)
    monkeypatch.setitem(sys.modules, "nodes", types.ModuleType("nodes"))
    monkeypatch.setitem(sys.modules, "latent_preview", types.ModuleType("latent_preview"))
    monkeypatch.setitem(sys.modules, "comfyui_version", comfyui_version_mod)

    sys.modules.pop("src.LanPaint.nodes", None)
    return importlib.import_module("src.LanPaint.nodes")


@pytest.mark.parametrize("comfyui_version", ["0.5.0", "0.6.0"])
def test_reshape_mask_accepts_bhw_and_5d_output_shape(monkeypatch, comfyui_version: str) -> None:
    lanpaint_nodes = _import_nodes(monkeypatch, comfyui_version)
    input_mask = torch.zeros((1, 4, 4))
    output_shape = (1, 16, 1, 8, 8)

    out = lanpaint_nodes.reshape_mask(input_mask, output_shape, video_inpainting=False)
    assert tuple(out.shape) == output_shape


def test_prepare_mask_accepts_hw_and_moves_device(monkeypatch) -> None:
    lanpaint_nodes = _import_nodes(monkeypatch, "0.5.0")
    input_mask = torch.zeros((4, 4))
    output_shape = (2, 3, 8, 8)

    out = lanpaint_nodes.prepare_mask(input_mask, output_shape, device=torch.device("cpu"), video_inpainting=False)
    assert tuple(out.shape) == output_shape
    assert out.device.type == "cpu"


def test_video_union_covers_picked_frames(monkeypatch) -> None:
    """Coverage pooling retains strokes and temporal dilation joins nearby bins."""
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")

    # Strokes in separate temporal bins and separate spatial cells.
    mask = torch.zeros(8, 8, 8)
    mask[2, 5, 5] = 1.0
    mask[6, 7, 7] = 1.0

    out = lanpaint_nodes.reshape_mask(mask, (1, 16, 2, 4, 4), video_inpainting=True)
    assert tuple(out.shape) == (1, 16, 2, 4, 4)
    # both slices are marked (slice-level union spreads the picked values)
    assert out[0, 0, 0].max().item() == 1.0
    assert out[0, 0, 1].max().item() == 1.0
    # the stroke positions land on the token containing the stroke pixel
    assert out[0, 0, 0, 2, 2].item() == 1.0
    assert out[0, 0, 1, 3, 3].item() == 1.0


def test_video_union_preserves_unpicked_frames(monkeypatch) -> None:
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")

    mask = torch.zeros(8, 8, 8)
    mask[3, 5, 5] = 1.0

    out = lanpaint_nodes.reshape_mask(mask, (1, 16, 2, 4, 4), video_inpainting=True)
    assert out.max().item() == 1.0


@pytest.mark.parametrize("frame", range(22))
def test_every_h3_source_frame_and_narrow_pixel_survives(monkeypatch, frame):
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")
    mask = torch.zeros(22, 8, 8)
    mask[frame, 0, 0] = 0.375
    out = lanpaint_nodes.reshape_mask(mask, (1, 24, 7, 2, 2), video_inpainting=True)
    assert out.max().item() == 0.375
    assert torch.count_nonzero(out[0, 0, :, 0, 0]) <= 7
    assert torch.count_nonzero(out[0, 0, :, 1, 1]) == 0


@pytest.mark.parametrize("frames,slices", [(1, 1), (3, 1), (5, 1), (22, 7), (39, 13)])
def test_video_coverage_zero_dense_soft_and_padded(monkeypatch, frames, slices):
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")
    shape = (2, 24, slices, 2, 2)
    for value in (0.0, 0.25, 1.0):
        out = lanpaint_nodes.reshape_mask(torch.full((frames, 8, 8), value), shape, video_inpainting=True)
        torch.testing.assert_close(out, torch.full(shape, value))
    mask = torch.zeros(frames, 8, 8)
    mask[0] = 1
    out = lanpaint_nodes.reshape_mask(mask, shape, video_inpainting=True)
    assert out[:, :, 0].min() == 1
    if slices > 3:
        assert out[:, :, 3:].max() == 0


def test_video_union_short_sequence(monkeypatch) -> None:
    """A short sequence's stroke survives projection to one slice."""
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")

    mask = torch.zeros(3, 8, 8)
    mask[1, 5, 5] = 1.0

    out = lanpaint_nodes.reshape_mask(mask, (1, 16, 1, 4, 4), video_inpainting=True)
    assert tuple(out.shape) == (1, 16, 1, 4, 4)
    assert out.max().item() == 1.0
