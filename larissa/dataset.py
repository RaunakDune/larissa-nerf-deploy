"""Loader for the exported Larissa-Synth dataset (see tools/export_dataset.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .scene import Camera


def load_rgb16(path) -> torch.Tensor:
    """16-bit RGB PNG -> float tensor (H,W,3) in [0,1]."""
    import png
    w, h, rows, _ = png.Reader(filename=str(path)).asDirect()
    a = np.vstack([np.asarray(r, np.uint16) for r in rows]).reshape(h, w, 3)
    return torch.from_numpy(a.astype(np.float32) / 65535)


def camera_from_json(c: dict) -> Camera:
    return Camera(torch.tensor(c["R_world_to_cam"], dtype=torch.float32),
                  torch.tensor(c["t_world_to_cam"], dtype=torch.float32),
                  c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"])


class LarissaSynth:
    """Access to the dataset. Images load lazily from disk."""

    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "dataset.json").read_text())
        self.components = {int(k): v for k, v in self.meta["components"].items()}
        self.scenarios = list(self.meta["inspection"])

    # training (healthy baseline capture)
    def train_cameras(self):
        return [camera_from_json(e["camera"]) for e in self.meta["train"]]

    def train_view(self, i: int) -> dict:
        d = self.root / "train"
        return {"rgb": load_rgb16(d / f"rgb_{i:03d}.png"),
                "depth": torch.from_numpy(np.asarray(Image.open(d / f"depth_{i:03d}.png"), np.float32) / 1000),
                "comp": torch.from_numpy(np.asarray(Image.open(d / f"comp_{i:03d}.png"), np.int64))}

    # held-out healthy views (used for calibration); nominal = believed pose
    def holdout(self):
        d = self.root / "holdout"
        cams = [camera_from_json(e["camera_nominal"]) for e in self.meta["holdout"]]
        imgs = [load_rgb16(d / f"rgb_{e['id']:03d}.png") for e in self.meta["holdout"]]
        return imgs, cams

    # inspection passes
    def inspection(self, scenario: str):
        info = self.meta["inspection"][scenario]
        d = self.root / "inspection" / scenario
        cams = [camera_from_json(e["camera_nominal"]) for e in info["views"]]
        imgs = [load_rgb16(d / f"rgb_{e['id']}.png") for e in info["views"]]
        gts = [torch.from_numpy(np.asarray(Image.open(d / f"gt_{e['id']}.png")) > 127) for e in info["views"]]
        return imgs, cams, gts
