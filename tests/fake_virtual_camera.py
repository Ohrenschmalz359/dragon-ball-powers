"""Run the app with a fake virtual camera driver (used by tests/test_virtual_camera.py).

Usage: ``python tests/fake_virtual_camera.py [app arguments]``. Every real pyvirtualcam
backend is replaced by the fake, so no real device (e.g. an installed OBS Virtual Camera)
is touched. What the fake receives is written to ``$DBP_FAKE_VCAM_DIR``: ``stats.json``
and the most detailed picture (highest standard deviation) as ``best.npy``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyvirtualcam
from pyvirtualcam import camera as pyvirtualcam_camera

OUT = Path(os.environ.get("DBP_FAKE_VCAM_DIR", "."))
STATS: dict = {"opened": 0, "closed": 0, "frames": 0, "size": None, "fps": None, "device": None, "best_std": -1.0}


class FakeBackend:
    """Implements pyvirtualcam's backend interface; frames arrive as 1-D RGB arrays."""

    def __init__(self, *, width: int, height: int, fps: float, fourcc: int, device: str | None, **kw) -> None:
        STATS["opened"] += 1
        STATS.update(size=[width, height], fps=fps, device=device)
        self._shape = (height, width, 3)
        OUT.mkdir(parents=True, exist_ok=True)

    def send(self, frame: np.ndarray) -> None:
        STATS["frames"] += 1
        img = frame.reshape(self._shape)
        std = float(img.std())
        if std > STATS["best_std"] + 1.0:
            STATS["best_std"] = std
            np.save(OUT / "best.npy", img)

    def close(self) -> None:
        STATS["closed"] += 1
        (OUT / "stats.json").write_text(json.dumps(STATS), encoding="utf-8")

    def device(self) -> str:
        return "Fake Virtual Camera"

    def native_fourcc(self) -> None:
        return None


def main() -> int:
    pyvirtualcam_camera.BACKENDS.clear()
    pyvirtualcam.register_backend("fake", FakeBackend)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from dragon_ball_powers.main import main as app_main

    return app_main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
