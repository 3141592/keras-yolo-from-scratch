# yolo_from_scratch/models/yolo_regressor.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

# ---------- Model ----------
def Conv2D(in_ch, out_ch, k, s=1):
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad),
        nn.LeakyReLU(0.1, inplace=True),
    )

class YoloRegressorModel(nn.Module):
    """
    Simple YOLO-style regressor that outputs [xc, yc, w, h] in normalized coords.
    Make sure this implementation matches what you had in your script.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 448,
        out_dims: int = 4,   # [xc, yc, w, h]
        hidden: int = 4096,
        p_drop: float = 0.5,
        leak: float = 0.1,
    ) -> None:
        super().__init__()

        pool = lambda: nn.MaxPool2d(kernel_size=2, stride=2)

        self.feat = nn.Sequential(
            # Layer 1
            Conv2D(3, 192, 7, s=2),
            pool(),
            # Layer 2
            Conv2D(192, 256, 3),
            pool(),
            # Layer 3
            Conv2D(256, 128, 1),
            Conv2D(128, 256, 3),
            Conv2D(256, 256, 1),
            Conv2D(256, 512, 3),
            pool(),
            # Layer 4
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 512, 1),
            Conv2D(512, 1024, 3),
            pool(),
            # Layer 5
            Conv2D(1024, 512, 1),
            Conv2D(512, 1024, 3),
            Conv2D(1024, 512, 1),
            Conv2D(512, 1024, 3),
            Conv2D(1024, 1024, 3),
            # stride-2 conv
            Conv2D(1024, 1024, 3, s=2),
            # Layer 6
            Conv2D(1024, 1024, 3),
            Conv2D(1024, 1024, 3),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 448, 448)
            flat = self.feat(dummy).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4)  # [xc,yc,w,h]
        )

    def _infer_flatten(self, input_shape: Tuple[int, int, int, int]) -> int:
        dummy = torch.zeros(*input_shape)
        out = self.features(dummy)
        return int(out.numel())

    def forward(self, x):
        x = self.feat(x)
        x = self.fc(x)
        return x

def load_model(
    weights_path: str | None = None,
    device: torch.device | str = "cpu",
    **kwargs,
) -> YoloRegressorModel:
    """
    Convenience factory: builds the model and (optionally) loads weights.

    Accepts the same kwargs as YoloRegressorModel(...).
    Handles either:
      - torch.save(model.state_dict())
      - torch.save({"model": model.state_dict(), ...})
    """
    model = YoloRegressorModel(**kwargs)
    model.to(device)

    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model", ckpt)  # support both formats
        model.load_state_dict(state, strict=True)

    return model

