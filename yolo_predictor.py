import os
import torch
from models import YoloRegressorModel, load_model
from torchsummary import summary
from pprint import pprint
from PIL import Image
import numpy as np
import math
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw
import torch

# ---- Config / Device ----
CKPT_PATH = "predictor.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Preprocess ----------

def load_image_for_model(
    image_path: str | Path,
    size: int = 448,
    letterbox: bool = False,
) -> tuple[np.ndarray, torch.Tensor, Dict]:
    """
    Returns:
      img_np: (H, W, 3) uint8 RGB of the original image
      x: torch.FloatTensor of shape (1, 3, size, size) with values in [0,1]
      meta: dict with mapping info to convert predictions back to original image
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    if not letterbox:
        # Simple warp-resize (use this if you trained with plain Resize((448,448)))
        img_resized = img.resize((size, size), Image.BILINEAR)
        pad_left = pad_top = 0
        scale = min(size / orig_w, size / orig_h)  # not used for warp, but keep shape
    else:
        # YOLO-style letterbox (keeps aspect ratio; adds padding)
        scale = min(size / orig_w, size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (size, size), (114, 114, 114))  # neutral pad
        pad_left = (size - new_w) // 2
        pad_top  = (size - new_h) // 2
        canvas.paste(img_resized, (pad_left, pad_top))
        img_resized = canvas

    # To NumPy (H, W, 3) uint8 and normalized float tensor (1,3,H,W)
    img_np = np.array(img)  # original image as uint8
    arr = np.array(img_resized).astype(np.float32) / 255.0  # [0,1], HWC
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)

    meta = dict(
        input_size=size,
        orig_w=orig_w,
        orig_h=orig_h,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        letterbox=letterbox,
    )
    return img_np, x, meta

# ---------- Inference ----------

@torch.inference_mode()
def predict_bbox(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    x: (1,3,448,448) in [0,1]
    Returns: np.array([xc, yc, w, h]) normalized to [0,1] w.r.t. the model input
    """
    model.eval().to(device)
    x = x.to(device, non_blocking=True)
    preds = model(x).detach().float().cpu().numpy().reshape(-1)
    # Clamp just in case
    preds = np.clip(preds, 0.0, 1.0)
    return preds  # [xc, yc, w, h] in [0,1]

# ---------- Mapping back to original image ----------

def box_to_original_xyxy(
    pred_norm: np.ndarray,
    meta: Dict,
) -> Tuple[int, int, int, int]:
    """
    pred_norm: [xc,yc,w,h] normalized [0,1] at (size,size) net input
    Returns: (x1,y1,x2,y2) in original image pixel coords (int)
    """
    size = meta["input_size"]
    xc_s = pred_norm[0] * size
    yc_s = pred_norm[1] * size
    w_s  = pred_norm[2] * size
    h_s  = pred_norm[3] * size

    if not meta["letterbox"]:
        # Inverse of plain warp resize
        sx = meta["orig_w"] / size
        sy = meta["orig_h"] / size
        xc = xc_s * sx
        yc = yc_s * sy
        w  =  w_s * sx
        h  =  h_s * sy
    else:
        # Remove padding then unscale
        xc = (xc_s - meta["pad_left"]) / meta["scale"]
        yc = (yc_s - meta["pad_top"])  / meta["scale"]
        w  =  w_s / meta["scale"]
        h  =  h_s / meta["scale"]

    # Convert center-size -> corners
    x1 = int(round(xc - w / 2))
    y1 = int(round(yc - h / 2))
    x2 = int(round(xc + w / 2))
    y2 = int(round(yc + h / 2))

    # Clamp to original image bounds
    x1 = max(0, min(meta["orig_w"] - 1, x1))
    y1 = max(0, min(meta["orig_h"] - 1, y1))
    x2 = max(0, min(meta["orig_w"] - 1, x2))
    y2 = max(0, min(meta["orig_h"] - 1, y2))
    return x1, y1, x2, y2

# ---------- (Optional) Draw/save box ----------

def draw_bbox_on_image(img_np: np.ndarray, xyxy: Tuple[int,int,int,int], color=(255, 0, 0), width=3) -> Image.Image:
    """
    Draws a rectangle on a copy of the original image and returns a PIL.Image
    """
    pil_img = Image.fromarray(img_np.copy())
    d = ImageDraw.Draw(pil_img)
    x1, y1, x2, y2 = xyxy
    for i in range(width):
        d.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    return pil_img

# ---------- Example usage ----------

if __name__ == "__main__":
    from models import YoloRegressorModel  # your class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Restore model (use your robust loader from earlier)
    model = YoloRegressorModel().to(device)
    ckpt = torch.load("predictor.pt", map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # Strip common prefixes if needed:
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # Read a non-training image and predict
    image = "cat1.png"
    # Set letterbox=True ONLY if you trained with letterbox padding
    img_np, x, meta = load_image_for_model(f"test_images/{image}", size=448, letterbox=False)

    pred_norm = predict_bbox(model, x, device)   # [xc,yc,w,h] in [0,1]
    xyxy = box_to_original_xyxy(pred_norm, meta) # (x1,y1,x2,y2) in original pixels

    print("Pred (normalized):", pred_norm)
    print("Box on original image (xyxy):", xyxy)

    # Draw & save
    out = draw_bbox_on_image(img_np, xyxy, color=(255, 0, 0), width=3)
    out.save(f"test_predictions/{image}")
    print(f"Saved: test_predictions/{image}")


