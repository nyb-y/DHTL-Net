#!/usr/bin/env python3
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

from train_DAF import AdaptiveInterpDynamic


warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "picture1"
OUTPUT_DIR = ROOT / "picture1_output"
CHECKPOINT_PATH = ROOT / "runs_binary_iqa" / "DAF_seed0" / "best_model.pth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = ["unusable", "usable"]


def build_model(device):
    model = AdaptiveInterpDynamic(
        num_classes=2,
        alpha_priors=[0.85, 0.75, 0.55, 0.30],
        channels=[256, 512, 1024, 2048],
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = {
        key: value
        for key, value in ckpt["model_state_dict"].items()
        if not (key.endswith(".total_ops") or key.endswith(".total_params") or key in {"total_ops", "total_params"})
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_font(size):
    for font_name in ("arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate_image(image, pred_class, pred_prob):
    image = image.convert("RGB")
    font = load_font(max(18, image.width // 26))
    text = f"Pred: {pred_class} | P({pred_class})={pred_prob:.3f}"

    draw_tmp = ImageDraw.Draw(image)
    bbox = draw_tmp.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = max(10, image.width // 45)
    pad_y = max(8, image.height // 70)
    band_h = text_h + pad_y * 2

    out = Image.new("RGB", (image.width, image.height + band_h), "black")
    out.paste(image, (0, band_h))
    draw = ImageDraw.Draw(out)
    x = min(pad_x, max(pad_x, image.width - text_w - pad_x))
    draw.text((x, pad_y), text, fill="white", font=font)
    return out


@torch.no_grad()
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_paths = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for image_path in image_paths:
        original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        x = transform(original).unsqueeze(0).to(device)
        prob = torch.softmax(model(x), dim=1)[0].detach().cpu()
        pred_idx = int(prob.argmax().item())
        pred_class = CLASSES[pred_idx]
        pred_prob = float(prob[pred_idx].item())

        annotated = annotate_image(original, pred_class, pred_prob)
        out_name = (
            f"{image_path.stem}_pred-{pred_class}_"
            f"punusable{float(prob[0]):.3f}_pusable{float(prob[1]):.3f}.jpg"
        )
        annotated.save(OUTPUT_DIR / out_name, quality=95)


if __name__ == "__main__":
    main()
