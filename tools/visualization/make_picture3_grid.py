#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from recommend import METHOD_SPECS


ROOT = Path(__file__).resolve().parent
PICTURE3_DIR = ROOT / "picture3"
MANIFEST_PATH = PICTURE3_DIR / "manifest.csv"
OUTPUT_PATH = PICTURE3_DIR / "picture3_grid_12x4.png"


def load_font(size: int):
    for font_name in ("arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    manifest_df = pd.read_csv(MANIFEST_PATH)
    method_order = [spec["method_id"] for spec in METHOD_SPECS]
    image_order = sorted(manifest_df["image_index"].unique().tolist())

    if len(image_order) != 4:
        raise RuntimeError(f"Expected 4 image columns, got {len(image_order)}")

    tile_map = {}
    for _, row in manifest_df.iterrows():
        method_id = str(row["method_id"])
        export_path = Path(str(row["export_path"]))
        local_path = PICTURE3_DIR / method_id / export_path.name
        tile_map[(method_id, int(row["image_index"]))] = local_path

    sample_images = []
    for key in [(method_order[0], image_order[0])]:
        path = tile_map.get(key)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing tile image: {key} -> {path}")
        sample_images.append(Image.open(path).convert("RGB"))

    tile_w = max(img.width for img in sample_images)
    tile_h = max(img.height for img in sample_images)

    cols = len(image_order)
    rows = len(method_order)
    header_h = 64
    side_w = 220
    pad = 8

    canvas_w = side_w + cols * tile_w + (cols + 1) * pad
    canvas_h = header_h + rows * tile_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    header_font = load_font(28)
    method_font = load_font(24)
    small_font = load_font(20)

    draw.rectangle((0, 0, canvas_w, header_h), fill=(24, 24, 24))
    draw.text((18, 16), "Method", fill="white", font=header_font)

    for col_idx, image_index in enumerate(image_order):
        x = side_w + pad + col_idx * tile_w + col_idx * pad
        draw.text((x + 12, 16), f"Image {image_index}", fill="white", font=header_font)

    for row_idx, method_id in enumerate(method_order):
        y = header_h + pad + row_idx * tile_h + row_idx * pad
        method_label = next(spec["method_label"] for spec in METHOD_SPECS if spec["method_id"] == method_id)
        draw.rectangle((0, y, side_w, y + tile_h), fill=(245, 245, 245))
        draw.text((16, y + 18), method_label, fill="black", font=method_font)
        draw.text((16, y + 54), method_id, fill=(90, 90, 90), font=small_font)

        for col_idx, image_index in enumerate(image_order):
            x = side_w + pad + col_idx * tile_w + col_idx * pad
            tile_path = tile_map.get((method_id, image_index))
            if tile_path is None or not tile_path.is_file():
                raise FileNotFoundError(f"Missing tile for method={method_id}, image_index={image_index}")

            tile = Image.open(tile_path).convert("RGB")
            if tile.size != (tile_w, tile_h):
                tile = tile.resize((tile_w, tile_h))
            canvas.paste(tile, (x, y))

    canvas.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
