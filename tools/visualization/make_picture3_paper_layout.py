#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from recommend import METHOD_SPECS


ROOT = Path(__file__).resolve().parent
PICTURE3_DIR = ROOT / "picture3"
MANIFEST_PATH = PICTURE3_DIR / "manifest.csv"
OUTPUT_PATH = PICTURE3_DIR / "picture3_paper_layout.png"


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
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    tile_map = {}
    for _, row in manifest_df.iterrows():
        method_id = str(row["method_id"])
        export_path = Path(str(row["export_path"]))
        local_path = PICTURE3_DIR / method_id / export_path.name
        tile_map[(method_id, int(row["image_index"]))] = local_path

    first_tile = tile_map[(method_order[0], image_order[0])]
    if not first_tile.is_file():
        raise FileNotFoundError(f"Missing tile image: {first_tile}")
    sample = Image.open(first_tile).convert("RGB")
    tile_w, tile_h = sample.size

    inner_cols = 3
    inner_rows = 4
    inner_gap_x = 8
    inner_gap_y = 8
    outer_pad = 22
    block_gap_x = 28
    block_gap_y = 34
    label_h = 72

    block_w = inner_cols * tile_w + (inner_cols - 1) * inner_gap_x
    block_h = inner_rows * tile_h + (inner_rows - 1) * inner_gap_y + label_h

    canvas_w = outer_pad * 2 + block_w * 2 + block_gap_x
    canvas_h = outer_pad * 2 + block_h * 2 + block_gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    label_font = load_font(68)

    for block_idx, image_index in enumerate(image_order):
        block_col = block_idx % 2
        block_row = block_idx // 2
        bx = outer_pad + block_col * (block_w + block_gap_x)
        by = outer_pad + block_row * (block_h + block_gap_y)

        for method_pos, method_id in enumerate(method_order):
            grid_row = method_pos // inner_cols
            grid_col = method_pos % inner_cols
            x = bx + grid_col * (tile_w + inner_gap_x)
            y = by + grid_row * (tile_h + inner_gap_y)

            tile_path = tile_map.get((method_id, image_index))
            if tile_path is None or not tile_path.is_file():
                raise FileNotFoundError(f"Missing tile for method={method_id}, image_index={image_index}")

            tile = Image.open(tile_path).convert("RGB")
            canvas.paste(tile, (x, y))

            if method_id == "DAF":
                highlight = ImageDraw.Draw(canvas)
                highlight.rectangle((x, y, x + tile_w, y + tile_h), outline=(210, 40, 40), width=5)

        bbox = draw.textbbox((0, 0), panel_labels[block_idx], font=label_font)
        text_w = bbox[2] - bbox[0]
        text_x = bx + (block_w - text_w) // 2
        text_y = by + inner_rows * tile_h + (inner_rows - 1) * inner_gap_y + 6
        draw.text((text_x, text_y), panel_labels[block_idx], fill="black", font=label_font)

    canvas.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
