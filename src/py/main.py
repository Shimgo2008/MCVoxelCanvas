"""
python3 main.py
    help: --help
    file: --input_image <input_image_path>
    output: --output_image <output_image_path>
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from colormap import color_palette


def convert_image_fast(input_image: Image.Image, shades: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """
    【高速化版】ハイブリッドアプローチで画像を変換する。
    """
    palette = np.array([shade.rgb for shade in shades])
    image_in_arr = np.array(input_image, dtype=np.float64)
    height, width, _ = image_in_arr.shape
    num_colors = len(palette)

    print(num_colors)
    print("Pre-calculating color costs for all pixels...")

    diff = image_in_arr[:, :, np.newaxis, :] - palette[np.newaxis, np.newaxis, :, :]
    # ユークリッド距離を計算 -> (h, w, n)
    print("Calculating color costs...")
    all_color_costs = np.linalg.norm(diff, axis=3)
    print("Color costs calculated.")
    image_out_arr = np.zeros_like(image_in_arr)
    print("Starting pixel-wise processing...")
    index_map = np.zeros_like(image_in_arr[:, :, 0], dtype=object)
    print("Processing each pixel...")

    # ループ内では、事前計算したcost_colorを使い、cost_gradientのみを計算する
    for y in tqdm(range(height), desc="Processing Rows (Fast)"):
        for x in range(width):

            # 事前計算したColor Costの配列を取得
            color_costs = all_color_costs[y, x]

            min_color_cost = np.min(color_costs)
            normalized_distance = min_color_cost / 441.7  # np.sqrt(255**2 * 3)

            decay_factor = 1.0 - normalized_distance
            W_GRADIENT_DYNAMIC = 0.03 * decay_factor  # 距離0で0.03、距離が最大に近づくと0
            # print(W_GRADIENT_DYNAMIC)
            W_COLOR_DYNAMIC = 1.0 - W_GRADIENT_DYNAMIC

            # Gradient Costを計算 (この部分は逐次処理が必須)
            gradient_costs = np.zeros(num_colors)
            num_neighbors = 0

            if x > 0:
                grad_new_x = palette - image_out_arr[y, x - 1]
                grad_orig_x = image_in_arr[y, x] - image_in_arr[y, x - 1]
                gradient_costs += np.linalg.norm(grad_new_x - grad_orig_x, axis=1)
                num_neighbors += 1
            if y > 0:
                grad_new_y = palette - image_out_arr[y - 1, x]
                grad_orig_y = image_in_arr[y, x] - image_in_arr[y - 1, x]
                gradient_costs += np.linalg.norm(grad_new_y - grad_orig_y, axis=1)
                num_neighbors += 1

            if num_neighbors > 0:
                gradient_costs /= num_neighbors

            # 総コストを計算
            total_costs = (W_COLOR_DYNAMIC * color_costs) + (W_GRADIENT_DYNAMIC * gradient_costs)

            # 最小コストの色のインデックスを見つける
            best_color_index = np.argmin(total_costs)

            # 最適な色を出力画像に配置
            image_out_arr[y, x] = palette[best_color_index]

            index_map[y, x] = (shades[best_color_index].id, shades[best_color_index].shade)

    return Image.fromarray(image_out_arr.astype(np.uint8)), index_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Color Quantization using Hybrid Approach (Fast Version)")
    parser.add_argument("-i", "--input_image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("-o", "--output_image", type=str, default="out/output_image_fast.png", help="Path to save the output image.")
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=['vanilla', 'all'],
        default='vanilla',
        help="Color palette mode: 'vanilla' for standard 3 shades, 'all' for extended 4 shades."
    )

    args = parser.parse_args()

    print("Loading color palette...")
    shades = color_palette(args.mode)

    print("Loading input image...")
    import_image = Image.open(args.input_image).convert("RGB")

    print("Starting image conversion (Fast version)...")
    output_image, index_map = convert_image_fast(import_image, shades)

    print("Conversion complete!")

    output_path = args.output_image
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_image.save(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(import_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(output_image)
    axes[1].set_title("Converted Image (Fast)")
    axes[1].axis('off')

    plt.show()
