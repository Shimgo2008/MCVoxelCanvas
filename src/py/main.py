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


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB (0-1) -> linear RGB"""
    a = 0.055
    linear = np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)
    return linear


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (..., 3) in 0-255
    return: same shape, CIELAB (L in ~0-100)
    """
    # normalize to 0-1
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    # linearize
    rgb_lin = _srgb_to_linear(rgb)
    # sRGB (linear) -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    # reshape for dot product
    orig_shape = rgb_lin.shape
    flat = rgb_lin.reshape(-1, 3)
    xyz = flat.dot(M.T)
    # scale to match Lab reference (Xn, Yn, Zn)
    # Use CIE reference white D65 with Yn = 1.0 (we keep normalized)
    # But Lab formula expects ratios X/Xn etc. We'll use Xn=0.95047, Yn=1.00000, Zn=1.08883 in normalized scale.
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[:, 0] / (Xn)
    y = xyz[:, 1] / (Yn)
    z = xyz[:, 2] / (Zn)

    eps = (6/29) ** 3
    kappa = 24389/27  # not strictly needed here

    def f(t):
        return np.where(t > eps, np.cbrt(t), (t * (841/108)) + (4/29))

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    lab = np.stack([L, a, b], axis=1).reshape(orig_shape)
    return lab


def convert_image_fast(input_image: Image.Image, shades: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """
    【高速化版】ハイブリッドアプローチで画像を変換する。
    色空間を RGB から CIELAB に変更し、色差・勾配差を Lab 空間で計算します。
    """
    # パレットRGB（n,3）
    palette_rgb = np.array([shade.rgb for shade in shades], dtype=np.uint8)
    # パレット Lab（n,3）
    palette_lab = rgb_to_lab(palette_rgb)

    image_in_arr = np.array(input_image, dtype=np.float64)  # (h,w,3) in 0-255
    height, width, _ = image_in_arr.shape

    # 入力画像を Lab に変換
    image_in_lab = rgb_to_lab(image_in_arr)

    num_colors = len(palette_lab)

    print(num_colors)
    print("Pre-calculating color costs for all pixels (Lab space)...")

    # 色コストを事前計算（Lab空間）
    diff = image_in_lab[:, :, np.newaxis, :] - palette_lab[np.newaxis, np.newaxis, :, :]
    all_color_costs = np.linalg.norm(diff, axis=3)  # (h,w,n)

    image_out_arr = np.zeros_like(image_in_arr)  # store RGB output
    image_out_arr_lab = np.zeros_like(image_in_lab)  # store Lab of placed colors (for gradient calc)

    index_map = np.zeros_like(image_in_arr[:, :, 0], dtype=object)

    # Lab空間での最大距離の目安（正規化に使用）
    max_lab_distance = 150.0  # 実用値。環境に応じて調整可。

    for y in tqdm(range(height), desc="Processing Rows (Fast)"):
        for x in range(width):
            color_costs = all_color_costs[y, x]  # (n,)

            min_color_cost = np.min(color_costs)
            normalized_distance = min_color_cost / max_lab_distance
            decay_factor = np.sign(1.0 - normalized_distance)
            W_GRADIENT_DYNAMIC = 0.03 * decay_factor
            W_COLOR_DYNAMIC = 1.0 - W_GRADIENT_DYNAMIC

            # Gradient Cost を Lab 空間で計算
            gradient_costs = np.zeros(num_colors)
            num_neighbors = 0

            if x > 0:
                # palette_lab - previous placed pixel (lab)
                grad_new_x = palette_lab - image_out_arr_lab[y, x - 1]  # (n,3)
                grad_orig_x = image_in_lab[y, x] - image_in_lab[y, x - 1]  # (3,)
                gradient_costs += np.linalg.norm(grad_new_x - grad_orig_x, axis=1)
                num_neighbors += 1
            if y > 0:
                grad_new_y = palette_lab - image_out_arr_lab[y - 1, x]
                grad_orig_y = image_in_lab[y, x] - image_in_lab[y - 1, x]
                gradient_costs += np.linalg.norm(grad_new_y - grad_orig_y, axis=1)
                num_neighbors += 1

            if num_neighbors > 0:
                gradient_costs /= num_neighbors

            total_costs = (W_COLOR_DYNAMIC * color_costs) + (W_GRADIENT_DYNAMIC * gradient_costs)
            best_color_index = np.argmin(total_costs)

            # 出力画像に RGB を配置し、Lab 配列も更新
            image_out_arr[y, x] = palette_rgb[best_color_index]
            image_out_arr_lab[y, x] = palette_lab[best_color_index]

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
