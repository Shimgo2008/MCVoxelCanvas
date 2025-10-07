import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

from colormap import color_palette

W_COLOR = 9.0
W_GRADIENT = 1.0


def convert_image_fast(input_image: Image.Image, palette: np.ndarray) -> Image.Image:
    """
    【高速化版】ハイブリッドアプローチで画像を変換する。
    """
    image_in_arr = np.array(input_image, dtype=np.float64)
    height, width, _ = image_in_arr.shape
    num_colors = len(palette)

    print(num_colors)

    print("Pre-calculating color costs for all pixels...")

    diff = image_in_arr[:, :, np.newaxis, :] - palette[np.newaxis, np.newaxis, :, :]
    # ユークリッド距離を計算 -> (h, w, n)
    all_color_costs = np.linalg.norm(diff, axis=3)

    image_out_arr = np.zeros_like(image_in_arr)

    # ループ内では、事前計算したcost_colorを使い、cost_gradientのみを計算する
    for y in tqdm(range(height), desc="Processing Rows (Fast)"):
        for x in range(width):

            # 事前計算したColor Costの配列を取得
            color_costs = all_color_costs[y, x]

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
            total_costs = (W_COLOR * color_costs) + (W_GRADIENT * gradient_costs)

            # 最小コストの色のインデックスを見つける
            best_color_index = np.argmin(total_costs)

            # 最適な色を出力画像に配置
            image_out_arr[y, x] = palette[best_color_index]

    return Image.fromarray(image_out_arr.astype(np.uint8))


if __name__ == "__main__":
    print("Loading color palette...")
    palette_rgb = np.array(color_palette('rgb'))

    print("Loading input image...")
    import_image = Image.open("asunoyozora.png").convert("RGB")

    print("Starting image conversion (Fast version)...")

    output_image = convert_image_fast(import_image, palette_rgb)

    print("Conversion complete!")
    output_image.save("output_image_fast.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(import_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(output_image)
    axes[1].set_title("Converted Image (Fast)")
    axes[1].axis('off')

    plt.show()
