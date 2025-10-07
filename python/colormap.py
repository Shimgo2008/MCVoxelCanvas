
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import json
from typing import List, Literal

color_data: str = ""


def plot_colors_in_rgb_space(hex_colors: List[str]) -> None:
    """
    HEXカラーコードのリストを受け取り、3DのRGB空間に点としてプロットする。

    Args:
        hex_colors: '#'から始まるHEXカラーコード文字列のリスト。
    """
    rgb_values = [
        tuple(int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        for h in hex_colors
    ]

    r = [rgb[0] for rgb in rgb_values]
    g = [rgb[1] for rgb in rgb_values]
    b = [rgb[2] for rgb in rgb_values]

    plot_colors = [tuple(val / 255.0 for val in rgb) for rgb in rgb_values]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(r, g, b, c=plot_colors, s=60, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Red (0-255)')
    ax.set_ylabel('Green (0-255)')
    ax.set_zlabel('Blue (0-255)')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)
    ax.set_title('Color Distribution in RGB Space')

    ax.view_init(elev=20, azim=135)

    ax.grid(True)

    plt.show()


def color_palette(type: Literal['hex', 'rgb'], mode: Literal[None, 'all'] = None) -> NDArray[np.str_] | NDArray[NDArray[np.int_]]:
    """Print color_data as pretty JSON when run as a script."""
    color_data_path: str = 'all_color_data.json' if mode else 'color_data.json'
    with open(color_data_path, "r", encoding="utf-8") as f:
        color_data = f.read()

    parsed_data = json.loads(color_data)
    shades = [shade for item in parsed_data for shade in item["shades"].values()]

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(shades, f, indent=4, ensure_ascii=False)

    results = []
    for color in shades:
        results.append(color[type])

    return np.array(results)


if __name__ == "__main__":
    color_palette("hex")
