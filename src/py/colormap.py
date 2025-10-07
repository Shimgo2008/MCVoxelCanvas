import json
from typing import List, Literal
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

color_data: str = ""


@dataclass
class Color:
    id: str
    shade: str
    rgb: List[int]
    hex: str

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


def color_palette(mode: Literal['vanilla', 'all'] = 'vanilla') -> List[Color]:
    """Print color_data as pretty JSON when run as a script."""

    color_data_path: str = 'all_color_data.json' if mode == 'all' else 'color_data.json'
    with open(color_data_path, "r", encoding="utf-8") as f:
        color_data = f.read()

    parsed_data = json.loads(color_data)
    shades: List[Color] = [
        Color(
            id=item["id"],
            shade=shade_name,
            **color_data
        )
        for item in parsed_data
        for shade_name, color_data in item["shades"].items()
    ]

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump([color.__dict__ for color in shades], f, indent=4, ensure_ascii=False)

    return shades


if __name__ == "__main__":
    color_palette("hex")
