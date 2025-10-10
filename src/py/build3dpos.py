import numpy as np
import math


def block_structure(size: tuple[int, int, int], palette: list[dict[str]], blockstates: list[bytes]) -> np.ndarray:
    """
    Long Arrayの圧縮を解除し、ブロックインデックスを3D構造に再構築し、
    最後にパレットのブロック名に一括置換する。
    """

    palette_size = len(palette)
    if palette_size <= 1:
        bits_per_block = 1
    else:
        bits_per_block = math.ceil(math.log2(palette_size))

    mask = (1 << bits_per_block) - 1

    total_bits = len(blockstates) * 64
    max_parsable_blocks = total_bits // bits_per_block

    X, Y, Z = size
    block_count = np.abs(X * Y * Z)
    block_limit = min(block_count, max_parsable_blocks)

    block_indices = []

    for i in range(block_limit):
        start_bit = i * bits_per_block
        long_index = start_bit // 64
        bit_offset = start_bit % 64

        val = blockstates[long_index] >> bit_offset

        if bit_offset + bits_per_block > 64 and long_index + 1 < len(blockstates):
            val |= blockstates[long_index + 1] << (64 - bit_offset)

        idx = val & mask
        block_indices.append(idx)

    while len(block_indices) < block_count:
        block_indices.append(0)

    np_indices = np.array(block_indices, dtype=np.int32)

    try:
        blocks_3d_indices = np_indices.reshape((Y, Z, X))
    except ValueError as e:
        print(f"Error during reshape: {e}. Expected {block_count} blocks, got {len(block_indices)}.")
        return np.array([[]])

    palette_names = [entry['Name'] for entry in palette]

    np_palette = np.array(palette_names, dtype=object)

    blocks_3d_names = np_palette[blocks_3d_indices]

    return blocks_3d_names


def format_3d_list(data: list[list[list[str]]]) -> str:

    output_lines = []

    # 最外層のリストの開始
    output_lines.append('[')

    for i, plane in enumerate(data):

        output_lines.append('    [')

        for j, row in enumerate(plane):

            inner_content = ', '.join(f"'{item}'" for item in row)

            line = f'        [{inner_content}]'

            if j < len(plane) - 1:
                line += ','

            output_lines.append(line)

        end_bracket = '    ]'
        if i < len(data) - 1:
            end_bracket += ','

        output_lines.append(end_bracket)

    # 最外層のリストの終了
    output_lines.append(']')

    return '\n'.join(output_lines)
