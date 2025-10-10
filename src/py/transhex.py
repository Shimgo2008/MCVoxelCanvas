import argparse
import struct
import os
import zlib
from dataclasses import dataclass

from typing import Optional, List, Tuple, Literal, Dict

from build3dpos import block_structure, format_3d_list

# NBT Tag Type Constants
TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12

TAG_NAMES = {
    0: "End", 1: "Byte", 2: "Short", 3: "Int", 4: "Long", 5: "Float",
    6: "Double", 7: "ByteArray", 8: "String", 9: "List", 10: "Compound",
    11: "IntArray", 12: "LongArray"
}

head = """
;00 End
;01 Byte
;02 Short
;03 Int
;04 Long
;05 Float
;06 Double
;07 Byte Array
;08 String
;09 List
;0A Compound
;0B Int Array
;0C Long Array

;基本的な構造: [タグID (1byte)] [名前の長さ (2byte)] [名前 (UTF-8)] [データ本体]

"""


@dataclass
class MainData:
    size: Tuple[int, int, int]  # (width, height, depth)
    palette: List[Dict]  # list of dict entries for BlockStatePalette
    states: List[int]


@dataclass
class arg:
    input_file: str
    output_file: Optional[str] = None
    mode: Literal['compact', 'detailed'] = 'compact'


def bin2nbt_styled(data: bytes, mode: Literal['compact', 'detailed'] = 'compact') -> tuple[str, MainData]:
    """
    Parses NBT binary data and outputs a styled, human-readable string representation,
    similar to a debugger view, including tag types, name lengths, and payloads.
    """

    hex_str = data.hex()
    hex_data: List[str] = [hex_str[i:i + 2] for i in range(0, len(hex_str), 2)]
    hex_len: int = len(hex_data)

    ci: int = 0
    indent_level: int = 0
    output_lines: List[str] = []
    show_values = (mode == 'detailed')

    main_data = MainData(size=(0, 0, 0), palette=[], states=[])

    def _format_numeric_value(tag_type: int, payload_bytes: bytes) -> Optional[str]:
        """Return string of parsed value for numeric tag types, or None if not applicable."""
        try:
            if tag_type == TAG_BYTE:
                return str(int.from_bytes(payload_bytes, 'big', signed=True))
            if tag_type == TAG_SHORT:
                return str(int.from_bytes(payload_bytes, 'big', signed=True))
            if tag_type == TAG_INT:
                return str(int.from_bytes(payload_bytes, 'big', signed=True))
            if tag_type == TAG_LONG:
                return str(int.from_bytes(payload_bytes, 'big', signed=True))
            if tag_type == TAG_FLOAT:
                # big-endian float
                return str(struct.unpack('>f', payload_bytes)[0])
            if tag_type == TAG_DOUBLE:
                return str(struct.unpack('>d', payload_bytes)[0])
        except Exception:
            return None
        return None

    def ensure_available(start: int, need: int, what: str = "bytes") -> None:
        if start + need > hex_len:
            available = hex_len - start
            raise ValueError(
                f"Unexpected EOF while reading {what} at offset {start}. "
                f"Needed {need}, but only {available} available."
            )

    def peek_bytes(offset: int, count: int) -> bytes:
        ensure_available(offset, count, f"peek {count}")
        s = ''.join(hex_data[offset:offset + count])
        return bytes.fromhex(s)

    def read_bytes(count: int, what: str = "bytes") -> bytes:
        nonlocal ci
        b = peek_bytes(ci, count)
        ci += count
        return b

    def read_uint(count: int, what: str = "uint") -> int:
        b = read_bytes(count, what)
        return int.from_bytes(b, 'big', signed=False)

    def read_ushort_hex() -> Tuple[int, str]:
        b = read_bytes(2, "ushort")
        val = int.from_bytes(b, 'big')
        return val, b.hex().upper()

    def get_tag_header(offset: int) -> Optional[Tuple[int, int, str, bytes]]:
        ensure_available(offset, 1, "tag type")
        tag_type = int(hex_data[offset], 16)

        if tag_type == TAG_END:
            return None

        start_ci = offset
        ensure_available(start_ci, 3, "tag header (type + name length)")

        tag_type = int(hex_data[start_ci], 16)
        name_len = int(''.join(hex_data[start_ci + 1:start_ci + 3]), 16)

        ensure_available(start_ci + 3, name_len, "tag name bytes")
        name_bytes = peek_bytes(start_ci + 3, name_len)
        name = name_bytes.decode('utf-8', errors='replace')

        return tag_type, name_len, name, name_bytes

    # --- Tag Handlers (Core Logic) ---

    def parse_tag(parents_name: str = None, is_list_element: bool = False, list_index: Optional[int] = None, list_type: Optional[int] = None, collector: Optional[dict] = None):
        nonlocal ci, indent_level

        # 1. Read Header (Type, Name Length, Name)
        try:
            if is_list_element:
                tag_type = list_type
                tag_name = f"[{list_index}]"
                name_len = 0
            else:
                header = get_tag_header(ci)
                if header is None:
                    tag_type = TAG_END
                    tag_name = ""
                    name_len = 0
                else:
                    tag_type, name_len, tag_name, _ = header

                ci += (3 + name_len) if tag_type != TAG_END else 1

        except ValueError as e:
            output_lines.append(f"{'    ' * indent_level}; Error reading tag header: {e}")
            raise e

        tag_type_hex = f"{tag_type:02X}"
        prefix = f"{'    ' * indent_level}{tag_type_hex}"

        if tag_type == TAG_END:
            output_lines.append(f"{'    ' * (indent_level - 1)}00")
            return

        if not is_list_element:
            name_len_hex = f"{name_len:04X}"
            prefix += f" {name_len_hex} {tag_name}"

        if tag_type in (TAG_BYTE, TAG_SHORT, TAG_INT, TAG_FLOAT):
            size = {TAG_BYTE: 1, TAG_SHORT: 2, TAG_INT: 4, TAG_FLOAT: 4}[tag_type]
            payload_bytes = read_bytes(size, TAG_NAMES[tag_type])

            if tag_type == TAG_INT:
                value = _format_numeric_value(tag_type, payload_bytes)
                if parents_name == "Size" and (tag_name == "x" or tag_name == "y" or tag_name == "z"):
                    if tag_name == "x":
                        main_data.size = (int(value), main_data.size[1], main_data.size[2])
                    elif tag_name == "y":
                        main_data.size = (main_data.size[0], int(value), main_data.size[2])
                    elif tag_name == "z":
                        main_data.size = (main_data.size[0], main_data.size[1], int(value))

                if show_values:
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()} (Value: {value})")
                else:
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")
            elif tag_type == TAG_SHORT:
                if show_values:
                    value = _format_numeric_value(tag_type, payload_bytes)
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()} (Value: {value})")
                else:
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")
            else:  # TAG_BYTE, TAG_FLOAT
                output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")

        elif tag_type in (TAG_LONG, TAG_DOUBLE):
            size = {TAG_LONG: 8, TAG_DOUBLE: 8}[tag_type]
            payload_bytes = read_bytes(size, TAG_NAMES[tag_type])

            if tag_type == TAG_LONG:
                if show_values:
                    value = _format_numeric_value(tag_type, payload_bytes)
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()} (Value: {value})")
                else:
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")
            else:  # TAG_DOUBLE
                if show_values:
                    value = _format_numeric_value(tag_type, payload_bytes)
                    if value is not None:
                        output_lines.append(f"{prefix} {payload_bytes.hex().upper()} (Value: {value})")
                    else:
                        output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")
                else:
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")

        elif tag_type == TAG_STRING:
            str_len, str_len_hex = read_ushort_hex()
            str_bytes = read_bytes(str_len, "string payload")
            s = str_bytes.decode('utf-8', errors='replace')
            output_lines.append(f"{prefix} {str_len_hex} {s}")

            if collector is not None:
                if tag_name == 'Name':
                    collector['Name'] = s

                elif parents_name == 'Properties':
                    collector.setdefault('Properties', {})[tag_name] = s

        elif tag_type in (TAG_BYTE_ARRAY, TAG_INT_ARRAY, TAG_LONG_ARRAY):
            type_name = TAG_NAMES[tag_type]

            array_len = read_uint(4, f"{type_name} length")
            array_len_hex = f"{array_len:08X}"

            element_size = {TAG_BYTE_ARRAY: 1, TAG_INT_ARRAY: 4, TAG_LONG_ARRAY: 8}[tag_type]
            data_size = array_len * element_size

            payload_bytes = read_bytes(data_size, f"{type_name} payload")

            payload_snippet = payload_bytes.hex().upper()
            split_bytes = [payload_snippet[i:i + element_size * 2] for i in range(0, len(payload_snippet), element_size * 2)]

            output_lines.append(f"{prefix} {array_len_hex} (Length: {array_len})")
            for i in split_bytes:
                value = _format_numeric_value({TAG_BYTE_ARRAY: TAG_BYTE, TAG_INT_ARRAY: TAG_INT, TAG_LONG_ARRAY: TAG_LONG}[tag_type], bytes.fromhex(i))
                output_lines.append(f"{'    ' * (indent_level + 1)}{i} (Value: {value})")

            if tag_type == TAG_LONG_ARRAY and tag_name == 'BlockStates':
                try:
                    states = []
                    for off in range(0, len(payload_bytes), 8):
                        chunk = payload_bytes[off:off + 8]
                        states.append(f"0x{chunk.hex().upper()}")
                    main_data.states = states
                    output_lines.append(f"{'    ' * (indent_level + 1)}Parsed BlockStates count: {len(states)}")
                except Exception as e:
                    output_lines.append(f"{'    ' * (indent_level + 1)}; Error parsing BlockStates: {e}")

        elif tag_type == TAG_LIST:
            list_type = read_uint(1, "list type")
            list_type_hex = f"{list_type:02X}"
            list_type_name = TAG_NAMES.get(list_type, f"Unknown({list_type})")

            list_len = read_uint(4, "list length")
            list_len_hex = f"{list_len:08X}"

            output_lines.append(f"{prefix} {list_type_hex} ({list_type_name}) {list_len_hex} (Count: {list_len})")

            indent_level += 1
            for idx in range(list_len):
                # Pass the list's name as parents_name so child compounds/strings know their parent
                # さらに collector を伝搬する（必要な場合に collector を作成するロジックが子側で働く）
                if list_type in (TAG_COMPOUND, TAG_LIST):
                    parse_tag(parents_name=tag_name, is_list_element=True, list_index=idx, list_type=list_type, collector=collector)
                elif list_type != TAG_END:
                    size = {
                        TAG_BYTE: 1, TAG_SHORT: 2, TAG_INT: 4, TAG_LONG: 8,
                        TAG_FLOAT: 4, TAG_DOUBLE: 8,
                    }.get(list_type, -1)

                    element_type_hex = f"{list_type:02X}"
                    element_prefix = f"{'    ' * indent_level}{element_type_hex} {tag_name}[{idx}]"

                    if size > 0:
                        element_bytes = read_bytes(size, f"list {list_type_name} element")
                        if show_values and list_type in (TAG_BYTE, TAG_SHORT, TAG_INT, TAG_LONG, TAG_FLOAT, TAG_DOUBLE):
                            val = _format_numeric_value(list_type, element_bytes)
                            if val is not None:
                                output_lines.append(f"{element_prefix} {element_bytes.hex().upper()} (Value: {val})")
                            else:
                                output_lines.append(f"{element_prefix} {element_bytes.hex().upper()}")
                        else:
                            output_lines.append(f"{element_prefix} {element_bytes.hex().upper()}")
                    elif list_type == TAG_STRING:
                        str_len, str_len_hex = read_ushort_hex()
                        str_bytes = read_bytes(str_len, "list string payload")
                        s = str_bytes.decode('utf-8', errors='replace')
                        output_lines.append(f"{element_prefix} {str_len_hex} {s}")
                    elif list_type in (TAG_BYTE_ARRAY, TAG_INT_ARRAY, TAG_LONG_ARRAY):
                        type_name = TAG_NAMES[list_type]
                        array_len = read_uint(4, f"list {type_name} length")
                        element_sz = {TAG_BYTE_ARRAY: 1, TAG_INT_ARRAY: 4, TAG_LONG_ARRAY: 8}[list_type]
                        data_size = array_len * element_sz
                        payload_bytes = read_bytes(data_size, f"list {type_name} payload")

                        payload_snippet = payload_bytes[:min(16, data_size)].hex().upper()
                        if data_size > 16:
                            payload_snippet += "..."
                        output_lines.append(f"{element_prefix} {array_len:08X} (Length: {array_len}) {payload_snippet}")
                    else:
                        raise ValueError(f"Unsupported list element type: {list_type_hex}")

            indent_level -= 1

        elif tag_type == TAG_COMPOUND:
            output_lines.append(prefix)

            # 親から渡された collector を保持。BlockStatePalette のリスト要素であれば新しい collector を作る。
            current_collector = collector
            append_collector = False
            if is_list_element and parents_name == 'BlockStatePalette' and list_type == TAG_COMPOUND:
                current_collector = {}
                append_collector = True

            indent_level += 1
            while ci < hex_len:
                ensure_available(ci, 1, "compound inner tag type peek")
                if hex_data[ci] == "00":
                    parse_tag(parents_name=tag_name, is_list_element=False, collector=current_collector)
                    break
                parse_tag(parents_name=tag_name, is_list_element=False, collector=current_collector)

            if ci >= hex_len and (ci == 0 or hex_data[ci-1] != "00"):  # noqa
                output_lines.append(f"{'    ' * indent_level}; Error: Compound did not terminate with TAG_END.")  # noqa

            indent_level -= 1

            if append_collector and current_collector is not None:
                main_data.palette.append(current_collector)

        else:
            raise ValueError(f"Unsupported tag type encountered: {tag_type_hex} ({TAG_NAMES.get(tag_type)}) at offset {ci}")

    try:
        parse_tag(is_list_element=False)

        if ci < hex_len:
            output_lines.append(f"\n; WARNING: {hex_len - ci} trailing bytes remaining after parsing.")

    except Exception as e:
        output_lines.append(f"\n; FATAL ERROR during parsing at byte offset {ci}: {e}")

    return "\n".join(output_lines), main_data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Styled NBT Binary to Text Transformer.")

    arg_parser.add_argument("-i", "--input_file", type=str, help="Input binary NBT file")
    arg_parser.add_argument("-o", "--output_file", type=str, help="Output text file (optional, defaults to stdout)")
    arg_parser.add_argument("-m", "--mode", choices=['compact', 'detailed'], default='compact', help="Display mode")
    arg_parser.add_argument("-d", "--debug", default=True, action='store_false', help="Enable debug mode (overrides input/output)")

    args = arg_parser.parse_args()

    if args.debug:
        print("--- DEBUG MODE ENABLED ---")
        args.input_file = "beacon.litematic"
        args.output_file = "beacon.r"
        args.mode = 'compact'

    if args.input_file.endswith('.litematic'):
        try:
            with open(args.input_file, 'rb') as f:
                compressed_data = f.read()
                data = zlib.decompress(compressed_data, zlib.MAX_WBITS | 16)
        except Exception as e:
            print(f"Error reading or decompressing litematic file: {e}")
            exit(1)
    else:
        try:
            with open(args.input_file, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input_file}' not found.")
            exit(1)

    result_string, main_data = bin2nbt_styled(data, mode=args.mode)

    structure = block_structure(
        size=main_data.size,
        palette=main_data.palette,
        blockstates=[int(s, 16) for s in main_data.states]
    )

    if args.output_file:

        if args.mode == 'detailed':
            result_string = head + result_string

        output_dir = os.path.dirname(args.output_file)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_file, "w") as f:
            f.write(result_string)

            f.write("\n\n; --- Block Structure ---\n\n")
            result_string = format_3d_list(structure.tolist())
            f.write(result_string)
