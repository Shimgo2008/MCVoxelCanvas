import argparse
import struct
import os
import zlib

from typing import Optional, List, Tuple, Literal


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

def bin2nbt_styled(data: bytes, mode: Literal['compact', 'detailed']= 'compact') -> str:
    """
    Parses NBT binary data and outputs a styled, human-readable string representation,
    similar to a debugger view, including tag types, name lengths, and payloads.
    """
    
    hex_str = data.hex()
    hex_data: List[str] = [hex_str[i:i + 2] for i in range(0, len(hex_str), 2)]
    hex_len: int = len(hex_data)
    
    ci: int = 0  # Current Index (byte offset)
    indent_level: int = 0
    output_lines: List[str] = []
    show_values = (mode == 'detailed')

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
    
    # --- Stream Reading Helpers (変更なし) ---
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
        name_len = int(''.join(hex_data[start_ci+1:start_ci+3]), 16)
        
        ensure_available(start_ci + 3, name_len, "tag name bytes")
        name_bytes = peek_bytes(start_ci + 3, name_len)
        name = name_bytes.decode('utf-8', errors='replace')
        
        return tag_type, name_len, name, name_bytes

    # --- Tag Handlers (Core Logic) ---

    def parse_tag(is_list_element: bool = False, list_index: Optional[int] = None, list_type: Optional[int] = None):
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
            output_lines.append(f"{'    ' * indent_level}00")
            return

        if not is_list_element:
            name_len_hex = f"{name_len:04X}"
            prefix += f" {name_len_hex} {tag_name}"

        # 2. Read Payload and format output
        
        # Pylanceの警告を回避するため、match/caseではなくif/elifを多用する
        
        if tag_type in (TAG_BYTE, TAG_SHORT, TAG_INT, TAG_FLOAT):
            size = {TAG_BYTE: 1, TAG_SHORT: 2, TAG_INT: 4, TAG_FLOAT: 4}[tag_type]
            payload_bytes = read_bytes(size, TAG_NAMES[tag_type])
            
            if tag_type == TAG_INT:
                if show_values:
                    value = _format_numeric_value(tag_type, payload_bytes)
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()} (Value: {value})")
                else:
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")
            elif tag_type == TAG_SHORT:
                if show_values:
                    value = _format_numeric_value(tag_type, payload_bytes)
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()} (Value: {value})")
                else:
                    output_lines.append(f"{prefix} {payload_bytes.hex().upper()}")
            else: # TAG_BYTE, TAG_FLOAT
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
            else: # TAG_DOUBLE
                if show_values:
                    # attempt to show parsed double value
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

        elif tag_type in (TAG_BYTE_ARRAY, TAG_INT_ARRAY, TAG_LONG_ARRAY):
            type_name = TAG_NAMES[tag_type]
            
            array_len = read_uint(4, f"{type_name} length")
            array_len_hex = f"{array_len:08X}"
            
            element_size = {TAG_BYTE_ARRAY: 1, TAG_INT_ARRAY: 4, TAG_LONG_ARRAY: 8}[tag_type]
            data_size = array_len * element_size
            
            payload_bytes = read_bytes(data_size, f"{type_name} payload")
            
            payload_snippet = payload_bytes[:min(16, data_size)].hex().upper()
            if data_size > 16:
                payload_snippet += "..."
                
            output_lines.append(f"{prefix} {array_len_hex} (Length: {array_len}) {payload_snippet}")

        elif tag_type == TAG_LIST:
            list_type = read_uint(1, "list type")
            list_type_hex = f"{list_type:02X}"
            list_type_name = TAG_NAMES.get(list_type, f"Unknown({list_type})")
            
            list_len = read_uint(4, "list length")
            list_len_hex = f"{list_len:08X}"

            output_lines.append(f"{prefix} {list_type_hex} ({list_type_name}) {list_len_hex} (Count: {list_len})")
            
            indent_level += 1
            for idx in range(list_len):
                if list_type in (TAG_COMPOUND, TAG_LIST):
                    parse_tag(is_list_element=True, list_index=idx, list_type=list_type)
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
            
            indent_level += 1
            while ci < hex_len:
                ensure_available(ci, 1, "compound inner tag type peek")
                if hex_data[ci] == "00":
                    parse_tag(is_list_element=False) 
                    break
                parse_tag(is_list_element=False)
            
            if ci >= hex_len and (ci == 0 or hex_data[ci-1] != "00"):
                 output_lines.append(f"{'    ' * indent_level}; Error: Compound did not terminate with TAG_END.")

            indent_level -= 1
        
        else:
            raise ValueError(f"Unsupported tag type encountered: {tag_type_hex} ({TAG_NAMES.get(tag_type)}) at offset {ci}")

    # --- Main Parsing Loop (変更なし) ---
    
    try:
        parse_tag(is_list_element=False)
        
        if ci < hex_len:
            output_lines.append(f"\n; WARNING: {hex_len - ci} trailing bytes remaining after parsing.")

    except Exception as e:
        output_lines.append(f"\n; FATAL ERROR during parsing at byte offset {ci}: {e}")

    return "\n".join(output_lines)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Styled NBT Binary to Text Transformer.")
    arg_parser.add_argument("-i", "--input_file", type=str, required=True, help="Input binary NBT file")
    arg_parser.add_argument("-o", "--output_file", type=str, help="Output text file (optional, defaults to stdout)")
    arg_parser.add_argument("-m", "--mode", choices=['compact', 'detailed'], default='compact', help="Display mode: 'compact' (no numeric values) or 'detailed' (show numeric values)")
    args = arg_parser.parse_args()
    
    # litematic fileの場合gzip展開する
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

    result_string = bin2nbt_styled(data, mode=args.mode)
    print(result_string)

    if args.output_file:
            output_dir = os.path.dirname(args.output_file)
            
            # ディレクトリパスが存在する場合（空文字列でない場合）にのみ作成
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(args.output_file, "w") as f:
                f.write(result_string)
