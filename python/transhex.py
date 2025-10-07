import zlib

if __name__ == "__main__":
    result = ""
    with open('data', "rb") as f:
        data = f.read()

    for b in data:
        if 65 <= b <= 90 or 97 <= b <= 122:
            result += chr(b)
        else:
            result += f"{b:02X}"  # 2桁16進数で表現

    print(result)

    with open('data.litematic', 'rb') as f:
        compressed_data = f.read()
        decompressed_data = zlib.decompress(compressed_data, zlib.MAX_WBITS | 16)
        decompressed_data = decompressed_data.hex()

    split = [decompressed_data[i:i + 2] for i in range(0, len(decompressed_data), 2)]

    for i in split:
        # asciiでアルファベットの場合append
        if chr(int(i, 16)).isalpha():
            result += chr(int(i, 16))
        else:
            result += str(int(i, 16))

    print(len(split))
