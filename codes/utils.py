import base64


def img2base64(image_path):
    with open(image_path, "rb") as f:
        # b64encode是编码，b64decode是解码
        base64_data = base64.b64encode(f.read())
    return base64_data

