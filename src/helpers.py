import base64
from PIL import Image
from io import BytesIO

def image_to_base64(image_byte):
        base64_string = base64.b64encode(image_byte).decode("utf-8")

        return base64_string
    
def base64_to_image(base64_string):

    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)

    return image_data