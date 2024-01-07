import cv2
import numpy as np
import base64
from src.controllers import *
from PIL import Image
from io import BytesIO
from src.helpers import *
from pathlib import Path
import time
import uuid
import os
from ultralytics import YOLO
from PIL import Image as Img
from PIL import ImageDraw
from configparser import ConfigParser

ROOT = Path(__file__).parents[1]
config = ConfigParser()
config.read("config.ini")


def get_unique_name():
    timestamp = time.time()
    unique_id = str(uuid.uuid4().hex)
    unique_name = f"{int(timestamp)}_{unique_id}"

    return str(unique_name)


def save_image(image_path, image_data):
    image_name = get_unique_name() + ".jpg"
    image_path = os.path.join(image_path, image_name)
    image_data = base64_to_image(image_data)
    cv2.imwrite(image_path, image_data)
    return image_path


def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    image_bytes = buffer.tobytes()
    base64_string = base64.b64encode(image_bytes).decode("utf-8")
    return base64_string


def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    img_array = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def is_file(file_path):
    return os.path.isfile(file_path)


def compute_physical_area(num_pixels):
    scale_factor = config.get("camera", "scale_factor")
    GSD = config.get("camera", "GSD")
    area = float(num_pixels) * (float(GSD) * 0.01 / float(scale_factor)) ** 2
    return round(area, 2)

#TODO: Xoay ảnh để vậy thể theo chiều thẳng
def title_correct_image(path, export_path):
    img = cv2.imread(path)
    img_copy = img
    # Convert the img to grayscale
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array

    r_theta = lines[-1]
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr

    if theta > np.pi/2:
        theta = - 180 + theta*180/np.pi
    else:
        theta = - theta*180/np.pi
        
    from scipy import ndimage

    #rotation angle in degree
    rotated = ndimage.rotate(img, theta)
    cv2.imwrite(img = rotated, filename=export_path)