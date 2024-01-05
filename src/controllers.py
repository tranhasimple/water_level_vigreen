from flask import request
from src.models import ImageProcessor, Image
from configparser import ConfigParser
from src.helpers import *
import os
from pathlib import Path
import cv2
import time
from src.utils import is_file

config = ConfigParser()
config.read("config.ini")

PARENT_PATH = os.getcwd()
ROOT = Path(__file__).parents[1]


def process_image_controller(base64_image):
    if base64_image:
        if is_file(os.path.join(config.get("image", "image_path"), base64_image)):
            timestamp = time.time()
            file_name = base64_image.split(".")[0] + f"_{int(timestamp)}_result.jpg"
            result_path = os.path.join(
                ROOT, config.get("image", "image_path"), file_name
            )

            image_path = os.path.join(config.get("image", "image_path"), base64_image)
            image_path = os.path.join(ROOT, image_path)

            img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image_processor = ImageProcessor(image=img, export_file=result_path)
            result = image_processor.process_image()
            if not result:
                return {"success": False}
            return {
                "success": True,
                "original_file": base64_image,
                "result_file": file_name,
                "water_level": result["water_level"],
            }
        else:
            temp_path = config.get("image", "image_path")
            save_path = config.get("image", "save_path")
            cropped_path = config.get("image", "cropped_path")
            save_dir = os.path.join(ROOT, save_path)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            image_path = os.path.join(save_dir, temp_path)
            cropped_image_path = os.path.join(save_dir, cropped_path)
            image_processor = ImageProcessor(image=base64_image, export_file=image_path, cropped_image_path=cropped_image_path)
            result = image_processor.process_base64()
            if not result:
                return {"success": False}
            return {
                "success": True,
                "base64image": base64_image,
                "water_level": result["water_level"],
            }
    else:
        return None


def process_regions_controller(base64_image):
    if base64_image:
        if is_file(os.path.join(config.get("image", "image_path"), base64_image)):
            timestamp = time.time()
            file_name = base64_image.split(".")[0] + f"_{int(timestamp)}_result.jpg"
            result_path = os.path.join(
                ROOT, config.get("image", "image_path"), file_name
            )

            image_path = os.path.join(config.get("image", "image_path"), base64_image)
            image_path = os.path.join(ROOT, image_path)

            img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image_processor = ImageProcessor(image=img, export_file=result_path)
            result = image_processor.process_regions()
            if not result:
                return {"success": False}
            return {
                "success": True,
                "original_file": base64_image,
                "result_file": file_name,
                "regions": result["regions"],
            }
        else:
            temp_path = config.get("image", "image_path")
            image_path = os.path.join(ROOT, temp_path)
            image_processor = ImageProcessor(image=base64_image, export_file=image_path)
            result = image_processor.process_regions()
            if not result:
                return {"success": False}
            return {
                "success": True,
                "base64image": result["base64image"],
                "regions": result["regions"],
            }
    else:
        return None


def convert_image_controller(image_data):
    if image_data:
        base64img = image_to_base64(image_byte=image_data)
        return {"base64image": base64img}


def load_image(image_file):
    return image_file.read()
