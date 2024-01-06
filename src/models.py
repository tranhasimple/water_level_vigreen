import cv2
import torch
import numpy as np
from src.controllers import *
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from PIL import Image as Img
from PIL import ImageDraw, ImageFont
from src.utils import (
    base64_to_image,
    image_to_base64,
    compute_physical_area,
)


ROOT = Path(__file__).parents[1]


class ImageProcessor:
    def __init__(self, image, export_file, cropped_image_path):
        self.image = image
        self.export_file = export_file
        self.cropped_image_path = cropped_image_path
        self.recognize_waterlevel_model = RecognizationModel()

    def process_image(self):
        img = self.image
        cv2.imwrite(img=img, filename=self.export_file)
        recognized_level = self.recognize_waterlevel_model.recognize_water_level(self.export_file, self.cropped_image_path)
        if not recognized_level:
            return None
        result = {
            "water_level": recognized_level,
        }
        return result

    def process_base64(self):
        img = base64_to_image(self.image)
        cv2.imwrite(img=img, filename=self.export_file)

        recognized_level = self.recognize_waterlevel_model.recognize_water_level(self.export_file, self.cropped_image_path)
        print(recognized_level)
        if not recognized_level:
            return None
        recognized_level_base64 = image_to_base64(recognized_level["cropped_image"])
        result = {
            "base64image": recognized_level_base64,
            "water_level": recognized_level["water_level"],
        }
        return result

    def process_regions(self):
        img = self.image
        segmented_image = self.recognize_waterlevel_model.segment_multiple_areas(img)
        if not segmented_image:
            return None
        cv2.imwrite(self.export_file, segmented_image["segmentation"])
        result = {
            "regions": segmented_image["regions"],
        }
        return result

    def process_regions_base64(self):
        img = base64_to_image(self.image)
        segmented_image = self.segmentation_model.segment_multiple_areas(img)
        if not segmented_image:
            return None
        segmented_image_base64 = image_to_base64(segmented_image["segmentation"])
        result = {
            "base64image": segmented_image_base64,
            "regions": segmented_image["regions"],
        }
        return result

import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
model_id = "microsoft/trocr-small-printed"

processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

class OCRModel():
    def __init__(self) -> None:
        self.model = easyocr.Reader(['en'])
        self.processor = processor
        self.model = model

    def image2text(self, path):
        image = Image.open(path)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            return int(generated_text)
        except Exception as ex:
            
            print(ex)
            return generated_text
        
    def image2text_(self, path):
        result = self.model.readtext(path)

        try:
            return int(result[0][1])
        except Exception as ex:
            
            print(ex)
            return result

class RecognizationModel:
    def __init__(self, weight_path="weights/best.pt"):
        self.weight_path = weight_path
        self.yolo_model = YOLO(self.weight_path)
        self.ocr_model = OCRModel()


    def recognize_water_level(self, image_path, cropped_img_path):
        try:

            # get number object from image
            nums = self.detect_num(image_path=image_path)
            converted_nums = self.xyxy2xywh(nums)

            lowest_num = self.choose_lowest_num(converted_nums)
            x, y, w, h = lowest_num

            # load image
            image = Img.open(image_path)

            # save cropped num to temp path
            image.crop((x - w/2, y - h/2,  x + w/2, y + h/2)).save(cropped_img_path)
            image_copy = cv2.imread(image_path)

            # Định nghĩa các tham số cho hình chữ nhật
            pt1 = (int(x - w/2), int(y - h/2))
            pt2 = (int(x + w/2), int(y + h/2))  # Góc dưới bên phải
            color = (0, 255, 0)  # Màu trong định dạng BGR

            # Vẽ hình chữ nhật trên ảnh
            cv2.rectangle(image_copy, pt1, pt2, color, thickness=2)
            label_position = (pt2[0], pt1[1] - 10)  # Vị trí label sẽ hiển thị, 10 pixel phía trên góc trên của hình chữ nhật

            

            ocr_text = self.ocr_model.image2text(cropped_img_path)

            # Hiển thị label trên ảnh
            cv2.putText(image_copy, str(f"water_level = {ocr_text}"), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(0,0,255), thickness=1)
            cv2.imwrite(img=image_copy, filename=image_path)
            return {
                "cropped_image": image_copy,
                "water_level": ocr_text
            }
        except Exception as ex:
        
            print("Exception ====> ", ex)
            return None

    def detect_num(self, image_path):
        results = self.yolo_model.predict(image_path, save=False, imgsz=320, conf=0.4, save_crop=False)
        num_cordinates = []
        for r in results:
            for box in r.boxes:
                cls = box.cls
                if int(cls) == 1:
                    num_cordinates.append([float(x) for x in box.xyxy[0]])

        return np.array(num_cordinates)
    
    def choose_lowest_num(self, nums):
        min_y = nums[0][1]
        min_num = nums[0]

        if len(nums)>1:
            for cor in nums[1:]:
                if cor[1] > min_y:
                    min_y = cor[1]
                    min_num = cor
        return min_num
    
    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y
