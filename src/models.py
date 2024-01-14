import cv2
import torch
import numpy as np
import os
from src.controllers import *
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from PIL import Image as Img
from PIL import ImageDraw, ImageFont
from configparser import ConfigParser
from src.utils import (
    base64_to_image,
    image_to_base64,
    compute_physical_area,
)

config = ConfigParser()
config.read("config.ini")
ROOT = Path(__file__).parents[1]
h_num_real = float(config.get("image", "h_num_real"))

class ImageProcessor:
    def __init__(self, image, export_file, cropped_image_path):
        self.image = image
        self.export_file = export_file
        self.cropped_image_path = cropped_image_path
        self.gauge_img_path = os.path.join(ROOT,config.get("image", "save_path"), config.get("image", "gauge_path"))
        self.recognize_waterlevel_model = RecognizationModel()

    def process_image(self):
        img = self.image
        cv2.imwrite(img=img, filename=self.export_file)
        recognized_level = self.recognize_waterlevel_model.recognize_water_level(self.export_file, self.cropped_image_path, self.gauge_img_path)
        if not recognized_level:
            return None
        result = {
            "water_level": recognized_level,
        }
        return result

    def process_base64(self):
        img = base64_to_image(self.image)
        cv2.imwrite(img=img, filename=self.export_file)

        recognized_level = self.recognize_waterlevel_model.recognize_water_level(self.export_file, self.cropped_image_path, self.gauge_img_path)
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

# TODO: Load OCR model
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
            return abs(int(generated_text))
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


    def recognize_water_level(self, image_path, cropped_img_path, gauge_img_path):
        global h_num_real

        try:
            # get number object from image
            gauges, nums = self.detect_num(image_path=image_path)
            converted_nums = self.xyxy2xywh(nums)
            converted_gauges = self.xyxy2xywh(gauges)
            x_g, y_g, w_g, h_g = converted_gauges[0]

            lowest_num = self.choose_lowest_num(converted_nums)
            x, y, w, h = lowest_num

            # load image
            image = Img.open(image_path)

            # save cropped num to temp path
            image.crop((x - w/2, y - h/2,  x + w/2, y + h/2)).save(cropped_img_path)
            image.crop((x_g - w_g/2, y_g - h_g/2,  x_g + w_g/2, y_g + h_g/2)).save(gauge_img_path)
            
            gauge_image = cv2.imread(gauge_img_path)
            ocr_text = self.ocr_model.image2text(cropped_img_path)

            # Preprocess gauge image
            processed_gauge_img, y_wl = img_processing_water_surface_line(gauge_image)
            diff_level = calculate_diff_level(y_num=y + h/2, y_wl=y_wl, h_num=h, h_num_real=h_num_real)

            # calculate real water level
            try:
                real_water_level = ocr_text - diff_level
            except:
                real_water_level = "Can not find"

            image_copy = cv2.imread(image_path)

            # Định nghĩa các tham số cho hình chữ nhật
            pt1 = (int(x - w/2), int(y - h/2))
            pt2 = (int(x + w/2), int(y + h/2))  # Góc dưới bên phải
            color = (0, 255, 0)  # Màu trong định dạng BGR

            # Vẽ hình chữ nhật trên ảnh
            cv2.rectangle(image_copy, pt1, pt2, color, thickness=2)
            label_position = (pt2[0], pt1[1] - 10)  # Vị trí label sẽ hiển thị, 10 pixel phía trên góc trên của hình chữ nhật


            text = str(f"water_level = {real_water_level}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 255, 0)
            text_color_bg = (255,255,255)

            # Hiển thị label trên ảnh
            text_size, _ = cv2.getTextSize(text, font, font_scale,-1)
            
            print(text_size)
            text_w, text_h = text_size
            pos = (pt2[0],pt1[1] - text_h - 5)
            x, y = pos
            
            cv2.putText(image_copy, text, (x, y + text_h + font_scale - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=text_color, thickness=font_thickness)
            cv2.imwrite(img=image_copy, filename=image_path)
            return {
                "cropped_image": image_copy,
                "water_level": real_water_level,
                "y_num": y + h/2,
                "h_num": h,
                "diff_level": diff_level
            }
        except Exception as ex:
            print("Exception ====> ", ex)
            return None
        
    def anotate_result(self,  img):
        pass

    def detect_num(self, image_path):
        results = self.yolo_model.predict(image_path, save=True, imgsz=320, conf=0.4, save_crop=True)
        num_cordinates = []
        gauge_cordinates = []
        for r in results:
            for box in r.boxes:
                cls = box.cls
                if int(cls) == 1:
                    num_cordinates.append([float(x) for x in box.xyxy[0]])
                if int(cls) == 0:
                    gauge_cordinates.append([float(x) for x in box.xyxy[0]])

        return np.array(gauge_cordinates), np.array(num_cordinates)
    
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
    

def title_correct_image(gauge_image):
    # img = cv2.imread(path)
    img_copy = gauge_image
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
        theta = theta*180/np.pi
        
    from scipy import ndimage

    #rotation angle in degree
    rotated = ndimage.rotate(gauge_image, theta)
    return rotated

def find_water_surface_line(projection_curve, threshold, consecutive_times):
    count_below_threshold = 0
    water_surface_line = len(projection_curve)

    for row, sum_value in enumerate(projection_curve):
        if sum_value < threshold:
            count_below_threshold += 1
        else:
            count_below_threshold = 0

        if count_below_threshold == consecutive_times:
            water_surface_line = row - consecutive_times + 1
            break

    return water_surface_line

def apply_gaussian_filter(binary_image_path, kernel_size=(5, 5)):
    # Đọc ảnh binary từ đường dẫn
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng Gaussian filter
    blurred_image = cv2.GaussianBlur(binary_image, kernel_size, 0)

    return blurred_image

def binarize_image(img):

    # Áp dụng phương pháp Otsu
    _, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

import numpy as np
def horizontal_projection(img):

    # Tính tổng số điểm ảnh trên mỗi hàng
    projection = np.sum(img, axis=1)

    return projection

def img_processing_water_surface_line(gauge_image):
    tilted_img = title_correct_image(gauge_image)
    grayscaled = cv2.cvtColor(tilted_img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5),0)

    horizontal_projection_result = horizontal_projection(blurred_image)

    # tìm water_surface_line theo 
    water_surface_line = find_water_surface_line(projection_curve=horizontal_projection_result, consecutive_times=10, threshold=3100)
    return blurred_image, water_surface_line

def calculate_diff_level(y_num, y_wl, h_num, h_num_real):
    print(type(y_num), type(y_wl), type(h_num), type(h_num_real))
    print(type(abs(y_wl - y_num)))
    result = h_num_real * abs(y_wl - y_num) / h_num
    return result