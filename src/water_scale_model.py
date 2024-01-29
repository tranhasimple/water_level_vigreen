import cv2
import torch
import numpy as np
import os
from src.controllers import *
from PIL import Image, ImageFilter
from pathlib import Path
from ultralytics import YOLO
from PIL import Image as Img
from scipy import ndimage
from configparser import ConfigParser
from src.utils import base64_to_image, image_to_base64, compute_iou

config = ConfigParser()
config.read("config.ini")
ROOT = Path(__file__).parents[1]
h_num_real = float(config.get("water_level", "h_num_real"))
padding = 10


class WaterScaleImageProcessor:
    def __init__(self, image, exported_file):
        self.image = image
        self.exported_file = exported_file
        self.recognize_waterlevel_model = RecognizationModel()

    def process_image(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        recognized_level = self.recognize_waterlevel_model.recognize_water_level(img)
        if not recognized_level:
            return None
        cv2.imwrite(self.exported_file, recognized_level["annotated_image"])
        result = {"water_level": recognized_level["water_level"]}
        return result

    def process_base64(self):
        img = base64_to_image(self.image)
        # cv2.imwrite(img=img, filename=self.export_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        recognized_level = self.recognize_waterlevel_model.recognize_water_level(img)
        if not recognized_level:
            return None
        recognized_level_base64 = image_to_base64(recognized_level["annotated_image"])
        result = {
            "base64image": recognized_level_base64,
            "water_level": recognized_level["water_level"],
        }
        return result


from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# TODO: Load OCR model

model = None
processor = None


class OCRModel:
    def __init__(self) -> None:
        global model
        global processor

        self.model_id = "microsoft/trocr-small-printed"
        if processor == None:
            processor = TrOCRProcessor.from_pretrained(self.model_id, use_fast=False)
        if model == None:
            model = VisionEncoderDecoderModel.from_pretrained(self.model_id)

        self.model = model
        self.processor = processor

    def image2text(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        generated_ids = self.model.generate(pixel_values, max_new_tokens=10)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return abs(int(generated_text))


class RecognizationModel:
    def __init__(self, weight_path="weights/water_scale.pt"):
        self.weight_path = weight_path
        self.yolo_model = YOLO(self.weight_path)
        self.ocr_model = OCRModel()

    def recognize_water_level(self, ori_img):
        global h_num_real

        try:
            # get number object from image
            gauges, _ = self.detect_num(img=ori_img)
            x_g, y_g, w_g, h_g = gauges[0]
            image = Image.fromarray(ori_img)

            gauge_img = image.crop(
                (
                    x_g - w_g / 2 - padding,
                    y_g - h_g / 2 - padding,
                    x_g + w_g / 2 + padding,
                    y_g + h_g / 2 + padding,
                )
            )
            # gauge_img.save("gauge_img.png", "PNG")

            _, nums = self.detect_num(img=gauge_img)
            lowest_num = self.choose_lowest_num(nums)
            x, y, w, h = lowest_num
            no_of_lines = count_lines([x - w / 2, y + h / 2], np.array(gauge_img))
            image_gauge_copy = gauge_img

            # save cropped num to temp path
            num_image = image_gauge_copy.crop(
                (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            )
            ocr_text = self.ocr_model.image2text(image=num_image)

            # Preprocess gauge image
            processed_gauge_img, y_wl = img_processing_water_surface_line(gauge_img)
            diff_level = calculate_diff_level(
                y_num=y + h / 2, y_wl=y_wl, h_num=h, h_num_real=h_num_real
            )

            # calculate real water level
            try:
                real_water_level = ocr_text
                real_water_level = float(real_water_level) - (no_of_lines - 1) * 2
                # real_water_level = ocr_text - diff_level
            except:
                real_water_level = ocr_text

            im0 = ori_img.copy()
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

            # Định nghĩa các tham số cho hình chữ nhật
            pt1 = (int(x - w / 2), int(y - h / 2))
            pt2 = (int(x + w / 2), int(y + h / 2))  # Góc dưới bên phải

            text = str(f"water level: {round(float(real_water_level)/100, 2)} (m)")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 5
            text_color = (0, 255, 0)

            # Hiển thị label trên ảnh
            text_size, _ = cv2.getTextSize(text, font, font_scale, -1)

            text_w, text_h = text_size
            pos = (pt2[0], pt1[1] - text_h - 5)
            x, y = pos

            cv2.rectangle(
                im0,
                (int(x_g - w_g / 2 - padding), int(y_g - h_g / 2 - padding)),
                (int(x_g + w_g / 2 + padding), int(y_g + h_g / 2 + padding)),
                text_color,
                5,
            )

            text_color_bg = (0, 0, 0, 127)
            cv2.rectangle(im0, (0, 0), (text_w + 20, text_h + 20), text_color_bg, -1)
            cv2.putText(
                im0,
                text,
                (5, 5 + text_h + font_scale - 1),
                font,
                font_scale,
                color=text_color,
                thickness=font_thickness,
            )
            return {
                "annotated_image": im0,
                "water_level": real_water_level,
                "y_num": y + h / 2,
                "h_num": h,
                "diff_level": diff_level,
            }
        except Exception as ex:
            print("Exception ====> ", ex)
            return None

    def detect_num(self, img):
        results = self.yolo_model.predict(
            img, save=False, imgsz=320, conf=0.4, save_crop=False
        )
        num_cordinates = []
        gauge_cordinates = []
        for r in results:
            for box in r.boxes:
                cls = box.cls
                if int(cls) == 2:
                    num_cordinates.append([float(x) for x in box.xywh[0]])
                if int(cls) == 0:
                    gauge_cordinates.append([float(x) for x in box.xywh[0]])

        return np.array(gauge_cordinates), np.array(num_cordinates)

    def choose_lowest_num(self, nums):
        min_y = nums[0][1]
        min_num = nums[0]

        if len(nums) > 1:
            for cor in nums[1:]:
                if cor[1] > min_y:
                    min_y = cor[1]
                    min_num = cor
        return min_num


def title_correct_image(gauge_image):
    # Convert the image to grayscale
    gray = gauge_image.convert("L")

    # Apply edge detection method on the image
    edges = gray.filter(ImageFilter.FIND_EDGES)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(np.array(edges), 1, np.pi / 180, 150)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    r_theta = lines[-1]
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr

    if theta > np.pi / 2:
        theta = -180 + theta * 180 / np.pi
    else:
        theta = theta * 180 / np.pi

    # Rotation angle in degrees
    rotated = ndimage.rotate(np.array(gauge_image), theta)

    # Convert the rotated array back to a Pillow image
    rotated_image = Image.fromarray(rotated.astype("uint8"))

    return rotated_image


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


def binarize_image(img):
    # Áp dụng phương pháp Otsu
    thresholded = img.point(lambda p: p > 128 and 255)  # Chuyển thành ảnh đen trắng
    return thresholded


def horizontal_projection(img):
    # Tính tổng số điểm ảnh trên mỗi hàng
    projection = np.sum(np.array(img), axis=1)

    return projection


def img_processing_water_surface_line(gauge_image):
    tilted_img = title_correct_image(gauge_image)
    grayscaled = tilted_img.convert("L")
    binary_image = binarize_image(grayscaled)
    blurred_image = binary_image.filter(ImageFilter.GaussianBlur(radius=5))

    horizontal_projection_result = horizontal_projection(blurred_image)

    # tìm water_surface_line theo
    water_surface_line = find_water_surface_line(
        projection_curve=horizontal_projection_result,
        consecutive_times=10,
        threshold=3100,
    )
    return blurred_image, water_surface_line


def calculate_diff_level(y_num, y_wl, h_num, h_num_real):
    result = h_num_real * abs(y_wl - y_num) / h_num
    return result


def count_lines(lowest_points, img):
    x_lowest, y_lowest = lowest_points
    im0 = img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    edges = cv2.Canny(img, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    count = 0
    prev_bb = None
    current_bb = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width / 5 or h >= height / 10 or h / w > 0.5 or x > x_lowest:
            pass
        else:
            if y >= y_lowest or abs(y - y_lowest) <= 10:
                current_bb = [x, y, x + w, y + h]
                if not prev_bb:
                    count += 1
                    cv2.rectangle(im0, (x, y), (x + w, y + h), 255, 1)
                else:
                    iou = compute_iou(prev_bb, current_bb)
                    print(iou)
                    if iou != 0 and iou > 0.1:
                        cv2.rectangle(im0, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    else:
                        count += 1
                        cv2.rectangle(im0, (x, y), (x + w, y + h), 255, 1)
                prev_bb = [x, y, x + w, y + h]

    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
    cv2.imwrite("contours.png", im0)

    return count
