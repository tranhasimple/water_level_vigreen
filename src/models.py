import cv2
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
    def __init__(self, image, export_file):
        self.image = image
        self.export_file = export_file
        self.segmentation_model = SegmentationModel()

    def process_image(self):
        img = self.image
        segmented_image = self.segmentation_model.segment_image(img)
        if not segmented_image:
            return None
        cv2.imwrite(self.export_file, segmented_image["segmentation"])
        result = {
            "num_pixel": segmented_image["num_pixel"],
        }
        return result

    def process_base64(self):
        img = base64_to_image(self.image)

        segmented_image = self.segmentation_model.segment_image(img)
        if not segmented_image:
            return None
        segmented_image_base64 = image_to_base64(segmented_image["segmentation"])
        result = {
            "base64image": segmented_image_base64,
            "num_pixel": segmented_image["num_pixel"],
        }
        return result

    def process_regions(self):
        img = self.image
        segmented_image = self.segmentation_model.segment_multiple_areas(img)
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


class SegmentationModel:
    def __init__(self, weight_path="weights/best.pt"):
        self.weight_path = weight_path
        self.model = YOLO(self.weight_path)

    def segment_image(self, img):
        try:
            results = self.model.predict(
                source=img, conf=0.25, save=False, show_labels=False, show_boxes=False
            )

            if results[0].masks == None:
                return None

            im = Img.fromarray(img)
            binary_mask = Img.new("1", im.size, 0)
            draw = ImageDraw.Draw(binary_mask)
            im_mask = Image.new("RGBA", im.size, (0, 0, 0, 0))
            im_draw = ImageDraw.Draw(im_mask)

            for result in results:
                for mask_ in result.masks:
                    polygon = mask_.xy[0]
                    draw.polygon(polygon, outline=1, fill=1)
                    im_draw.polygon(polygon, fill=(255, 0, 0, 128))

            im = Image.alpha_composite(im.convert("RGBA"), im_mask)
            binary_mask_array = np.array(binary_mask)
            white_pixels_np = np.sum(binary_mask_array == 1)
            
            im0 = np.array(im)

            pos = (0, 0)
            x, y = pos
            text = f"Water area: {compute_physical_area(white_pixels_np)} m2"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 6
            font_thickness = 5
            text_color = (0, 255, 0)
            text_color_bg = (255, 255, 255, 10)
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(
                im0, pos, (x + text_w + 20, y + text_h + 20), text_color_bg, -1
            )
            cv2.putText(
                im0,
                text,
                (x, y + text_h + font_scale - 1),
                font,
                font_scale,
                text_color,
                font_thickness,
            )

            return {"segmentation": im0, "num_pixel": int(white_pixels_np)}
        except Exception as e:
            print(e)
            return None

    def segment_multiple_areas(self, img):
        try:
            im0s = img
            results = self.model.predict(
                source=img, conf=0.25, save=False, show_labels=False, show_boxes=False
            )

            if results[0].masks == None:
                return None

            im = Img.fromarray(img)
            binary_mask = Img.new("1", im.size, 0)
            draw = ImageDraw.Draw(binary_mask)
            im_mask = Image.new("RGBA", im.size, (0, 0, 0, 0))
            im_draw = ImageDraw.Draw(im_mask)
            regions = []
            for result in results:
                for index, mask_ in enumerate(result.masks):
                    polygon = mask_.xy[0]
                    draw.polygon(polygon, outline=1, fill=1)
                    im_draw.polygon(polygon, fill=(255, 0, 0, 128))

                    # find regions
                    sub_binary_mask = Img.new("1", im.size, 0)
                    sub_binary_mask_draw = ImageDraw.Draw(sub_binary_mask)
                    sub_binary_mask_draw.polygon(polygon, outline=1, fill=1)

                    sub_binary_mask_array = np.array(sub_binary_mask)
                    sub_binary_mask_image = (sub_binary_mask_array * 255).astype(
                        np.uint8
                    )
                    sub_binary_mask_image_bgr = cv2.cvtColor(
                        sub_binary_mask_image, cv2.COLOR_GRAY2BGR
                    )
                    sub_binary_mask_image_bgr = cv2.resize(
                        sub_binary_mask_image_bgr, (1280, 640)
                    )
                    sub_white_pixels_np = np.sum(sub_binary_mask_array == 1)
                    regions.append(
                        {
                            "id": f"region{index + 1}",
                            "num_pixel": int(sub_white_pixels_np),
                            "area": compute_physical_area(sub_white_pixels_np),
                            "polygon": polygon.tolist(),
                        }
                    )

            binary_mask_array = np.array(binary_mask)
            white_pixels_np = np.sum(binary_mask_array == 1)

            im = Image.alpha_composite(im.convert("RGBA"), im_mask)
            im0 = np.array(im)
            im0s = im0
            
            pos = (0, 0)
            x, y = pos
            text = f"Water area: {compute_physical_area(white_pixels_np)} m2"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 5
            font_thickness = 10
            text_color = (0, 255, 0)
            text_color_bg = (255, 255, 255, 128)
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            cv2.rectangle(
                im0, pos, (x + text_w + 20, y + text_h + 20), text_color_bg, -1
            )
            cv2.putText(
                im0,
                text,
                (x, y + text_h + font_scale - 1),
                font,
                font_scale,
                text_color,
                font_thickness,
            )
            
            alpha = 0.4
            im0s = cv2.addWeighted(im0, alpha, im0s, 1 - alpha, 0)

            for region in regions:
                polygon = region["polygon"]
                center_point = np.mean(polygon, axis=0)

                center_point = np.mean(polygon, axis=0)
                center_point = tuple(map(int, center_point))
                text = f'{region["id"]}: {region["area"]} m2'
                font_scale = 3
                font_thickness = 10
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_w, text_h = text_size
                cv2.rectangle(
                    im0,
                    center_point,
                    (center_point[0] + text_w + 20, center_point[1] + text_h + 20),
                    text_color_bg,
                    -1,
                )
                cv2.putText(
                    im0,
                    text,
                    (center_point[0], center_point[1] + text_h + font_scale - 1),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                )
                del region["polygon"]

            regions = sorted(regions, key=lambda x: x["area"])

            return {"segmentation": im0, "regions": regions}
        except Exception as e:
            print(e)
            return None
