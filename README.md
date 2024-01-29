# Flood Segmentation

### Dataset
Dataset link [download link (Google Drive)](https://drive.google.com/drive/folders/1jY72M3wW6hVrtNanXG_k-NKuKpEt_p0A?usp=sharing).


### Weights

Downloading weights:

- Download trained weight and move it into `./weights/`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1-fciOovNtRASXkryfJrMejCSQyuQLd45/view?usp=sharing).

### Mô tả APIs
- `/api/process_image`: Nhận dữ liệu đầu vào là ảnh dạng base64, kết quả trả về là ảnh đã được segmented và chuyển thành dạng base64

- `/api/compute_pixels`: Nhận dữ liệu đầu vào là ảnh dạng base64, kết quả trả về là số lượng pixels được segmented là nước

- `/api/convert_image`: Nhận dữ liệu đầu vào là ảnh được upload lên, kết quả trả về là ảnh đầu vào chuyển thành dạng base64