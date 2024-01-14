# Water level measurement

### Dataset
Dataset link [download link (Google Drive)](put data link).


### Weights

Downloading weights:

- Download trained weight `best.pt` and move it into `./weights/`, which can be found in this [download link (Google Drive)](https://drive.google.com/drive/folders/1-LHIZRIpkV13lPjMF-niYhrDnQ4SIsuL).

Model OCR:
- Model id found in this [ copy model_id (Huggingface) ] (https://huggingface.co/microsoft/trocr-small-printed)

### Mô tả APIs
- `/api/process_image`: Nhận dữ liệu đầu vào là ảnh dạng base64, kết quả trả về là ảnh đã được detect vị trí của số chỉ định mực nước và số mực nước.

### Chỉnh sửa phù hợp với loại thước
- Thay đổi giá trị `h_num_real` trong file `config.ini` bằng độ cao của số tương ứng với đơn vị đo thực tế
- Ví dụ: h_num_real = 1.5 tương ứng với 1 số trong thực tế có độ cao bằng 1.5 (đơn vị đo)