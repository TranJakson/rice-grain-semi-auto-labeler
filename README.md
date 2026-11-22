# Automated Rice Grain Counting & Quality Assessment
### Module: Semi-Automatic Labeling Tool (Pure DIP + Tkinter GUI + ML Feature Extraction)

Dự án môn học Xử lý ảnh số (Digital Image Processing) — Đếm số lượng & Đánh giá chất lượng hạt gạo tự động (Gạo chuẩn vs Gạo gãy/lỗi) bằng Computer Vision thuần túy và Học máy cổ điển.

---

## 📌 Tổng quan kiến trúc 3 Chặng

```
[Ảnh gốc thô] ──> [Chặng 1: Xử lý ảnh thuần túy] ──> [Chặng 2: Trích xuất đặc trưng] ──> [Chặng 3: Human Review GUI]
                        • Otsu / Top-Hat                    • Area, Length, Width           • Click đổi nhãn Normal/Defect
                        • Watershed tách hạt                • Aspect Ratio, Circularity     • Chẻ đôi hạt dính (Phím S)
                        • NMS lọc hộp lọt lòng              • RGB & CIELAB Colors           • Xuất file CSV cho nhóm train ML
```

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

Yêu cầu môi trường: Python 3.10+ với các thư viện:
```bash
pip install opencv-python numpy scipy pillow
```

Khởi chạy ứng dụng giao diện trực quan:
```bash
python run_labeler.py
```

---

## ⌨️ Phím tắt thao tác nhanh trên Giao diện GUI

* **Cuộn chuột**: Phóng to / Thu nhỏ ảnh tại vị trí chuột.
* **Giữ `Space` + Kéo chuột trái**: Di chuyển khung nhìn (Pan).
* **Click chuột trái vào hạt (`Q`)**: Đổi nhanh nhãn **Normal (Xanh lá) ↔ Defective (Đỏ)**.
* **Click chuột phải vào hạt**: Xóa nhanh hộp rác / nhiễu nền.
* **Kéo chuột trên vùng trống (`W`)**: Vẽ thêm hộp mới cho hạt bị sót.
* **Phím `S` + Click vào hộp**: Chẻ đôi hạt gạo dính **mạn sườn** (song song).
* **Phím `E` + Click vào hộp**: Chẻ đôi hạt gạo dính **đầu - đuôi**.
* **Phím `1` / `2`**: Đặt nhanh toàn bộ ảnh thành **Normal (1)** hoặc **Defective (2)**.
* **Phím `Ctrl + Z`**: Hoàn tác (Undo).
* **Phím `R`**: Chạy lại Auto-Detect riêng cho ảnh hiện tại với các thông số vừa tinh chỉnh.
* **Phím `V`**: Xem cửa sổ trực quan 4 bước xử lý ảnh (Top-Hat, Otsu, Distance Transform, Watershed).
* **Phím `Enter`**: Lưu ảnh hiện tại, cập nhật dataset CSV và tự động sang ảnh tiếp theo.

---

## 📂 Dữ liệu đầu ra (`labeled_dataset/`)

* **`rice_features_dataset.csv`**: File tổng hợp 25 cột đặc trưng hình học + màu sắc của từng hạt gạo để nhóm huấn luyện Machine Learning (Random Forest / SVM).
* **`sessions/*.json`**: Lưu lịch sử duyệt và thông số tinh chỉnh riêng của từng ảnh.
* **`visualized/*_labeled.jpg`**: Ảnh xuất kèm bảng thống kê chất lượng hạt gạo.
* **`yolo_labels/*.txt`**: File nhãn chuẩn tọa độ bounding box chuẩn hóa.
