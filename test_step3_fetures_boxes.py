import cv2
import numpy as np
import os
import glob

# 1. Đọc ảnh (dùng chính tấm ảnh bạn vừa chạy rất đẹp ở step 2)
img_path = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images\IMG_0485_MOV-0002_jpg_jpg.rf.717f7b3c21791ee970fb662667f9359b.jpg"
src_bgr = cv2.imread(img_path)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

# 2. Otsu thuần túy + Morphological Opening khử nhiễu
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)

# 3. Watershed tách hạt
sure_bg = cv2.dilate(opening, clean_kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

dist_smooth = cv2.GaussianBlur(dist_transform, (11, 11), 0)
dist_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
dist_dilated = cv2.dilate(dist_smooth, dist_kernel)

local_peaks = (dist_smooth == dist_dilated) & (dist_smooth > 3.5)
sure_fg = np.uint8(local_peaks * 255)

unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(src_bgr, markers)

# ==========================================================
# 4. TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC VÀ VẼ BOUNDING BOX
# ==========================================================
# Tạo một bản sao ảnh để vẽ kết quả
vis_img = src_bgr.copy()

# markers nhận các giá trị từ 2 trở đi là các hạt gạo riêng biệt
# (markers == 1 là nền, markers == -1 là đường ranh giới)
grain_labels = np.unique(markers)
grain_labels = [lbl for lbl in grain_labels if lbl > 1]

print(f"--- ĐANG ĐO ĐẠC {len(grain_labels)} HẠT GẠO ---")
print(f"{'ID':<5} | {'Diện tích':<10} | {'Dài (L)':<8} | {'Rộng (W)':<8} | {'Tỷ lệ L/W':<10} | {'Dự đoán sơ bộ'}")
print("-" * 65)

for idx, lbl in enumerate(grain_labels, start=1):
    # Tạo mask riêng cho đúng 1 hạt gạo có nhãn lbl
    grain_mask = np.uint8(markers == lbl) * 255
    
    # Tìm đường viền của hạt gạo này
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = contours[0]
    
    # 1. Tính diện tích (Area)
    area = cv2.contourArea(cnt)
    if area < 30:  # Bỏ qua các hạt bụi siêu nhỏ (nhiễu)
        continue
        
    # 2. Tính Chiều dài & Chiều rộng bằng hình chữ nhật xoay (Rotated Bounding Box)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (dim1, dim2), angle = rect
    length = max(dim1, dim2)
    width = min(dim1, dim2)
    
    # 3. Tính tỷ lệ Aspect Ratio (Dài / Rộng)
    aspect_ratio = length / width if width > 0 else 0
    
    # 4. Gợi ý nhãn sơ bộ (Heuristic đơn giản để hỗ trợ Semi-Auto):
    # Hạt chuẩn thường thon dài (aspect_ratio >= 2.2) và diện tích đủ lớn (area >= 400)
    if aspect_ratio >= 2.2 and area >= 400:
        label_text = "Normal"
        box_color = (0, 255, 0)  # Xanh lá: Gạo chuẩn
    else:
        label_text = "Defective"
        box_color = (0, 0, 255)  # Đỏ: Gạo gãy/nhỏ
        
    # 5. Vẽ Bounding Box xoay ôm sát hạt gạo lên ảnh
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)
    cv2.drawContours(vis_img, [box_points], 0, box_color, 2)
    
    # In ra terminal xem 5 hạt đầu tiên
    if idx <= 5:
        print(f"{idx:<5} | {area:<10.1f} | {length:<8.1f} | {width:<8.1f} | {aspect_ratio:<10.2f} | {label_text}")

# Lưu ảnh kết quả
out_dir = r"F:\AAA\Image\Project\step3_results"
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "bounding_boxes.jpg"), vis_img)
print("-" * 65)
print(f"Đã xuất ảnh có vẽ Bounding Box tại: {os.path.join(out_dir, 'bounding_boxes.jpg')}")