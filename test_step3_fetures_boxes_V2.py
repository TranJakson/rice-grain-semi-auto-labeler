import cv2
import numpy as np
import os
import glob

# 1. Đọc ảnh
img_path = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images\IMG_0485_MOV-0002_jpg_jpg.rf.717f7b3c21791ee970fb662667f9359b.jpg"
src_bgr = cv2.imread(img_path)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

# 2. Otsu thuần túy + Morphological Opening khử nhiễu
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)

# 3. Watershed tách hạt sơ bộ
sure_bg = cv2.dilate(opening, clean_kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

dist_smooth = cv2.GaussianBlur(dist_transform, (13, 13), 0)
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
# 4. TRÍCH XUẤT ĐẶC TRƯNG & TỰ ĐỘNG CHẺ ĐÔI HỘP DÍNH
# ==========================================================
vis_img = src_bgr.copy()

grain_labels = np.unique(markers)
grain_labels = [lbl for lbl in grain_labels if lbl > 1]

# Kích thước chuẩn của 1 hạt đơn lẻ trên ảnh này:
# - Chiều dài ~ 56 px
# - Chiều rộng ~ 17 - 18 px
# - Diện tích ~ 700 - 800 px
NORMAL_GRAIN_WIDTH = 17.5
NORMAL_GRAIN_AREA = 740.0

total_boxes = 0
normal_count = 0
defective_count = 0
split_count = 0

for lbl in grain_labels:
    grain_mask = np.uint8(markers == lbl) * 255
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = contours[0]
    
    area = cv2.contourArea(cnt)
    if area < 30:  # Bỏ qua nhiễu
        continue
        
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (dim1, dim2), angle = rect
    
    # Xác định cạnh dài (Length) và cạnh ngắn (Width)
    if dim1 > dim2:
        length = dim1
        width = dim2
        is_dim1_length = True
    else:
        length = dim2
        width = dim1
        is_dim1_length = False
        
    aspect_ratio = length / width if width > 0 else 0
    
    # -------------------------------------------------------------
    # 💡 ĐIỀU KIỆN NHẬN DIỆN "2 HẠT NẰM ÉP SÁT MẠN SƯỜN":
    # 1. Bề rộng mập gấp đôi bình thường (width > 28 px)
    # 2. Diện tích to gấp rưỡi đến gấp đôi hạt đơn (area > 1050 px)
    # 3. Tỷ lệ dài/rộng bị tụt xuống < 2.1 (nhìn vuông vức do 2 hạt ghép lại)
    # -------------------------------------------------------------
    is_double_grain = (width >= 26.0) and (area >= 1000.0) and (aspect_ratio < 2.1)
    
    if is_double_grain:
        split_count += 1
        # BỔ ĐÔI HÌNH HỌC: Chia đôi theo chiều ngang (chia đôi width)
        # Vector chỉ phương của trục ngắn (Width) để dịch chuyển tâm 2 nửa hạt
        # Tính góc của trục ngắn:
        theta = np.deg2rad(angle if not is_dim1_length else angle + 90)
        ux = np.cos(theta)
        uy = np.sin(theta)
        
        # Dịch chuyển tâm về 2 phía một đoạn = 1/4 bề rộng
        shift = width / 4.0
        c1 = (cx + shift * ux, cy + shift * uy)
        c2 = (cx - shift * ux, cy - shift * uy)
        
        # Kích thước mỗi hạt mới sau khi bổ đôi
        new_width = width / 2.0
        new_dim1 = length if is_dim1_length else new_width
        new_dim2 = new_width if is_dim1_length else length
        
        rect1 = (c1, (new_dim1, new_dim2), angle)
        rect2 = (c2, (new_dim1, new_dim2), angle)
        
        # Vẽ 2 hộp màu Xanh lá cho 2 hạt gạo chuẩn đã được tách
        for r in [rect1, rect2]:
            pts = np.int32(cv2.boxPoints(r))
            cv2.drawContours(vis_img, [pts], 0, (0, 255, 0), 2)
            total_boxes += 1
            normal_count += 1
            
    else:
        # Hạt đơn lẻ thông thường:
        # Hạt đơn lẻ thông thường:
        if aspect_ratio >= 1.7 and area >= 200:
            box_color = (0, 255, 0)  # Xanh lá: Gạo chuẩn
            normal_count += 1
        else:
            box_color = (0, 0, 255)  # Đỏ: Gạo gãy/nhỏ
            defective_count += 1
            
        pts = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(vis_img, [pts], 0, box_color, 2)
        total_boxes += 1

# Lưu kết quả
out_dir = r"F:\AAA\Image\Project\step3_results"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "bounding_boxes_v2_split.jpg")
cv2.imwrite(out_file, vis_img)

print("=" * 55)
print("--- KẾT QUẢ TỰ ĐỘNG PHÁT HIỆN & CHẺ ĐÔI HỘP DÍNH ---")
print(f"Số cụm hạt dính đôi đã được tự động bổ đôi: {split_count}")
print(f"Tổng số hạt phát hiện: {total_boxes}")
print(f"  - Số hạt chuẩn (Normal - Xanh):   {normal_count}")
print(f"  - Số hạt lỗi (Defective - Đỏ):    {defective_count}")
print(f"Tỷ lệ gạo đạt chuẩn: {(normal_count / total_boxes) * 100:.2f}%")
print(f"Ảnh kết quả đã lưu tại: {out_file}")
print("=" * 55)

