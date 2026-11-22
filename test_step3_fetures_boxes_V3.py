import cv2
import numpy as np
import os
import glob

# 1. Đọc ảnh
img_path = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images\IMG_0488_MOV-0002_jpg_jpg.rf.4d17fe3ea67c6ca48d291f88252008b5.jpg"
src_bgr = cv2.imread(img_path)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

# 2. Otsu thuần túy + Morphological Opening khử nhiễu
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)

# 3. Watershed tách hạt sơ bộ
sure_bg = cv2.dilate(opening, clean_kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Tối ưu 1: Kernel 25x25 bao phủ trọn vẹn thân hạt gạo dài
dist_smooth = cv2.GaussianBlur(dist_transform, (11, 11), 0)
dist_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
dist_dilated = cv2.dilate(dist_smooth, dist_kernel)

local_peaks = (dist_smooth == dist_dilated) & (dist_smooth > 4.0)
sure_fg = np.uint8(local_peaks * 255)

unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(src_bgr, markers)

# ==========================================================
# 4. TRÍCH XUẤT CÁC HỘP BAN ĐẦU
# ==========================================================
raw_boxes = []
for lbl in np.unique(markers):
    if lbl <= 1: 
        continue
    grain_mask = np.uint8(markers == lbl) * 255
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    if area < 50: # Bỏ qua nhiễu
        continue
        
    rect = cv2.minAreaRect(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    raw_boxes.append({'area': area, 'rect': rect, 'bbox': [x, y, w, h]})

# ==========================================================
# 5. TỐI ƯU 2: NON-MAXIMUM SUPPRESSION (LỌC HỘP ĐỎ LỌT LÒNG)
# ==========================================================
keep = [True] * len(raw_boxes)
for i in range(len(raw_boxes)):
    if not keep[i]: 
        continue
    b1 = raw_boxes[i]['bbox']
    for j in range(len(raw_boxes)):
        if i == j or not keep[j]: 
            continue
        b2 = raw_boxes[j]['bbox']
        # Tính phần giao nhau giữa 2 hộp
        xi1 = max(b1[0], b2[0])
        yi1 = max(b1[1], b2[1])
        xi2 = min(b1[0] + b1[2], b2[0] + b2[2])
        yi2 = min(b1[1] + b1[3], b2[1] + b2[3])
        if xi2 > xi1 and yi2 > yi1:
            inter_area = (xi2 - xi1) * (yi2 - yi1)
            area_j = b2[2] * b2[3]
            # Nếu hộp j nằm lọt thỏm > 50% trong hộp i và có diện tích bé hơn -> Xóa hộp j!
            if (inter_area / area_j > 0.5) and (raw_boxes[i]['area'] > raw_boxes[j]['area']):
                keep[j] = False

clean_boxes = [raw_boxes[k] for k in range(len(raw_boxes)) if keep[k]]

# ==========================================================
# 6. PHÂN LOẠI & TỰ ĐỘNG CHẺ HẠT DÍNH ĐÔI
# ==========================================================
vis_img = src_bgr.copy()
normal_count = 0
defective_count = 0

for b in clean_boxes:
    rect = b['rect']
    area = b['area']
    dim1, dim2 = rect[1]
    length = max(dim1, dim2)
    width = min(dim1, dim2)
    aspect_ratio = length / width if width > 0 else 0
    is_dim1_length = (dim1 > dim2)
    
    # Kiểm tra hạt dính đôi (width to gấp đôi bình thường)
    if (width >= 26.0) and (area >= 1000.0) and (aspect_ratio < 2.1):
        # Bổ đôi hình học thành 2 hạt chuẩn
        angle = rect[2]
        theta = np.deg2rad(angle if not is_dim1_length else angle + 90)
        ux = np.cos(theta)
        uy = np.sin(theta)
        shift = width / 4.0
        cx, cy = rect[0]
        c1 = (cx + shift * ux, cy + shift * uy)
        c2 = (cx - shift * ux, cy - shift * uy)
        new_width = width / 2.0
        new_d1 = length if is_dim1_length else new_width
        new_d2 = new_width if is_dim1_length else length
        
        for r in [(c1, (new_d1, new_d2), angle), (c2, (new_d1, new_d2), angle)]:
            pts = np.int32(cv2.boxPoints(r))
            cv2.drawContours(vis_img, [pts], 0, (0, 255, 0), 2)
            normal_count += 1
    else:
        # Hạt đơn thông thường:
        if aspect_ratio >= 1.7 and area >= 200:
            box_color = (0, 255, 0) # Xanh: Gạo chuẩn
            normal_count += 1
        else:
            box_color = (0, 0, 255) # Đỏ: Gạo lỗi/gãy
            defective_count += 1
            
        pts = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(vis_img, [pts], 0, box_color, 2)

total_boxes = normal_count + defective_count

# Lưu kết quả
out_dir = r"F:\AAA\Image\Project\step3_results"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "bounding_boxes_v3_clean.jpg")
cv2.imwrite(out_file, vis_img)

print("=" * 55)
print("--- KẾT QUẢ SAU KHI LỌC BẰNG NMS ---")
print(f"Tổng số hạt phát hiện: {total_boxes}")
print(f"  - Số hạt chuẩn (Xanh lá): {normal_count}")
print(f"  - Số hạt lỗi (Đỏ):         {defective_count}")
print(f"Tỷ lệ gạo đạt chuẩn: {(normal_count / total_boxes) * 100:.2f}%")
print(f"Ảnh kết quả đã lưu tại: {out_file}")
print("=" * 55)