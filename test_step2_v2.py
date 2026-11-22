import cv2
import numpy as np
import os
import glob

# 1. Đọc ảnh
img_dir = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images"
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
# Chọn tấm ảnh IMG_0481... (index 11) hoặc ảnh đầu tiên img_paths[0]
src_bgr = cv2.imread(r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images\IMG_0488_HEIC_jpg.rf.b64663e39641305f66034475ed827b13.jpg") 
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

# ==========================================================
# 2. CHỈ DÙNG OTSU THUẦN TÚY (KHÔNG DÙNG TOP-HAT)
# ==========================================================
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Lọc nhiễu nhẹ
clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)

# ==========================================================
# 3. WATERSHED SEGMENTATION
# ==========================================================
# A. Vùng chắc chắn là nền
sure_bg = cv2.dilate(opening, clean_kernel, iterations=3)

# B. Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# C. Làm mượt và tìm đỉnh tâm hạt
dist_smooth = cv2.GaussianBlur(dist_transform, (11, 11), 0)
dist_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
dist_dilated = cv2.dilate(dist_smooth, dist_kernel)

local_peaks = (dist_smooth == dist_dilated) & (dist_smooth > 3.5)
sure_fg = np.uint8(local_peaks * 255)

# D. Vùng chưa chắc chắn
unknown = cv2.subtract(sure_bg, sure_fg)

# E. Đánh nhãn Marker
num_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# F. Thực thi Watershed
markers = cv2.watershed(src_bgr, markers)

# G. Vẽ kết quả ranh giới đỏ
result_img = src_bgr.copy()
result_img[markers == -1] = [0, 0, 255]

# Lưu ảnh kết quả
out_dir = r"F:\AAA\Image\Project\step2_results_otsu_only"
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "thresh_otsu_only.jpg"), thresh)
cv2.imwrite(os.path.join(out_dir, "watershed_otsu_only.jpg"), result_img)

print(f"Tổng số hạt phát hiện được (chỉ dùng Otsu): {num_markers - 1}")
print(f"Xem kết quả tại: {out_dir}")