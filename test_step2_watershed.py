import cv2
import numpy as np
import os
import glob # Dùng làm gì ??

# 1. Đọc lại ảnh đầu tiên và tiền xử lý như Bước 1
img_dir = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images"
target_name = "IMG_0485_MOV-0002_jpg_jpg.rf.717f7b3c21791ee970fb662667f9359b.jpg"
img_path = os.path.join(img_dir, target_name)
src_bgr = cv2.imread(img_path)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY) # dùng làm gì ?? chuyển ảnh xám hả 

# Lọc Top-hat + Otsu để ra ảnh nhị phân 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)) 
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
_, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Dùng làm gì ???

# Làm sạch nhiễu hạt bụi li ti bằng Morphological Opening nhỏ (3x3)
# Bước này làm gì nhỉ tại sao phải có này nhỉ 
# Bước này có nằm trong step 1 top hat không
clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=1)

# ==========================================================
# 2. BẮT ĐẦU WATERSHED
# ==========================================================
# A. Tìm vùng chắc chắn là NỀN (Sure Background): Giãn nở ảnh (Dilation)
# Vùng màu đen chắc chắn 100% không chứa hạt gạo
sure_bg = cv2.dilate(opening, clean_kernel, iterations=3)

# B. Distance Transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# 1. BƯỚC QUAN TRỌNG NHẤT: Làm mượt dist_transform để xóa sạch gợn sóng băm vụn hạt
dist_smooth = cv2.GaussianBlur(dist_transform, (11, 11), 0)

# 2. Kernel tìm đỉnh phải đủ lớn (bằng cỡ 1/2 đến 2/3 bề ngang hạt gạo, khoảng 15x15)
# Để không bao giờ bị 2 đỉnh trên cùng 1 hạt!
dist_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
dist_dilated = cv2.dilate(dist_smooth, dist_kernel)

# 3. Tìm đỉnh duy nhất cho mỗi hạt 
local_peaks = (dist_smooth == dist_dilated) & (dist_smooth > 3.5)

# Chuyển thành ảnh nhị phân tâm hạt
sure_fg = np.uint8(local_peaks * 255)

# D. Vùng chưa chắc chắn
unknown = cv2.subtract(sure_bg, sure_fg)

# E. Connected Components
num_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# F. Watershed
markers = cv2.watershed(src_bgr, markers)

# G. Vẽ kết quả
result_img = src_bgr.copy()
result_img[markers == -1] = [0, 0, 255]

# ==========================================================
# 3. LƯU ẢNH ĐỂ QUAN SÁT TỪNG BƯỚC TOÁN HỌC
# ==========================================================
out_dir = r"F:\AAA\Image\Project\step2_results"
os.makedirs(out_dir, exist_ok=True)

# Chuẩn hóa Distance Transform về 0-255 để mắt người nhìn thấy được
dist_visual = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
dist_visual = np.uint8(dist_visual)
cv2.imwrite(os.path.join(out_dir, "1_dist_transform.jpg"), dist_visual)
cv2.imwrite(os.path.join(out_dir, "2_sure_fg_centers.jpg"), sure_fg)
cv2.imwrite(os.path.join(out_dir, "3_watershed_split.jpg"), result_img)
print(f"Tổng số hạt phát hiện được: {num_markers - 1}")
print(f"Đã lưu kết quả tại: {out_dir}")

