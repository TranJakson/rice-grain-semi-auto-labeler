import cv2 
import numpy as np 
import os 
import glob # thư viện gì đây ???

img_dir = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images"

# Ok thì biến này làm gì đây nào ???
img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
if not img_paths:
    print("Không tìm thấy ảnh!")
    exit()

img_path = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images\IMG_0485_HEIC_jpg.rf.9dcc2e9b12e478ea34b73daf00ded2b2.jpg"
print(f"Đang thử nghiệm trên ảnh: {os.path.basename(img_path)}")

# Đọc ảnh và chuyển sang ảnh xám (Grayscale)
src_bgr = cv2.imread(img_path) 
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY) # tại sao phải bỏ biến vào như vậy bỏ thằng vào luôn không được hả

# 3. Tạo Structuring Element (Kernel hình elip, kích thước lớn hơn bề ngang hạt gạo một chút)
# Giả sử bề ngang hạt gạo khoảng 15-25 pixel, ta chọn kernel khoảng 31x31
# Khúc này mình chưa hiểu là tại sao phải tạo mấy cái kernel ??? kernel là gì ?? Những kích thước đó là lấy từ đâu ra vậy ???
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31)) 


# 4. Áp dụng White Top - Hat: cv2.morphologyEx với cờ cv2.MORPH_TOPHAT
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

# 5. So sánh phân ngưỡng Otsu trực tiếp vs Otsu sau Top-hat 
_, direct_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, tophat_otsu = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

# 6. Lưu ảnh kết quả ra thư mục để kiểm tra
out_dir = r"F:\AAA\Image\Project\step1_results"

os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "1_original_gray.jpg"), gray)
cv2.imwrite(os.path.join(out_dir, "2_tophat_enhanced.jpg"), tophat)
cv2.imwrite(os.path.join(out_dir, "3_direct_otsu_failed.jpg"), direct_otsu)
cv2.imwrite(os.path.join(out_dir, "4_tophat_otsu_success.jpg"), tophat_otsu)

print(f"Hoàn thành! Bạn hãy vào xem kết quả tại thư mục: {out_dir}")

