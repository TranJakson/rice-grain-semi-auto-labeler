"""
==============================================================================
CHẶNG 1 & CHẶNG 2: BỘ LÕI XỬ LÝ ẢNH THUẦN TÚY (DIP) & TRÍCH XUẤT ĐẶC TRƯNG
Đề tài: Automated Rice Grain Counting & Quality Assessment (Semi-Auto Labeling)
==============================================================================
Module này chứa toàn bộ các thuật toán Xử lý ảnh số kinh điển:
  - Chặng 1 (Segmentation):
      1.1 Tiền xử lý: Grayscale + (tùy chọn) Top-Hat xóa nền gỗ / ánh sáng loang
      1.2 Phân ngưỡng tự động: Otsu Thresholding + Morphological Opening khử nhiễu
      1.3 Tách hạt dính: Distance Transform + Gaussian Smooth + Local Maxima + Watershed
      1.4 Hậu xử lý: Non-Maximum Suppression (NMS) lọc hộp lọt lòng & tự động chẻ hạt dính đôi
  - Chặng 2 (Feature Extraction):
      2.1 Đặc trưng hình học: Area, Perimeter, Length, Width, Aspect Ratio,
          Circularity, Solidity, Extent
      2.2 Đặc trưng màu sắc: Mean/Std RGB và CIELAB (nhận diện gạo ố vàng, mốc, sẫm màu)
      2.3 Gợi ý nhãn sơ bộ: Normal (0) hoặc Defective (1)
"""

import cv2
import numpy as np
from scipy import ndimage
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Tuple, Optional


@dataclass
class DIPParams:
    """Bộ tham số xử lý ảnh có thể tinh chỉnh riêng cho từng bức ảnh từ giao diện GUI."""
    use_tophat: bool = False          # Tắt = Otsu thuần (mặc định tốt nhất trên nền gỗ tối); Bật = Top-hat + Otsu
    tophat_kernel: int = 41           # Kích thước kernel elip cho Top-Hat (nên lớn hơn bề ngang hạt)
    thresh_offset: int = 0            # Bù ngưỡng sáng Otsu (-50..+50): âm = bắt thêm hạt mờ, dương = bỏ bớt nền sáng
    open_iterations: int = 1          # Số lần Morphological Opening (3x3) lọc nhiễu bụi
    gauss_kernel: int = 11            # Độ mượt bản đồ Distance Transform (số lẻ: 3, 5, 7, 9, 11, 13...)
    peak_kernel: int = 25             # Kích thước cửa sổ tìm đỉnh Local Maxima (số lẻ: 15..35)
    min_dist_peak: float = 4.0        # Độ cao tối thiểu của đỉnh tâm hạt trên Distance Map
    min_area: float = 50.0            # Diện tích tối thiểu (pixel) để không tính bụi bẩn
    use_nms: bool = True              # Lọc hộp lọt lòng (Non-Maximum Suppression)
    nms_containment: float = 0.50     # Ngưỡng phần trăm lọt lòng để xóa hộp rác bên trong
    auto_split_double: bool = True    # Tự động chẻ đôi 2 hạt nằm ép sát mạn sườn
    double_min_width: float = 26.0    # Bề ngang tối thiểu để nghi ngờ là 2 hạt dính đôi
    double_min_area: float = 1000.0   # Diện tích tối thiểu của cụm dính đôi
    double_max_ar: float = 2.1        # Tỷ lệ Dài/Rộng tối đa của cụm dính đôi
    normal_min_ar: float = 1.7        # Gợi ý nhãn sơ bộ: Aspect Ratio tối thiểu cho hạt chuẩn
    normal_min_area: float = 200.0    # Gợi ý nhãn sơ bộ: Diện tích tối thiểu cho hạt chuẩn

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "DIPParams":
        if not data:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def _ensure_odd(val: int, min_val: int = 3) -> int:
    """Đảm bảo kích thước kernel luôn là số lẻ >= min_val."""
    val = max(int(val), min_val)
    return val if val % 2 == 1 else val + 1


def extract_features_from_roi(
    src_bgr: np.ndarray,
    src_lab: np.ndarray,
    rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    contour: Optional[np.ndarray] = None,
    global_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    CHẶNG 2: Trích xuất đầy đủ vector đặc trưng hình học & màu sắc cho 1 hạt gạo.
    Hỗ trợ cả hạt có sẵn contour từ Watershed lẫn hạt do người dùng vẽ tay / chia đôi.
    """
    h_img, w_img = src_bgr.shape[:2]
    (cx, cy), (dim1, dim2), angle = rect
    length = float(max(dim1, dim2))
    width = float(min(dim1, dim2))
    aspect_ratio = float(length / width) if width > 1e-5 else 0.0

    pts = cv2.boxPoints(rect)
    pts_int = np.int32(np.round(pts))

    # Vùng bao trục tọa độ (Axis-Aligned ROI) để cắt nhanh, tránh quét toàn bộ ảnh 4K
    x_min = max(0, int(np.floor(pts[:, 0].min())))
    y_min = max(0, int(np.floor(pts[:, 1].min())))
    x_max = min(w_img, int(np.ceil(pts[:, 0].max())) + 1)
    y_max = min(h_img, int(np.ceil(pts[:, 1].max())) + 1)

    roi_w = x_max - x_min
    roi_h = y_max - y_min

    # Fallback nếu hộp nằm ngoài rìa ảnh
    if roi_w <= 1 or roi_h <= 1:
        return {
            "cx": float(cx), "cy": float(cy),
            "length": length, "width": width, "angle": float(angle),
            "area": float(length * width), "perimeter": float(2 * (length + width)),
            "aspect_ratio": aspect_ratio, "circularity": 0.0,
            "solidity": 1.0, "extent": 1.0,
            "mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0,
            "std_r": 0.0, "std_g": 0.0, "std_b": 0.0,
            "mean_L": 0.0, "mean_a": 0.0, "mean_b_lab": 0.0,
        }

    roi_bgr = src_bgr[y_min:y_max, x_min:x_max]
    roi_lab = src_lab[y_min:y_max, x_min:x_max]

    # Tạo mặt nạ nhị phân cục bộ (Local Mask) đúng phần thân hạt gạo bên trong hộp
    local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    if contour is not None and len(contour) >= 3:
        shifted_cnt = contour - np.array([[[x_min, y_min]]], dtype=contour.dtype)
        cv2.drawContours(local_mask, [shifted_cnt], -1, 255, thickness=cv2.FILLED)
        cnt_for_shape = contour
    else:
        # Hạt do người dùng vẽ tay hoặc tách đôi -> vẽ hộp xoay, giao với mask Otsu toàn cục
        shifted_box = pts_int - np.array([[x_min, y_min]], dtype=np.int32)
        box_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(box_mask, shifted_box, 255)

        if global_mask is not None:
            g_roi = global_mask[y_min:y_max, x_min:x_max]
            intersect = cv2.bitwise_and(box_mask, g_roi)
            # Nếu bên trong hộp có đủ pixel hạt gạo sáng -> dùng mask thực của hạt
            if cv2.countNonZero(intersect) >= max(15, int(0.15 * cv2.countNonZero(box_mask))):
                local_mask = intersect
            else:
                local_mask = box_mask
        else:
            local_mask = box_mask

        cnts, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt_for_shape = max(cnts, key=cv2.contourArea) if cnts else shifted_box

    # --- 2.1 Đặc trưng hình học (Shape & Geometry Features) ---
    area = float(cv2.contourArea(cnt_for_shape))
    if area < 1.0:
        area = float(cv2.countNonZero(local_mask))
    perimeter = float(cv2.arcLength(cnt_for_shape, True))

    # Độ tròn (Circularity = 4 * pi * A / P^2; hình tròn hoàn hảo = 1.0, hạt dài ~ 0.45 - 0.65)
    circularity = float(4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 1e-5 else 0.0

    # Độ đặc (Solidity = Area / Convex Hull Area)
    hull = cv2.convexHull(cnt_for_shape)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 1e-5 else 1.0

    # Tỷ lệ lấp đầy hộp bao (Extent = Area / (Length * Width))
    box_area = length * width
    extent = float(area / box_area) if box_area > 1e-5 else 0.0

    # --- 2.2 Đặc trưng màu sắc (Color Features: BGR -> RGB và CIELAB) ---
    # Chỉ tính trung bình & độ lệch chuẩn trên các pixel thực sự thuộc thân hạt gạo
    if cv2.countNonZero(local_mask) == 0:
        local_mask[:] = 255

    mean_bgr, std_bgr = cv2.meanStdDev(roi_bgr, mask=local_mask)
    mean_lab, _ = cv2.meanStdDev(roi_lab, mask=local_mask)

    mean_b, mean_g, mean_r = float(mean_bgr[0, 0]), float(mean_bgr[1, 0]), float(mean_bgr[2, 0])
    std_b, std_g, std_r = float(std_bgr[0, 0]), float(std_bgr[1, 0]), float(std_bgr[2, 0])
    mean_L, mean_a, mean_b_lab = float(mean_lab[0, 0]), float(mean_lab[1, 0]), float(mean_lab[2, 0])

    return {
        "cx": round(float(cx), 2),
        "cy": round(float(cy), 2),
        "length": round(length, 2),
        "width": round(width, 2),
        "angle": round(float(angle), 2),
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "extent": round(extent, 3),
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "std_r": round(std_r, 2),
        "std_g": round(std_g, 2),
        "std_b": round(std_b, 2),
        "mean_L": round(mean_L, 2),
        "mean_a": round(mean_a, 2),
        "mean_b_lab": round(mean_b_lab, 2),
    }


def split_rotated_rect(
    rect: Tuple[Tuple[float, float], Tuple[float, float], float],
    along_width: bool = True,
) -> Tuple[Tuple, Tuple]:
    """
    Chẻ đôi hình học (Geometry Splitting) một hộp xoay thành 2 hộp bằng nhau.
    - along_width=True : Chẻ dọc theo chiều dài (tách 2 hạt nằm ép sát mạn sườn).
    - along_width=False: Chẻ ngang theo bề rộng (tách 2 hạt dính đầu-đuôi).
    """
    (cx, cy), (dim1, dim2), angle = rect
    is_dim1_length = dim1 >= dim2
    length = dim1 if is_dim1_length else dim2
    width = dim2 if is_dim1_length else dim1

    if along_width:
        # Dịch tâm dọc theo phương trục ngắn (width)
        theta = np.deg2rad(angle + 90.0 if is_dim1_length else angle)
        shift = width / 4.0
        new_w = width / 2.0
        new_l = length
    else:
        # Dịch tâm dọc theo phương trục dài (length)
        theta = np.deg2rad(angle if is_dim1_length else angle + 90.0)
        shift = length / 4.0
        new_w = width
        new_l = length / 2.0

    ux, uy = float(np.cos(theta)), float(np.sin(theta))
    c1 = (cx + shift * ux, cy + shift * uy)
    c2 = (cx - shift * ux, cy - shift * uy)

    new_d1 = new_l if is_dim1_length else new_w
    new_d2 = new_w if is_dim1_length else new_l

    return (c1, (new_d1, new_d2), angle), (c2, (new_d1, new_d2), angle)


def run_dip_pipeline(
    src_bgr: np.ndarray,
    params: Optional[DIPParams] = None,
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    CHẶNG 1 + CHẶNG 2 TỔNG HỢP:
    Chạy toàn bộ pipeline Xử lý ảnh thuần túy trên 1 ảnh BGR và trả về:
      1. grains: Danh sách các hạt gạo phát hiện được, mỗi phần tử là dict:
            {
              "rect": ((cx, cy), (w, h), angle),
              "label": "Normal" | "Defective",
              "features": {...},
              "source": "auto" | "auto_split" | "manual"
            }
      2. debug_imgs: Các ảnh trung gian của từng bước (để hiển thị trực quan khi báo cáo):
            - "step1_enhanced": Ảnh xám hoặc Top-Hat
            - "step2_otsu": Ảnh nhị phân sau Otsu + Opening
            - "step3_distance": Bản đồ Distance Transform (đã chuẩn hóa 0-255)
            - "step4_watershed": Ảnh gốc có vẽ đường ranh giới đỏ Watershed
    """
    if params is None:
        params = DIPParams()

    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)

    # ----------------------------------------------------------
    # BƯỚC 1.1: Tiền xử lý (Grayscale hoặc Top-Hat xóa nền gỗ)
    # ----------------------------------------------------------
    if params.use_tophat:
        th_k = _ensure_odd(params.tophat_kernel, 15)
        th_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_k, th_k))
        enhanced = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, th_kernel)
    else:
        enhanced = gray.copy()

    # ----------------------------------------------------------
    # BƯỚC 1.2: Phân ngưỡng tự động Otsu (+ bù sáng tùy ảnh) + Opening
    # ----------------------------------------------------------
    otsu_t, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if params.thresh_offset != 0:
        adj_t = float(np.clip(otsu_t + params.thresh_offset, 1.0, 254.0))
        _, thresh = cv2.threshold(enhanced, adj_t, 255, cv2.THRESH_BINARY)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if params.open_iterations > 0:
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, clean_kernel, iterations=params.open_iterations)
    else:
        opening = thresh.copy()

    # ----------------------------------------------------------
    # BƯỚC 1.3: Distance Transform + Local Maxima + Watershed
    # ----------------------------------------------------------
    sure_bg = cv2.dilate(opening, clean_kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    g_k = _ensure_odd(params.gauss_kernel, 3)
    p_k = _ensure_odd(params.peak_kernel, 5)
    dist_smooth = cv2.GaussianBlur(dist_transform, (g_k, g_k), 0)
    dist_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p_k, p_k))
    dist_dilated = cv2.dilate(dist_smooth, dist_kernel)

    local_peaks = (dist_smooth == dist_dilated) & (dist_smooth > params.min_dist_peak)
    sure_fg = np.uint8(local_peaks * 255)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(src_bgr, markers)

    # Ảnh minh họa ranh giới Watershed
    ws_vis = src_bgr.copy()
    ws_vis[markers == -1] = [0, 0, 255]

    dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    dist_vis = np.uint8(dist_vis)

    debug_imgs = {
        "step1_enhanced": enhanced,
        "step2_otsu": opening,
        "step3_distance": dist_vis,
        "step4_watershed": ws_vis,
    }

    # ----------------------------------------------------------
    # BƯỚC 1.4: Trích xuất contour ban đầu từ các nhãn Watershed
    # Tối ưu hóa tốc độ >100x bằng scipy.ndimage.find_objects:
    # chỉ cắt đúng khung nhỏ (ROI) của từng hạt thay vì quét lại cả ảnh 2048x1536!
    # ----------------------------------------------------------
    raw_items = []
    # markers chứa nền=1, ranh giới=-1, các hạt bắt đầu từ 2..N
    pos_markers = np.where(markers > 1, markers - 1, 0)
    slices = ndimage.find_objects(pos_markers)

    for obj_idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sy, sx = sl
        y0, x0 = sy.start, sx.start
        local_mask = (pos_markers[sl] == obj_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt_local = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt_local)
        if area < params.min_area:
            continue

        # Dịch tọa độ contour cục bộ về tọa độ ảnh gốc
        cnt = cnt_local + np.array([[[x0, y0]]], dtype=cnt_local.dtype)
        rect = cv2.minAreaRect(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        raw_items.append({
            "cnt": cnt,
            "area": area,
            "rect": rect,
            "bbox": (x, y, w, h),
        })

    # ----------------------------------------------------------
    # BƯỚC 1.5: Non-Maximum Suppression (NMS) lọc hộp lọt lòng
    # ----------------------------------------------------------
    if params.use_nms and len(raw_items) > 1:
        keep = [True] * len(raw_items)
        for i in range(len(raw_items)):
            if not keep[i]:
                continue
            x1, y1, w1, h1 = raw_items[i]["bbox"]
            for j in range(len(raw_items)):
                if i == j or not keep[j]:
                    continue
                x2, y2, w2, h2 = raw_items[j]["bbox"]
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                if xi2 > xi1 and yi2 > yi1:
                    inter_area = (xi2 - xi1) * (yi2 - yi1)
                    area_j = max(1, w2 * h2)
                    if (inter_area / area_j > params.nms_containment) and (
                        raw_items[i]["area"] > raw_items[j]["area"]
                    ):
                        keep[j] = False
        clean_items = [raw_items[k] for k in range(len(raw_items)) if keep[k]]
    else:
        clean_items = raw_items

    # ----------------------------------------------------------
    # BƯỚC 2: Tách đôi hạt dính mạn sườn + Trích xuất đặc trưng + Gợi ý nhãn
    # ----------------------------------------------------------
    grains: List[Dict] = []
    for item in clean_items:
        rect = item["rect"]
        area = item["area"]
        dim1, dim2 = rect[1]
        length = max(dim1, dim2)
        width = min(dim1, dim2)
        aspect_ratio = (length / width) if width > 1e-5 else 0.0

        is_double = (
            params.auto_split_double
            and (width >= params.double_min_width)
            and (area >= params.double_min_area)
            and (aspect_ratio < params.double_max_ar)
        )

        if is_double:
            r1, r2 = split_rotated_rect(rect, along_width=True)
            for r in (r1, r2):
                feats = extract_features_from_roi(src_bgr, src_lab, r, contour=None, global_mask=opening)
                lbl_str = (
                    "Normal"
                    if (feats["aspect_ratio"] >= params.normal_min_ar and feats["area"] >= params.normal_min_area)
                    else "Defective"
                )
                grains.append({
                    "rect": r,
                    "label": lbl_str,
                    "features": feats,
                    "source": "auto_split",
                })
        else:
            feats = extract_features_from_roi(src_bgr, src_lab, rect, contour=item["cnt"], global_mask=opening)
            lbl_str = (
                "Normal"
                if (feats["aspect_ratio"] >= params.normal_min_ar and feats["area"] >= params.normal_min_area)
                else "Defective"
            )
            grains.append({
                "rect": rect,
                "label": lbl_str,
                "features": feats,
                "source": "auto",
            })

    return grains, debug_imgs
