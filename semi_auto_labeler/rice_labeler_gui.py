"""
==============================================================================
CHẶNG 3: GIAO DIỆN DUYỆT & CHỈNH SỬA NHÃN BÁN TỰ ĐỘNG (SEMI-AUTO REVIEW GUI)
Đề tài: Automated Rice Grain Counting & Quality Assessment
==============================================================================
Tính năng tiết kiệm >80% thời gian gán nhãn toàn bộ thư mục ảnh:
  - Tự động chạy Xử lý ảnh thuần túy (Otsu / Top-Hat + Watershed + NMS + Split đôi)
  - Zoom bằng con lăn chuột, kéo ảnh (Pan) bằng chuột giữa hoặc giữ Space + chuột trái
  - Click chuột trái vào hạt: Đổi nhãn tức thì Normal (Xanh) <-> Defective (Đỏ)
  - Click chuột phải vào hạt: Xóa hộp rác/nhiễu
  - Phím [S] / Chế độ Split: Click vào hộp dính 2 hạt để chẻ đôi tức thì
  - Phím [W] / Chế độ Vẽ thêm: Kéo chuột vẽ hộp bao mới cho hạt bị sót
  - Phím [1] / [2]: Đặt NHANH toàn bộ ảnh thành Normal (ảnh gạo thường) hoặc Defective (ảnh gạo gãy)
  - Phím [Ctrl+Z]: Hoàn tác (Undo) mọi thao tác chỉnh sửa
  - Phím [V]: Xem cửa sổ 4 bước xử lý ảnh (Top-Hat/Gray, Otsu, Distance Transform, Watershed)
  - Phím [Enter / Ctrl+S]: Lưu phiên làm việc, xuất ảnh minh họa, YOLO txt & tự động cập nhật file CSV
"""

import os
import csv
import json
import copy
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from dip_engine import (
    DIPParams,
    run_dip_pipeline,
    extract_features_from_roi,
    split_rotated_rect,
)


# Đường dẫn mặc định trỏ tới thư mục dữ liệu hiện tại của dự án
DEFAULT_IMG_DIR = r"F:\AAA\Image\Project\Rice thay nhu 3 - auto label.v1i.yolov11\train\images"
DEFAULT_OUT_DIR = r"F:\AAA\Image\Project\labeled_dataset"

CSV_COLUMNS = [
    "image_name", "grain_id", "label_name", "label_id", "source",
    "cx", "cy", "length", "width", "angle",
    "area", "perimeter", "aspect_ratio", "circularity", "solidity", "extent",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "mean_L", "mean_a", "mean_b_lab",
]


class RiceSemiAutoLabelerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rice Grain Semi-Auto Labeler (Pure DIP + Human Review)")
        self.root.geometry("1460x860")
        self.root.minsize(1180, 720)
        self.root.configure(bg="#1e1e2e")

        # Đường dẫn làm việc
        self.img_dir: str = DEFAULT_IMG_DIR if os.path.isdir(DEFAULT_IMG_DIR) else ""
        self.out_dir: str = DEFAULT_OUT_DIR
        self._ensure_output_dirs()

        # Danh sách ảnh & trạng thái hiện tại
        self.img_paths: List[str] = []
        self.current_idx: int = -1
        self.approved_set: set = set()
        # Tham số DIP mặc định & bộ nhớ tham số RIÊNG cho từng bức ảnh (không ảnh hưởng ảnh khác)
        self.default_params: DIPParams = DIPParams()
        self.per_image_params: Dict[str, DIPParams] = {}

        # Dữ liệu ảnh đang mở
        self.src_bgr: Optional[np.ndarray] = None
        self.src_rgb: Optional[np.ndarray] = None
        self.src_lab: Optional[np.ndarray] = None
        self.pil_img: Optional[Image.Image] = None
        self.global_otsu_mask: Optional[np.ndarray] = None
        self.debug_imgs: Dict[str, np.ndarray] = {}

        # Danh sách các hạt gạo của ảnh hiện tại & Undo stack
        self.grains: List[Dict] = []
        self.undo_stack: List[List[Dict]] = []
        self.hovered_idx: int = -1
        self.is_dirty: bool = False  # Đánh dấu có thay đổi chưa lưu

        # Tham số hiển thị & tương tác Canvas (Zoom / Pan / Draw)
        self.scale: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self._photo_ref: Optional[ImageTk.PhotoImage] = None
        self.tool_mode = tk.StringVar(value="toggle")  # toggle | draw | split_w | split_l | delete
        self._panning: bool = False
        self._space_down: bool = False
        self._pan_start: Tuple[int, int] = (0, 0)
        self._draw_start_img: Optional[Tuple[float, float]] = None
        self._draw_rect_id: Optional[int] = None

        # Khởi tạo UI & nạp thư mục mặc định
        self._setup_styles()
        self._build_ui()
        self._bind_shortcuts()

        if self.img_dir:
            self.load_folder(self.img_dir)

    # ------------------------------------------------------------------
    # KHỞI TẠO GIAO DIỆN (DARK THEME HIỆN ĐẠI)
    # ------------------------------------------------------------------
    def _ensure_output_dirs(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "visualized"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "yolo_labels"), exist_ok=True)

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bg = "#1e1e2e"
        panel = "#262637"
        fg = "#e6e9ef"
        accent = "#3b82f6"
        style.configure("TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel)
        style.configure("TLabel", background=panel, foreground=fg, font=("Segoe UI", 9))
        style.configure("Header.TLabel", background=panel, foreground="#93c5fd", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", background=bg, foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        style.configure("TCheckbutton", background=panel, foreground=fg, font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", panel)])
        style.configure("TRadiobutton", background=panel, foreground=fg, font=("Segoe UI", 9))
        style.map("TRadiobutton", background=[("active", panel)])
        style.configure("Accent.TButton", background=accent, foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=5)
        style.map("Accent.TButton", background=[("active", "#2563eb")])
        style.configure("Success.TButton", background="#16a34a", foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=6)
        style.map("Success.TButton", background=[("active", "#15803d")])
        style.configure("Danger.TButton", background="#dc2626", foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=5)
        style.map("Danger.TButton", background=[("active", "#b91c1c")])
        style.configure("Tool.TButton", background="#334155", foreground="#f8fafc", font=("Segoe UI", 9), padding=4)
        style.map("Tool.TButton", background=[("active", "#475569")])

    def _build_ui(self):
        # === THANH CÔNG CỤ TRÊN CÙNG (TOP TOOLBAR) ===
        top_bar = tk.Frame(self.root, bg="#181825", height=48, padx=10, pady=6)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top_bar, text="📂 Chọn thư mục ảnh", style="Tool.TButton", command=self.choose_img_folder).pack(side=tk.LEFT, padx=3)
        ttk.Button(top_bar, text="📁 Thư mục lưu Output", style="Tool.TButton", command=self.choose_out_folder).pack(side=tk.LEFT, padx=3)

        tk.Frame(top_bar, width=2, bg="#334155").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(top_bar, text="◀ Trước (A)", style="Tool.TButton", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="Sau (D) ▶", style="Tool.TButton", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="✅ Lưu & Kế tiếp (Enter)", style="Success.TButton", command=self.save_and_next).pack(side=tk.LEFT, padx=6)

        tk.Frame(top_bar, width=2, bg="#334155").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(top_bar, text="🔍 Vừa khung (F)", style="Tool.TButton", command=self.fit_to_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="↩ Hoàn tác (Ctrl+Z)", style="Tool.TButton", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="👁 Xem 4 bước DIP (V)", style="Accent.TButton", command=self.show_dip_steps_popup).pack(side=tk.LEFT, padx=6)

        ttk.Button(top_bar, text="📊 Xuất CSV Tổng hợp", style="Success.TButton", command=self.export_master_csv).pack(side=tk.RIGHT, padx=4)

        # === THÂN CHÍNH (3 CỘT: DANH SÁCH ẢNH | CANVAS ẢNH | BẢNG ĐIỀU KHIỂN) ===
        main_pane = tk.Frame(self.root, bg="#1e1e2e")
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---------- CỘT TRÁI: DANH SÁCH ẢNH & TIẾN ĐỘ ----------
        left_panel = tk.Frame(main_pane, bg="#262637", width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        self.lbl_progress = tk.Label(
            left_panel, text="Đã duyệt: 0 / 0 ảnh", bg="#262637", fg="#4ade80",
            font=("Segoe UI", 10, "bold"), anchor="w", padx=10, pady=8
        )
        self.lbl_progress.pack(fill=tk.X)

        self.progress_bar = ttk.Progressbar(left_panel, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(0, 8))

        # Lọc hiển thị danh sách
        filter_frame = tk.Frame(left_panel, bg="#262637", padx=10)
        filter_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(filter_frame, text="⏭ Tới ảnh chưa duyệt", style="Tool.TButton", command=self.jump_to_next_unapproved).pack(fill=tk.X)

        # Listbox chứa danh sách file ảnh
        list_frame = tk.Frame(left_panel, bg="#262637", padx=8, pady=4)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.img_listbox = tk.Listbox(
            list_frame, bg="#181825", fg="#e2e8f0", selectbackground="#3b82f6",
            selectforeground="#ffffff", font=("Consolas", 9), activestyle="none",
            yscrollcommand=scrollbar.set, borderwidth=0, highlightthickness=0
        )
        self.img_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.img_listbox.yview)
        self.img_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Thống kê nhanh cho ảnh hiện tại
        stat_box = tk.LabelFrame(
            left_panel, text=" Thống kê ảnh hiện tại ", bg="#262637", fg="#93c5fd",
            font=("Segoe UI", 9, "bold"), padx=8, pady=6
        )
        stat_box.pack(fill=tk.X, padx=8, pady=8)

        self.lbl_stat_total = tk.Label(stat_box, text="Tổng số hạt: 0", bg="#262637", fg="#ffffff", font=("Segoe UI", 10, "bold"), anchor="w")
        self.lbl_stat_total.pack(fill=tk.X)
        self.lbl_stat_normal = tk.Label(stat_box, text="● Normal (Xanh): 0", bg="#262637", fg="#4ade80", font=("Segoe UI", 9, "bold"), anchor="w")
        self.lbl_stat_normal.pack(fill=tk.X)
        self.lbl_stat_defect = tk.Label(stat_box, text="● Defective (Đỏ): 0", bg="#262637", fg="#f87171", font=("Segoe UI", 9, "bold"), anchor="w")
        self.lbl_stat_defect.pack(fill=tk.X)
        self.lbl_stat_ratio = tk.Label(stat_box, text="Tỷ lệ gạo chuẩn: 0.0%", bg="#262637", fg="#facc15", font=("Segoe UI", 10, "bold"), anchor="w")
        self.lbl_stat_ratio.pack(fill=tk.X, pady=(4, 0))

        # ---------- CỘT PHẢI: CÔNG CỤ SỬA NHÃN & THAM SỐ DIP ----------
        right_panel = tk.Frame(main_pane, bg="#262637", width=290)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # Khối 1: Chế độ thao tác chuột
        mode_box = tk.LabelFrame(
            right_panel, text=" 1. Chế độ Chỉnh sửa (Human Review) ", bg="#262637", fg="#93c5fd",
            font=("Segoe UI", 9, "bold"), padx=8, pady=6
        )
        mode_box.pack(fill=tk.X, padx=8, pady=(8, 4))

        modes = [
            ("🖱 Đổi nhãn nhanh (Click trái) [Q]", "toggle"),
            ("✏️ Vẽ thêm hạt bị sót (Kéo chuột) [W]", "draw"),
            ("✂️ Chẻ đôi hạt dính sườn [S]", "split_w"),
            ("🔪 Chẻ đôi hạt dính đầu-đuôi [E]", "split_l"),
            ("🗑 Xóa hộp (hoặc Click phải) [X]", "delete"),
        ]
        for text, val in modes:
            ttk.Radiobutton(mode_box, text=text, variable=self.tool_mode, value=val).pack(anchor="w", pady=2)

        # Khối 2: Thao tác hàng loạt (Tiết kiệm 90% cho ảnh gạo thuần!)
        bulk_box = tk.LabelFrame(
            right_panel, text=" 2. Gán nhanh cả ảnh (Ảnh thuần loại) ", bg="#262637", fg="#93c5fd",
            font=("Segoe UI", 9, "bold"), padx=8, pady=6
        )
        bulk_box.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(bulk_box, text="🟢 Đặt TẤT CẢ = Normal [Phím 1]", style="Success.TButton", command=lambda: self.set_all_labels("Normal")).pack(fill=tk.X, pady=2)
        ttk.Button(bulk_box, text="🔴 Đặt TẤT CẢ = Defective [Phím 2]", style="Danger.TButton", command=lambda: self.set_all_labels("Defective")).pack(fill=tk.X, pady=2)

        # Khối 3: Tinh chỉnh tham số Xử lý ảnh (CHỈ ÁP DỤNG RIÊNG CHO ẢNH HIỆN TẠI)
        dip_box = tk.LabelFrame(
            right_panel, text=" 3. Tinh chỉnh DIP (Riêng ảnh này) ", bg="#262637", fg="#93c5fd",
            font=("Segoe UI", 9, "bold"), padx=8, pady=6
        )
        dip_box.pack(fill=tk.X, padx=8, pady=4)

        self.lbl_param_scope = tk.Label(
            dip_box, text="● Mặc định (chỉ tác động ảnh này)", bg="#262637", fg="#94a3b8",
            font=("Segoe UI", 8, "italic"), anchor="w"
        )
        self.lbl_param_scope.pack(fill=tk.X, pady=(0, 3))

        self.var_use_tophat = tk.BooleanVar(value=self.default_params.use_tophat)
        self.var_use_nms = tk.BooleanVar(value=self.default_params.use_nms)
        self.var_auto_split = tk.BooleanVar(value=self.default_params.auto_split_double)

        ttk.Checkbutton(dip_box, text="Dùng Top-Hat khử nền loang sáng", variable=self.var_use_tophat).pack(anchor="w")
        ttk.Checkbutton(dip_box, text="Dùng NMS lọc hộp đỏ lọt lòng", variable=self.var_use_nms).pack(anchor="w")
        ttk.Checkbutton(dip_box, text="Tự động chẻ đôi hạt dính sườn", variable=self.var_auto_split).pack(anchor="w")

        self.var_thresh_off = tk.IntVar(value=self.default_params.thresh_offset)
        self.var_peak_k = tk.IntVar(value=self.default_params.peak_kernel)
        self.var_gauss_k = tk.IntVar(value=self.default_params.gauss_kernel)
        self.var_min_area = tk.IntVar(value=int(self.default_params.min_area))
        self.var_norm_ar = tk.DoubleVar(value=self.default_params.normal_min_ar)
        self.var_norm_area = tk.IntVar(value=int(self.default_params.normal_min_area))

        self._add_slider(dip_box, "Bù ngưỡng Otsu (+/-)", self.var_thresh_off, -50, 50, 2)
        self._add_slider(dip_box, "Peak Kernel (tách hạt)", self.var_peak_k, 11, 35, 2)
        self._add_slider(dip_box, "Gauss Blur (độ mượt)", self.var_gauss_k, 3, 19, 2)
        self._add_slider(dip_box, "Min Area (lọc bụi px)", self.var_min_area, 20, 300, 10)
        self._add_slider(dip_box, "Gợi ý Normal: AR ≥", self.var_norm_ar, 1.1, 3.0, 0.1)
        self._add_slider(dip_box, "Gợi ý Normal: Area ≥", self.var_norm_area, 80, 700, 20)

        dip_btn_row = tk.Frame(dip_box, bg="#262637")
        dip_btn_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Button(dip_btn_row, text="⚡ Chạy lại ảnh này [R]", style="Accent.TButton", command=self.re_run_auto_detect).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(dip_btn_row, text="↺ Mặc định", style="Tool.TButton", command=self.reset_current_image_params).pack(side=tk.RIGHT)

        # Khối 4: Xem thông số hạt đang rê chuột tới (Hover Inspector)
        info_box = tk.LabelFrame(
            right_panel, text=" 4. Thông số hạt đang trỏ chuột ", bg="#262637", fg="#93c5fd",
            font=("Segoe UI", 9, "bold"), padx=8, pady=6
        )
        info_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self.lbl_hover_info = tk.Label(
            info_box, text="Rê chuột vào một hộp gạo\nđể xem chi tiết đặc trưng ML...",
            bg="#262637", fg="#cbd5e1", font=("Consolas", 8), justify=tk.LEFT, anchor="nw"
        )
        self.lbl_hover_info.pack(fill=tk.BOTH, expand=True)

        # ---------- CỘT GIỮA: CANVAS HIỂN THỊ ẢNH TƯƠNG TÁC ----------
        center_frame = tk.Frame(main_pane, bg="#11111b")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(center_frame, bg="#11111b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)

        # === THANH TRẠNG THÁI DƯỚI CÙNG (STATUS BAR) ===
        self.status_bar = tk.Label(
            self.root,
            text="Sẵn sàng | Cuộn chuột: Zoom | Giữ Space + Kéo: Di chuyển ảnh | Click trái: Đổi nhãn | Click phải: Xóa hộp | Enter: Lưu & Kế tiếp",
            bg="#181825", fg="#94a3b8", font=("Segoe UI", 9), anchor="w", padx=10, pady=4
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _add_slider(self, parent, label, var, frm, to, res):
        row = tk.Frame(parent, bg="#262637")
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text=label, bg="#262637", fg="#cbd5e1", font=("Segoe UI", 8), width=18, anchor="w").pack(side=tk.LEFT)
        val_lbl = tk.Label(row, textvariable=var, bg="#262637", fg="#facc15", font=("Consolas", 8, "bold"), width=4, anchor="e")
        val_lbl.pack(side=tk.RIGHT)
        s = tk.Scale(
            row, variable=var, from_=frm, to=to, resolution=res, orient=tk.HORIZONTAL,
            showvalue=False, bg="#262637", fg="#ffffff", troughcolor="#11111b",
            highlightthickness=0, sliderlength=12, length=100
        )
        s.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _bind_shortcuts(self):
        self.root.bind("<KeyPress-space>", self._on_space_down)
        self.root.bind("<KeyRelease-space>", self._on_space_up)
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<Return>", lambda e: self.save_and_next())
        self.root.bind("<Control-s>", lambda e: self.save_current_image(silent=False))
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<q>", lambda e: self.tool_mode.set("toggle"))
        self.root.bind("<w>", lambda e: self.tool_mode.set("draw"))
        self.root.bind("<s>", lambda e: self.tool_mode.set("split_w"))
        self.root.bind("<e>", lambda e: self.tool_mode.set("split_l"))
        self.root.bind("<x>", lambda e: self.tool_mode.set("delete"))
        self.root.bind("<r>", lambda e: self.re_run_auto_detect())
        self.root.bind("<f>", lambda e: self.fit_to_canvas())
        self.root.bind("<v>", lambda e: self.show_dip_steps_popup())
        self.root.bind("<Key-1>", lambda e: self.set_all_labels("Normal"))
        self.root.bind("<Key-2>", lambda e: self.set_all_labels("Defective"))

    # ------------------------------------------------------------------
    # QUẢN LÝ THƯ MỤC ẢNH, THAM SỐ RIÊNG TỪNG ẢNH & SESSION JSON
    # ------------------------------------------------------------------
    def _get_dip_params(self) -> DIPParams:
        return DIPParams(
            use_tophat=self.var_use_tophat.get(),
            thresh_offset=int(self.var_thresh_off.get()),
            use_nms=self.var_use_nms.get(),
            auto_split_double=self.var_auto_split.get(),
            peak_kernel=int(self.var_peak_k.get()),
            gauss_kernel=int(self.var_gauss_k.get()),
            min_area=float(self.var_min_area.get()),
            normal_min_ar=round(float(self.var_norm_ar.get()), 2),
            normal_min_area=float(self.var_norm_area.get()),
        )

    def _apply_params_to_ui(self, params: DIPParams):
        """Đổ bộ tham số (mặc định hoặc riêng của ảnh này) lên các thanh trượt bên phải."""
        self.var_use_tophat.set(params.use_tophat)
        self.var_thresh_off.set(int(params.thresh_offset))
        self.var_use_nms.set(params.use_nms)
        self.var_auto_split.set(params.auto_split_double)
        self.var_peak_k.set(int(params.peak_kernel))
        self.var_gauss_k.set(int(params.gauss_kernel))
        self.var_min_area.set(int(params.min_area))
        self.var_norm_ar.set(round(float(params.normal_min_ar), 2))
        self.var_norm_area.set(int(params.normal_min_area))
        self._update_param_scope_badge(params)

    def _update_param_scope_badge(self, params: DIPParams):
        if params != self.default_params:
            self.lbl_param_scope.config(
                text="✨ Đang dùng thông số RIÊNG cho ảnh này", fg="#facc15"
            )
        else:
            self.lbl_param_scope.config(
                text="● Mặc định (chỉ tác động ảnh này)", fg="#94a3b8"
            )

    def reset_current_image_params(self):
        """Đưa riêng ảnh hiện tại về thông số mặc định và chạy lại Auto-Detect."""
        if self.current_idx < 0 or self.src_bgr is None:
            return
        img_name = os.path.basename(self.img_paths[self.current_idx])
        self.per_image_params.pop(img_name, None)
        self._apply_params_to_ui(self.default_params)
        self.re_run_auto_detect()
        self.status_bar.config(text=f"↺ Đã khôi phục thông số mặc định cho riêng ảnh: {img_name}")

    def _session_path(self, img_path: str) -> str:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.out_dir, "sessions", f"{stem}.json")

    def choose_img_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh hạt gạo", initialdir=self.img_dir or os.getcwd())
        if folder:
            self.load_folder(folder)

    def choose_out_folder(self):
        folder = filedialog.askdirectory(title="Chọn thư mục lưu dataset đầu ra", initialdir=self.out_dir or os.getcwd())
        if folder:
            self.out_dir = folder
            self._ensure_output_dirs()
            self._scan_approved_sessions()
            self._refresh_listbox()
            self.status_bar.config(text=f"Đã đổi thư mục lưu kết quả sang: {self.out_dir}")

    def load_folder(self, folder: str):
        self.img_dir = folder
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        self.img_paths = sorted(list(set(files)))

        if not self.img_paths:
            messagebox.showwarning("Thông báo", f"Không tìm thấy ảnh nào trong thư mục:\n{folder}")
            return

        self.per_image_params.clear()
        self._scan_approved_sessions()
        self._refresh_listbox()
        self.load_image_at(0)

    def _scan_approved_sessions(self):
        self.approved_set = set()
        for p in self.img_paths:
            if os.path.isfile(self._session_path(p)):
                self.approved_set.add(os.path.basename(p))

    def _refresh_listbox(self):
        self.img_listbox.delete(0, tk.END)
        for idx, p in enumerate(self.img_paths):
            name = os.path.basename(p)
            done = name in self.approved_set
            prefix = "[✓] " if done else "[ ] "
            short_name = name if len(name) <= 28 else (name[:16] + "..." + name[-9:])
            self.img_listbox.insert(tk.END, f"{idx+1:03d} {prefix}{short_name}")
            if done:
                self.img_listbox.itemconfig(idx, fg="#4ade80")
        self._update_progress_label()

    def _update_progress_label(self):
        total = len(self.img_paths)
        done = len(self.approved_set)
        pct = (done / total * 100.0) if total > 0 else 0.0
        self.lbl_progress.config(text=f"Đã duyệt: {done} / {total} ảnh ({pct:.1f}%)")
        self.progress_bar["maximum"] = max(1, total)
        self.progress_bar["value"] = done

    def _on_listbox_select(self, event):
        sel = self.img_listbox.curselection()
        if sel and sel[0] != self.current_idx:
            self.load_image_at(sel[0])

    def load_image_at(self, idx: int):
        if idx < 0 or idx >= len(self.img_paths):
            return
        self.current_idx = idx
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)

        # Đọc ảnh (hỗ trợ đường dẫn tiếng Việt bằng imdecode)
        data = np.fromfile(img_path, dtype=np.uint8)
        self.src_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if self.src_bgr is None:
            messagebox.showerror("Lỗi đọc ảnh", f"Không thể mở file:\n{img_path}")
            return

        self.src_rgb = cv2.cvtColor(self.src_bgr, cv2.COLOR_BGR2RGB)
        self.src_lab = cv2.cvtColor(self.src_bgr, cv2.COLOR_BGR2LAB)
        self.pil_img = Image.fromarray(self.src_rgb)

        # Nạp thông số DIP RIÊNG của ảnh này (từ session đã lưu / cache trong phiên),
        # nếu ảnh chưa từng tùy chỉnh -> tự động quay về Mặc định để không bị ảnh hưởng bởi ảnh trước!
        sess_file = self._session_path(img_path)
        saved = None
        if os.path.isfile(sess_file):
            with open(sess_file, "r", encoding="utf-8") as f:
                saved = json.load(f)

        if img_name in self.per_image_params:
            active_params = self.per_image_params[img_name]
        elif saved is not None and "dip_params" in saved:
            active_params = DIPParams.from_dict(saved["dip_params"])
            self.per_image_params[img_name] = active_params
        else:
            active_params = DIPParams()

        self._apply_params_to_ui(active_params)

        # Luôn chạy DIP engine với tham số riêng của ảnh này để có sẵn mask Otsu & 4 ảnh debug
        auto_grains, self.debug_imgs = run_dip_pipeline(self.src_bgr, active_params)
        self.global_otsu_mask = self.debug_imgs["step2_otsu"]

        # Nếu ảnh này đã từng được duyệt trước đó -> tải lại đúng nhãn đã duyệt
        if saved is not None:
            self.grains = []
            for g in saved.get("grains", []):
                r = g["rect"]
                rect_tuple = ((float(r[0][0]), float(r[0][1])), (float(r[1][0]), float(r[1][1])), float(r[2]))
                self.grains.append({
                    "rect": rect_tuple,
                    "label": g["label"],
                    "features": g["features"],
                    "source": g.get("source", "auto"),
                })
            status_msg = f"Đã tải phiên đã duyệt: {img_name} ({len(self.grains)} hạt)"
            self.is_dirty = False
        else:
            self.grains = auto_grains
            status_msg = f"Auto-Detect xong: {img_name} ({len(self.grains)} hạt) — Hãy kiểm tra và bấm Enter để duyệt!"
            self.is_dirty = True

        self.undo_stack.clear()
        self.hovered_idx = -1

        # Highlight trên listbox
        self.img_listbox.selection_clear(0, tk.END)
        self.img_listbox.selection_set(idx)
        self.img_listbox.see(idx)

        self._update_stats()
        self.fit_to_canvas()
        self.status_bar.config(text=status_msg)

    def re_run_auto_detect(self):
        if self.src_bgr is None or self.current_idx < 0:
            return
        self._push_undo()
        img_name = os.path.basename(self.img_paths[self.current_idx])
        params = self._get_dip_params()
        # Lưu thông số tùy chỉnh CHỈ cho đúng bức ảnh hiện tại
        if params != self.default_params:
            self.per_image_params[img_name] = params
        else:
            self.per_image_params.pop(img_name, None)
        self._update_param_scope_badge(params)

        self.grains, self.debug_imgs = run_dip_pipeline(self.src_bgr, params)
        self.global_otsu_mask = self.debug_imgs["step2_otsu"]
        self.is_dirty = True
        self._update_stats()
        self.redraw_canvas()
        self.status_bar.config(
            text=f"⚡ Đã chạy lại Auto-Detect RIÊNG cho {img_name}: {len(self.grains)} hạt (không ảnh hưởng ảnh khác)."
        )

    def prev_image(self):
        if self.current_idx > 0:
            self.load_image_at(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.img_paths) - 1:
            self.load_image_at(self.current_idx + 1)

    def jump_to_next_unapproved(self):
        if not self.img_paths:
            return
        start = (self.current_idx + 1) % len(self.img_paths)
        for offset in range(len(self.img_paths)):
            i = (start + offset) % len(self.img_paths)
            if os.path.basename(self.img_paths[i]) not in self.approved_set:
                self.load_image_at(i)
                return
        messagebox.showinfo("Hoàn tất!", "Tuyệt vời! Tất cả ảnh trong thư mục đều đã được duyệt xong!")

    # ------------------------------------------------------------------
    # LƯU KẾT QUẢ & XUẤT DATASET CSV / YOLO / VISUALIZATION
    # ------------------------------------------------------------------
    def save_current_image(self, silent: bool = False):
        if self.current_idx < 0 or self.src_bgr is None:
            return
        img_path = self.img_paths[self.current_idx]
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]
        h_img, w_img = self.src_bgr.shape[:2]

        # 1. Lưu file Session JSON (kèm thông số DIP riêng của ảnh này để mở lại vẫn giữ nguyên)
        cur_params = self._get_dip_params()
        if cur_params != self.default_params:
            self.per_image_params[img_name] = cur_params
        sess_data = {
            "image_name": img_name,
            "image_width": w_img,
            "image_height": h_img,
            "dip_params": cur_params.to_dict(),
            "grains": [
                {
                    "rect": [[g["rect"][0][0], g["rect"][0][1]], [g["rect"][1][0], g["rect"][1][1]], g["rect"][2]],
                    "label": g["label"],
                    "features": g["features"],
                    "source": g.get("source", "auto"),
                }
                for g in self.grains
            ],
        }
        with open(self._session_path(img_path), "w", encoding="utf-8") as f:
            json.dump(sess_data, f, indent=2, ensure_ascii=False)

        # 2. Xuất ảnh Visualize có vẽ hộp xanh/đỏ + bảng thống kê góc trên
        vis = self.src_bgr.copy()
        n_norm, n_def = 0, 0
        yolo_lines = []
        for g in self.grains:
            is_norm = (g["label"] == "Normal")
            color = (0, 255, 0) if is_norm else (0, 0, 255)
            if is_norm:
                n_norm += 1
            else:
                n_def += 1
            pts = np.int32(np.round(cv2.boxPoints(g["rect"])))
            cv2.drawContours(vis, [pts], 0, color, 2)

            # Tọa độ chuẩn hóa YOLO (0 = Normal, 1 = Defective)
            bx, by, bw, bh = cv2.boundingRect(pts)
            xc = (bx + bw / 2.0) / w_img
            yc = (by + bh / 2.0) / h_img
            wn = bw / w_img
            hn = bh / h_img
            cls_id = 0 if is_norm else 1
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        total = n_norm + n_def
        ratio = (n_norm / total * 100.0) if total > 0 else 0.0
        cv2.rectangle(vis, (10, 10), (540, 105), (20, 20, 30), -1)
        cv2.putText(vis, f"Total: {total} | Normal: {n_norm} | Defective: {n_def}", (22, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(vis, f"Normal Rice Quality: {ratio:.2f}%", (22, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)

        vis_path = os.path.join(self.out_dir, "visualized", f"{stem}_labeled.jpg")
        ok, buf = cv2.imencode(".jpg", vis)
        if ok:
            buf.tofile(vis_path)

        # 3. Lưu file nhãn YOLO .txt
        yolo_path = os.path.join(self.out_dir, "yolo_labels", f"{stem}.txt")
        with open(yolo_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # Cập nhật danh sách đã duyệt
        self.approved_set.add(img_name)
        self.is_dirty = False
        self.img_listbox.delete(self.current_idx)
        short_name = img_name if len(img_name) <= 28 else (img_name[:16] + "..." + img_name[-9:])
        self.img_listbox.insert(self.current_idx, f"{self.current_idx+1:03d} [✓] {short_name}")
        self.img_listbox.itemconfig(self.current_idx, fg="#4ade80")
        self.img_listbox.selection_set(self.current_idx)
        self._update_progress_label()

        # Cập nhật luôn file CSV tổng hợp cho cả nhóm
        self.export_master_csv(show_popup=False)

        if not silent:
            self.status_bar.config(text=f"✅ Đã lưu {img_name} ({total} hạt) và cập nhật rice_features_dataset.csv!")

    def save_and_next(self):
        self.save_current_image(silent=True)
        if self.current_idx < len(self.img_paths) - 1:
            self.load_image_at(self.current_idx + 1)
        else:
            messagebox.showinfo("Hoàn tất!", "Bạn đã duyệt xong đến ảnh cuối cùng!\nFile CSV tổng hợp đã được lưu đầy đủ.")

    def export_master_csv(self, show_popup: bool = True):
        """Gom tất cả các file session JSON đã duyệt thành 1 file CSV chuẩn để train Machine Learning."""
        sess_files = sorted(glob.glob(os.path.join(self.out_dir, "sessions", "*.json")))
        csv_path = os.path.join(self.out_dir, "rice_features_dataset.csv")

        total_rows = 0
        total_normal = 0
        total_defect = 0

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for sf in sess_files:
                with open(sf, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                img_name = data.get("image_name", os.path.basename(sf))
                for gid, g in enumerate(data.get("grains", []), start=1):
                    feats = g.get("features", {})
                    lbl_name = g.get("label", "Normal")
                    lbl_id = 0 if lbl_name == "Normal" else 1
                    if lbl_id == 0:
                        total_normal += 1
                    else:
                        total_defect += 1

                    row = {
                        "image_name": img_name,
                        "grain_id": gid,
                        "label_name": lbl_name,
                        "label_id": lbl_id,
                        "source": g.get("source", "auto"),
                    }
                    for col in CSV_COLUMNS[5:]:
                        row[col] = feats.get(col, 0.0)
                    writer.writerow(row)
                    total_rows += 1

        if show_popup:
            messagebox.showinfo(
                "Xuất Dataset CSV thành công!",
                f"Đã tổng hợp từ {len(sess_files)} ảnh đã duyệt:\n"
                f"• Tổng số hạt gạo: {total_rows}\n"
                f"• Normal (0):      {total_normal}\n"
                f"• Defective (1):   {total_defect}\n\n"
                f"Đường dẫn file CSV (dùng để train ML):\n{csv_path}",
            )

    # ------------------------------------------------------------------
    # CÁC THAO TÁC CHỈNH SỬA NHÃN & UNDO
    # ------------------------------------------------------------------
    def _push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.grains))
        if len(self.undo_stack) > 40:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            self.status_bar.config(text="Không có thao tác nào để hoàn tác.")
            return
        self.grains = self.undo_stack.pop()
        self.is_dirty = True
        self._update_stats()
        self.redraw_canvas()
        self.status_bar.config(text="↩ Đã hoàn tác thao tác vừa rồi.")

    def set_all_labels(self, target_label: str):
        if not self.grains:
            return
        self._push_undo()
        for g in self.grains:
            g["label"] = target_label
        self.is_dirty = True
        self._update_stats()
        self.redraw_canvas()
        self.status_bar.config(text=f"Đã đặt nhanh toàn bộ {len(self.grains)} hạt thành [{target_label}]!")

    def _update_stats(self):
        total = len(self.grains)
        n_norm = sum(1 for g in self.grains if g["label"] == "Normal")
        n_def = total - n_norm
        ratio = (n_norm / total * 100.0) if total > 0 else 0.0

        self.lbl_stat_total.config(text=f"Tổng số hạt: {total}")
        self.lbl_stat_normal.config(text=f"● Normal (Xanh): {n_norm}")
        self.lbl_stat_defect.config(text=f"● Defective (Đỏ): {n_def}")
        self.lbl_stat_ratio.config(text=f"Tỷ lệ gạo chuẩn: {ratio:.1f}%")

    # ------------------------------------------------------------------
    # RENDER CANVAS SIÊU MƯỢT (CROP VIEWPORT + VECTOR POLYGONS)
    # ------------------------------------------------------------------
    def fit_to_canvas(self):
        if self.pil_img is None:
            return
        self.root.update_idletasks()
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        iw, ih = self.pil_img.size

        self.scale = min(cw / iw, ch / ih) * 0.96
        self.offset_x = (cw - iw * self.scale) / 2.0
        self.offset_y = (ch - ih * self.scale) / 2.0
        self.redraw_canvas()

    def _img_to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        return ix * self.scale + self.offset_x, iy * self.scale + self.offset_y

    def _canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx - self.offset_x) / self.scale, (cy - self.offset_y) / self.scale

    def redraw_canvas(self):
        self.canvas.delete("all")
        if self.pil_img is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = self.pil_img.size
        if cw <= 1 or ch <= 1:
            return

        # Chỉ crop vùng ảnh đang hiển thị trong khung nhìn để Zoom 5x vẫn siêu mượt
        ix0 = max(0, int(np.floor((0 - self.offset_x) / self.scale)))
        iy0 = max(0, int(np.floor((0 - self.offset_y) / self.scale)))
        ix1 = min(iw, int(np.ceil((cw - self.offset_x) / self.scale)))
        iy1 = min(ih, int(np.ceil((ch - self.offset_y) / self.scale)))

        if ix1 > ix0 and iy1 > iy0:
            crop = self.pil_img.crop((ix0, iy0, ix1, iy1))
            disp_w = max(1, int(round((ix1 - ix0) * self.scale)))
            disp_h = max(1, int(round((iy1 - iy0) * self.scale)))
            resample = Image.NEAREST if self.scale > 2.5 else Image.BILINEAR
            resized = crop.resize((disp_w, disp_h), resample)
            self._photo_ref = ImageTk.PhotoImage(resized)
            dx, dy = self._img_to_canvas(ix0, iy0)
            self.canvas.create_image(dx, dy, anchor=tk.NW, image=self._photo_ref)

        # Vẽ vector đa giác xoay (Rotated Bounding Box) cho từng hạt
        for i, g in enumerate(self.grains):
            pts = cv2.boxPoints(g["rect"])
            coords = []
            for px, py in pts:
                cx, cy = self._img_to_canvas(px, py)
                coords.extend([cx, cy])

            is_norm = (g["label"] == "Normal")
            is_hover = (i == self.hovered_idx)
            outline = "#22c55e" if is_norm else "#ef4444"
            if is_hover:
                outline = "#facc15"  # Highlight màu vàng rực khi rê chuột
            width = 3 if is_hover else 2

            self.canvas.create_polygon(coords, outline=outline, fill="", width=width)

    def _find_grain_at(self, ix: float, iy: float) -> int:
        """Tìm chỉ số hạt gạo chứa điểm (ix, iy); nếu chồng nhẹ thì ưu tiên hạt gần tâm nhất."""
        best_idx = -1
        best_dist = float("inf")
        for i, g in enumerate(self.grains):
            pts = cv2.boxPoints(g["rect"])
            if cv2.pointPolygonTest(pts, (float(ix), float(iy)), False) >= 0:
                cx, cy = g["rect"][0]
                d = (cx - ix) ** 2 + (cy - iy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    # SỰ KIỆN CHUỘT (ZOOM / PAN / CLICK ĐỔI NHÃN / SPLIT / VẼ THÊM / XÓA)
    # ------------------------------------------------------------------
    def _on_space_down(self, event):
        self._space_down = True
        self.canvas.config(cursor="fleur")

    def _on_space_up(self, event):
        self._space_down = False
        self._panning = False
        self.canvas.config(cursor="crosshair")

    def _on_mouse_wheel(self, event):
        if self.pil_img is None:
            return
        factor = 1.18 if event.delta > 0 else (1.0 / 1.18)
        new_scale = max(0.15, min(12.0, self.scale * factor))
        # Giữ cố định điểm ảnh nằm dưới con trỏ chuột khi zoom
        ix, iy = self._canvas_to_img(event.x, event.y)
        self.scale = new_scale
        self.offset_x = event.x - ix * self.scale
        self.offset_y = event.y - iy * self.scale
        self.redraw_canvas()

    def _on_pan_start(self, event):
        self._panning = True
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if not self._panning:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self._pan_start = (event.x, event.y)
        self.redraw_canvas()

    def _on_pan_end(self, event):
        self._panning = False

    def _on_mouse_move(self, event):
        if self.pil_img is None or self._panning:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        hit = self._find_grain_at(ix, iy)
        if hit != self.hovered_idx:
            self.hovered_idx = hit
            self.redraw_canvas()
            if hit >= 0:
                g = self.grains[hit]
                f = g["features"]
                self.lbl_hover_info.config(
                    text=(
                        f"Hạt #{hit+1} [{g['label']}] ({g.get('source', 'auto')})\n"
                        f"• Area:        {f['area']:.1f} px\n"
                        f"• Length (L):  {f['length']:.1f} px\n"
                        f"• Width  (W):  {f['width']:.1f} px\n"
                        f"• Aspect L/W:  {f['aspect_ratio']:.2f}\n"
                        f"• Circularity: {f['circularity']:.2f}\n"
                        f"• Solidity:    {f['solidity']:.2f}\n"
                        f"• RGB Mean:    ({f['mean_r']:.0f}, {f['mean_g']:.0f}, {f['mean_b']:.0f})\n"
                        f"• LAB Mean:    ({f['mean_L']:.0f}, {f['mean_a']:.0f}, {f['mean_b_lab']:.0f})"
                    )
                )
            else:
                self.lbl_hover_info.config(text="Rê chuột vào một hộp gạo\nđể xem chi tiết đặc trưng ML...")

    def _on_left_down(self, event):
        if self.pil_img is None:
            return
        if self._space_down:
            self._on_pan_start(event)
            return

        ix, iy = self._canvas_to_img(event.x, event.y)
        mode = self.tool_mode.get()
        hit = self._find_grain_at(ix, iy)

        if mode == "draw" or (mode == "toggle" and hit == -1):
            # Nếu đang ở chế độ Vẽ hoặc bấm vào vùng trống -> bắt đầu kéo vẽ thêm hộp mới
            self._draw_start_img = (ix, iy)
            self._draw_rect_id = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="#38bdf8", width=2, dash=(4, 2)
            )
            return

        if hit == -1:
            return

        if mode == "toggle":
            self._push_undo()
            cur = self.grains[hit]["label"]
            self.grains[hit]["label"] = "Defective" if cur == "Normal" else "Normal"
            self.is_dirty = True
            self._update_stats()
            self.redraw_canvas()

        elif mode == "delete":
            self._push_undo()
            self.grains.pop(hit)
            self.hovered_idx = -1
            self.is_dirty = True
            self._update_stats()
            self.redraw_canvas()

        elif mode in ("split_w", "split_l"):
            self._push_undo()
            g = self.grains.pop(hit)
            r1, r2 = split_rotated_rect(g["rect"], along_width=(mode == "split_w"))
            for r in (r1, r2):
                feats = extract_features_from_roi(self.src_bgr, self.src_lab, r, contour=None, global_mask=self.global_otsu_mask)
                self.grains.append({
                    "rect": r,
                    "label": g["label"],
                    "features": feats,
                    "source": "manual_split",
                })
            self.hovered_idx = -1
            self.is_dirty = True
            self._update_stats()
            self.redraw_canvas()
            self.status_bar.config(text="✂️ Đã chẻ đôi hạt dính thành 2 hộp độc lập!")

    def _on_left_drag(self, event):
        if self._panning:
            self._on_pan_drag(event)
            return
        if self._draw_start_img is not None and self._draw_rect_id is not None:
            sx, sy = self._img_to_canvas(*self._draw_start_img)
            self.canvas.coords(self._draw_rect_id, sx, sy, event.x, event.y)

    def _on_left_up(self, event):
        if self._panning:
            self._on_pan_end(event)
            return
        if self._draw_start_img is not None:
            ix0, iy0 = self._draw_start_img
            ix1, iy1 = self._canvas_to_img(event.x, event.y)
            self._draw_start_img = None
            if self._draw_rect_id is not None:
                self.canvas.delete(self._draw_rect_id)
                self._draw_rect_id = None

            w = abs(ix1 - ix0)
            h = abs(iy1 - iy0)
            if w >= 8 and h >= 8:
                cx = (ix0 + ix1) / 2.0
                cy = (iy0 + iy1) / 2.0
                rect = ((cx, cy), (w, h), 0.0)
                feats = extract_features_from_roi(self.src_bgr, self.src_lab, rect, contour=None, global_mask=self.global_otsu_mask)
                lbl = "Normal" if (feats["aspect_ratio"] >= self.var_norm_ar.get() and feats["area"] >= self.var_norm_area.get()) else "Defective"
                self._push_undo()
                self.grains.append({
                    "rect": rect,
                    "label": lbl,
                    "features": feats,
                    "source": "manual_draw",
                })
                self.is_dirty = True
                self._update_stats()
                self.redraw_canvas()
                self.status_bar.config(text=f"✏️ Đã vẽ thêm 1 hạt gạo mới ({lbl}) và tự động trích xuất đặc trưng!")

    def _on_right_click(self, event):
        """Click chuột phải bất cứ lúc nào để XÓA NHANH hộp rác/nhiễu."""
        if self.pil_img is None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        hit = self._find_grain_at(ix, iy)
        if hit >= 0:
            self._push_undo()
            self.grains.pop(hit)
            self.hovered_idx = -1
            self.is_dirty = True
            self._update_stats()
            self.redraw_canvas()
            self.status_bar.config(text="🗑 Đã xóa 1 hộp rác (Ctrl+Z để hoàn tác).")

    # ------------------------------------------------------------------
    # CỬA SỔ POPUP XEM 4 BƯỚC XỬ LÝ ẢNH (ĐỂ DEBUG & BẢO VỆ ĐỒ ÁN)
    # ------------------------------------------------------------------
    def show_dip_steps_popup(self):
        if not self.debug_imgs:
            return
        win = tk.Toplevel(self.root)
        win.title("Trực quan hóa 4 bước Xử lý ảnh thuần túy (DIP Pipeline)")
        win.configure(bg="#181825")
        win.geometry("1040x760")

        titles = [
            ("step1_enhanced", "Bước 1: Grayscale / Top-Hat"),
            ("step2_otsu", "Bước 2: Otsu + Opening (Mask nhị phân)"),
            ("step3_distance", "Bước 3: Distance Transform"),
            ("step4_watershed", "Bước 4: Watershed (Đường ranh giới đỏ)"),
        ]

        win._img_refs = []
        for idx, (key, title) in enumerate(titles):
            r, c = divmod(idx, 2)
            frame = tk.Frame(win, bg="#262637", padx=6, pady=6)
            frame.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            win.grid_rowconfigure(r, weight=1)
            win.grid_columnconfigure(c, weight=1)

            tk.Label(frame, text=title, bg="#262637", fg="#93c5fd", font=("Segoe UI", 10, "bold")).pack(pady=(0, 4))
            img = self.debug_imgs[key]
            if img.ndim == 2:
                pil_i = Image.fromarray(img)
            else:
                pil_i = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_i.thumbnail((480, 320), Image.BILINEAR)
            photo = ImageTk.PhotoImage(pil_i)
            win._img_refs.append(photo)
            tk.Label(frame, image=photo, bg="#11111b").pack(expand=True)


def main():
    root = tk.Tk()
    app = RiceSemiAutoLabelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

