"""
File khởi chạy nhanh giao diện Semi-Auto Labeling cho đề tài hạt gạo.
Chạy lệnh: python run_labeler.py
"""
import os
import sys

# Thêm thư mục semi_auto_labeler vào đường dẫn import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "semi_auto_labeler"))

from rice_labeler_gui import main

if __name__ == "__main__":
    main()

