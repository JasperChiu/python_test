import numpy as np
import cv2


class ImgPreprocess:
    @staticmethod
    def detect_contours(img, dilate_iter):
        img = cv2.medianBlur(img, 3)  # 初步灰階模糊化過濾極小雜訊
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # 圖像二值化
        kernel = np.ones((3, 5), np.uint8)  # 濾波器參數
        thresh = cv2.dilate(thresh, kernel, iterations=dilate_iter)  # dilate_iter 膨脹迭代次數
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 抓取輪廓邊緣
        return img, contours
