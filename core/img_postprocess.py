import numpy as np
import cv2


class ImgPostprocess:
    @staticmethod
    def masking_org_img(org_img, contours, keep_list):
        """
        原始圖像與遮罩合併產生新的圖像
        :param org_img: 原始圖像
        :param contours: 輪廓資料
        :param keep_list: 要保留的輪廓編號清單
        :return: 新的圖像
        """
        mask = np.full(org_img.shape, 255, dtype=np.uint8)  # 建立全白的蒙版
        for i in keep_list:  # 根據keep_list來保留需要的資訊
            mask = cv2.drawContours(mask, contours, i, (0, 0, 0), -1)  # 將要保留的區塊在蒙版上塗黑
        new_img = cv2.add(org_img, mask)  # 原圖與蒙版疊加產生新的圖像
        return new_img
