import os
import cv2
import numpy as np
from PIL import Image


class ImgTools:
    @staticmethod
    def read_img(img_path):
        org_img = Image.open(img_path).convert("L")  # 以PIL開啟圖片，避開cv2無法讀取中文的問題
        org_img = np.uint8(org_img)  # 轉cv2形式
        img = np.copy(org_img)  # 複製一份圖片做圖像處理
        return org_img, img

    @staticmethod
    def pil_save_img(new_img, img_path):
        file_name = os.path.basename(img_path)  # 提取出檔名.類型
        file_name = os.path.splitext(file_name)[0]  # 分割出檔名
        save_file_name = f"{file_name}_remove_noise.tif"  # 輸出檔名添加後綴_remove_noise

        _, new_img = cv2.threshold(new_img, 127, 255, cv2.THRESH_BINARY)  # 圖像二值化
        bool_img = np.bool_(new_img)  # 轉成True False陣列
        bool_img = Image.fromarray(bool_img)  # 轉成PIL圖片格式
        bool_img.save(save_file_name, dpi=(600, 600), compression="group4")  # 設定dpi，並以group4壓縮儲存
        return True

    @staticmethod
    def cv2_save_img(new_img, img_path):
        file_name = os.path.basename(img_path)  # 提取出檔名.類型
        file_name = os.path.splitext(file_name)[0]  # 分割出檔名
        save_file_name = f"{file_name}_remove_noise.tif"  # 輸出檔名添加後綴_remove_noise
        cv2.imencode('.tif', new_img, [int(cv2.IMWRITE_TIFF_RESUNIT), 2,  # 解析度單位
                                       int(cv2.IMWRITE_TIFF_COMPRESSION), 5,  # 壓縮方式lzw
                                       int(cv2.IMWRITE_TIFF_XDPI), 600,  # 設定水平dpi
                                       int(cv2.IMWRITE_TIFF_YDPI), 600,  # 設定垂直dpi
                                       ])[1].tofile(save_file_name)