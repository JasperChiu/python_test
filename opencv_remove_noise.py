import os
from utils.img_tools import ImgTools
from core.img_preprocess import ImgPreprocess
from core.img_detect_noise import ImgDetectNoise
from core.img_postprocess import ImgPostprocess


def main(img_path, dilate_iter, min_area, indent):
    # 讀入圖片
    org_img, img = ImgTools.read_img(img_path)
    # 圖像前處理 - 捕捉輪廓
    img, contours = ImgPreprocess.detect_contours(img, dilate_iter)
    # 圖像處理 - 檢測雜訊，設定最小面積閾值(min_area)、邊界內縮(indent)
    keep_list = ImgDetectNoise.rt_keep_list(img, contours, min_area, indent)
    # 圖像後處理 - 根據要保留的清單，將原始圖片內的資訊保留下來，生成新的圖片
    new_img = ImgPostprocess.masking_org_img(org_img, contours, keep_list)
    # 儲存圖片
    ImgTools.pil_save_img(new_img, img_path)


if __name__ == "__main__":
    img_path = "./(掃描檔)公孫龍與公孫龍子內文_結果15.jpg"
    dilate_iter = 1
    min_area = 100
    indent = 200
    main(img_path, dilate_iter, min_area, indent)
