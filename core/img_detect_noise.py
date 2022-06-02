import cv2


class ImgDetectNoise:
    @staticmethod
    def rt_keep_list(img, contours, min_area, indent):
        """
        根據設定的條件，回傳要保留的清單
        :param img:欲處理的圖像(用於提供邊界)
        :param contours:輪廓資料
        :param min_area:最小面積閾值
        :param indent:邊界內縮條件
        :return:要保留的輪廓編號清單
        """
        remove_list = []  # 將要清除的輪廓編號加入移除清單
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])  # 計算輪廓面積
            M = cv2.moments(contours[i])  # cv2計算圖像不變矩
            centroid_x = int(M["m10"] / M["m00"])  # 計算X方向質心
            centroid_y = int(M["m01"] / M["m00"])  # 計算Y方向質心
            # 若面積小於一定閾值，則加入清除清單
            if area < min_area:
                remove_list.append(i)
            # 設定四周邊界內縮距離
            if centroid_x < indent or centroid_x > (img.shape[1] - indent) or\
                    centroid_y < indent or centroid_y > (img.shape[0] - indent):
                remove_list.append(i)
        contours_list = list(range(len(contours)))  # 建立contours的輪廓清單
        # contours_list和要清除的清單remove_list取差集，便可以獲得要保留的清單
        keep_list = set(contours_list).difference(set(remove_list))
        return keep_list
