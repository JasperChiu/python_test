import os
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from nets.lenet import lenet


def img_convert_to_model_input(img_path, input_shape):
    """
    將原圖轉換成輸入模型的陣列
    :param img_path: 原始圖片路徑
    :param input_shape: 轉換輸入模型的大小
    :return: np.array陣列型態的圖片資料
    """
    input_img_size = input_shape[:2]  # 若輸入為三維，取前兩位(96, 96)
    channel = input_shape[2]  # 取出通道數
    if channel == 1:  # 若指定輸入模型的通道數為單通道，以tf將其轉成(96, 96, 1)
        img = Image.open(img_path).convert("L")  # 路徑開啟為灰階
        img = np.uint8(img)
        img = cv2.resize(img, input_img_size)
        img = img[:, :, np.newaxis]  # 將其轉成(96, 96, 1) new axis新增一個維度
        # img = tf.image.rgb_to_grayscale(img)  等校上面的作法 但只能RGB轉灰階
        img = np.array(img) / 255.
        return img
    else:
        img = Image.open(img_path).convert("RGB")  # 路徑開啟為灰階
        img = np.uint8(img)
        img = cv2.resize(img, input_img_size)
        img = np.array(img) / 255.
    return img


def main():
    weights_path = f"./model_weight/weights-improvement-ep09-val_loss1.288.h5"
    input_shape = (512, 512, 3)
    num_classes = 4
    model = lenet(input_shape, num_classes)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.load_weights(weights_path)

    folder_path = "./database/test_data/"
    files = os.listdir(folder_path)  # 讀取資料夾內所有資料
    img_files = []
    for file in files:
        if '.png' in file or '.tif' in file:  # 檢測若file檔名包含.tif 則加入要合併的清單
            img_files.append(folder_path + file)  # 各檔案完整路徑

    test_list = []
    for img_path in img_files:  # 讀清單內資料，讀出來轉成訓練用的np.array格式
        convert_img = img_convert_to_model_input(img_path, input_shape)
        test_list.append(convert_img)

    test_list = np.array(test_list)
    print(test_list.shape)  # 印出輸入模型的陣列形狀
    predict_img = model.predict(test_list)

    one_list = np.argmax(predict_img, axis=1)
    # print(one_list)  # 印出預測清單
    zipped = zip(one_list, files)
    predict_result = {0: "car", 1: "suv", 2: "truck", 3: "bus"}
    for result, file in zipped:
        print(f"{file} 預測結果為 {predict_result[result]}")


if __name__ == "__main__":
    main()
