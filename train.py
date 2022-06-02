import os
from datetime import datetime
from nets.lenet import lenet
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator


def main():
    input_shape = (512, 512, 3)
    num_classes = 4
    epochs = 10
    batch_size = 3

    (img_width, img_height, channel) = input_shape
    train_data_dir = f'./database/train_data'  # 訓練資料的資料夾
    validation_data_dir = f'./database/val_data'  # 驗證資料的資料夾
    # 訓練資料 (多次輸入有加一點旋轉變數)
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    # 驗證資料 (僅做正規化)
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # 建立生成器
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='sparse')  # class_mode決定回傳的標籤數組類型"binary" 是 1D 二進制標籤，"sparse" 是 1D 整數標籤
    # 將驗證資料壓成生成器，提供給模型訓練使用
    validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='sparse')

    model = lenet(input_shape, num_classes)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # model.load_weights("./model_weight/weights-improvement-ep09-val_loss1.288.h5")
    model_save_filepath = "./model_weight/weights-improvement-ep{epoch:02d}-val_loss{val_loss:.3f}.h5"
    # 使用檢查點(checkpoint)保存最好的模型，monitor為檢測方式預設為val_loss
    checkpoint = ModelCheckpoint(model_save_filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(
        train_generator,
        steps_per_epoch=24 // 3,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=12 // 3,
        callbacks=callbacks_list)


if __name__ == "__main__":
    main()
