# from keras import layers
from keras.layers import *
from keras.models import *


def lenet(input_shape=(96, 96, 3), num_classes=3):
    inputs = Input(input_shape)
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    # Block 2
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    # Block 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    if num_classes > 2:  # 最後一層分類層，根據要分類的數目決定隱藏層數
        x = Dense(num_classes, activation='softmax')(x)
    elif num_classes == 2 or num_classes == 1:
        x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    return model
