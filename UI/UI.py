#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from keras.layers import Flatten, BatchNormalization, PReLU, Dense, Dropout
from keras.models import Model, Input
from keras.applications.resnet50 import ResNet50
import numpy as np
import tkinter
import tkinter.filedialog
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time


def bn_relu(x):

    x = BatchNormalization()(x)
    #参数化的ReLU
    x = PReLU()(x)
    return x

def Alex_model(out_dims, input_shape=(128, 128, 1)):
    input_dim = Input(input_shape)

    x = Conv2D(96, (20, 20), strides=(2, 2), padding='valid')(input_dim)        # 55 * 55 * 96
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)      # 27 * 27 * 96
    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same')(x)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = Flatten()(x)
    fc1 = Dense(4096)(x)
    dr1 = Dropout(0.2)(fc1)
    fc2 = Dense(4096)(dr1)
    dr2 = Dropout(0.25)(fc2)
    fc3 = Dense(out_dims)(dr2)

    fc3 = Activation('softmax')(fc3)

    model = Model(inputs=input_dim, outputs=fc3)
    return model

def resner50(out_dims, input_shape=(128, 128, 1)):
    # input_dim = Input(input_shape)
    resnet_base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)

    x = resnet_base_model.output
    x = Flatten()(x)
    fc = Dense(512)(x)
    x = bn_relu(fc)
    x = Dropout(0.5)(x)
    x = Dense(out_dims)(x)
    x = Activation("softmax")(x)

    # buid myself model
    input_shape = resnet_base_model.input
    output_shape = x

    resnet50_100_model = Model(inputs=input_shape, outputs=output_shape)

    return resnet50_100_model


def my_model(out_dims, input_shape=(128, 128, 1)):
    input_dim = Input(input_shape)                     # 生成一个input_shape的张量

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input_dim)
    x = bn_relu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_relu(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_relu(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x_flat = Flatten()(x)

    fc1 = Dense(512)(x_flat)
    fc1 = bn_relu(fc1)
    dp_1 = Dropout(0.3)(fc1)

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('softmax')(fc2)

    model = Model(inputs=input_dim, outputs=fc2)
    return model


def load_image(image):
    img = Image.open(image).convert('L')
    img = img.resize((128,128))
    img = np.array(img)
    img = img / 255
    img = img.reshape((1,) + img.shape + (1,))
    return img

def get_label(image, model, top_k):
    prediction = model.predict(image)
    predict_list = list(prediction[0])
    min_label = min(predict_list)
    label_k = []
    for i in range(top_k):
        label = np.argmax(predict_list)
        predict_list.remove(predict_list[label])
        predict_list.insert(label, min_label)
        label_k.append(label)
    return label_k


def label_of_directory(directory):
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)

    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices


def get_key_from_classes(dict, index):
    for key, value in dict.items():
        if value == index:
            return key


def choose_file():
    selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
    e.set(selectFileName)



def upload_file(f):
    img = Image.open(f)
    plt.imshow(img)
    path = "/home/shallow/PycharmProjects/MyDesign/image_test/"
    name = time.time()
    img.save(path + str(name) + ".jpg")
    plt.show()


def start():
    img_name = os.listdir(image_path)
    img_list = []
    for name in img_name:
        img_list.append(os.path.join(image_path, name))

    img_list.sort()
    ans_key = []
    for img in img_list:
        image = load_image(img)
        temp_label = get_label(image, model, 5)
        key = get_key_from_classes(class_indices, temp_label[0])
        ans_key.append(key)
    tkinter.messagebox.showinfo(title="识别结果", message="您提交图片的识别汉字为：" + str(ans_key))


if __name__ == "__main__":
    """
        初始化权重
    """
    weight_path = 'best_weights_Alex.h5'
    image_path = '/home/shallow/PycharmProjects/MyDesign/image_test/'
    train_path = "/home/shallow/TMD_data/myTrain/train_data/"
    model = Alex_model(100)
    model.load_weights(weight_path)
    class_indices = label_of_directory(train_path)          # 获取训练集中标签对应汉字序列

    """
        窗口初始化
    """
    top = tkinter.Tk()
    top.title('基于深度学习的汉字书法字体识别')
    # top.geometry('1280x800')

    e = tkinter.StringVar()                                 # 可变字符型
    e_entry = tkinter.Entry(top, font = (18), width=68, textvariable=e)  # 文本框
    e_entry.pack()                                          # 布局


    submit_button = tkinter.Button(top, text ="选择文件", width = 10, font = (15), command = choose_file)
    submit_button.pack()
    submit_button = tkinter.Button(top, text ="上传", width = 10, font= (15), command = lambda:upload_file(e_entry.get()))
    submit_button.pack()
    submit_button = tkinter.Button(top, text="开始识别", width = 10, font=(15), command = start)
    submit_button.pack()


    img = Image.open("/home/shallow/PycharmProjects/MyDesign/UI/back.jpg")
    (x, y) = img.size
    x //= 4
    y //= 4
    img = img.resize((x, y))
    img = ImageTk.PhotoImage(img)
    backLabel = tkinter.Label(top, image=img)
    backLabel.pack()
    top.mainloop()
