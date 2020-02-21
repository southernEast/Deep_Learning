#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/23 14:25
# @Author  : shadow
# @Site    : 
# @File    : tk.py
# @Software: PyCharm

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from keras.layers import Flatten, BatchNormalization, PReLU, Dense, Dropout
from keras.models import Model, Input
from keras.applications.resnet50 import ResNet50
import numpy as np
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import time
import os
import tkinter.font as tkFont

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


# 弹窗
class PopupDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__()
        self.title('输出结果')

        self.parent = parent  # 显式地保留父窗口

        sw = self.parent.winfo_screenwidth()
        sh = self.parent.winfo_screenheight()
        ww = 500
        wh = 200
        x = (sw-ww) // 2
        y = (sh-wh) // 2
        self.geometry("%dx%d+%d+%d"%(ww, wh, x, y))

        ft = tkFont.Font(family='宋体', size=23)
        row1 = tk.Frame(self)
        row1.pack(anchor='s', ipady=25)
        tk.Label(row1, text=str(self.parent.ans_key), font = ft).pack()


        row2 = tk.Frame(self)
        row2.pack(fill="x")
        tk.Button(row2, text="确认", font = ft, command=self.cancel).pack()


    def cancel(self):
        self.destroy()


# 主窗
class MyApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('基于深度学习的书法字体识别')

        # 程序参数
        self.string = tk.StringVar()
        self.ans_key = []
        # 程序界面
        self.setupUI()

    def setupUI(self):
        # 初始化字体
        ft1 = tkFont.Font(family='宋体', size=17)         # 地址栏字体
        ft2 = tkFont.Font(family='宋体', size=20)         # 按钮字体

        row1 = tk.Frame(self)
        row1.pack(fill="x")             # x方向填充
        tk.Entry(row1, font=ft1, width=68, textvariable=self.string).pack()     # 选择文件地址栏

        # 三个按钮
        row2 = tk.Frame(self)
        row2.pack(fill="x")
        tk.Button(row2, text="选择文件", width=10, font=ft2, command = self.choose_file).pack()

        row3 = tk.Frame(self)
        row3.pack(fill="x")
        tk.Button(row3, text="上传", width=10, font=ft2, command = self.upload_file).pack()

        row4 = tk.Frame(self)
        row4.pack(fill="x")
        tk.Button(row4, text="开始识别", width=10, font=ft2, command = self.start).pack()

        # 背景图片
        row5 = tk.Frame(self)
        row5.pack(fill="x")
        global img
        img = Image.open("D:/Deep_learning_design/汉字/毕设重要版本/UI/back.jpg")
        (x, y) = img.size
        x //= 2
        y //= 2
        img = img.resize((x, y))
        img = ImageTk.PhotoImage(img)
        lab = tk.Label(row5, image=img)
        lab.pack()


  	# 选择文件
    def choose_file(self):
        selectFileName = tk.filedialog.askopenfilename(title = "选择文件")
        self.string.set(selectFileName)


    # 上传文件
    def upload_file(self):
        img = Image.open(self.string.get())
        plt.imshow(img)
        path = "D:/Deep_learning_design/汉字/毕设重要版本/image_test"
        name = time.time()
        img.save(path + str(name) + ".jpg")
        plt.show()


  	# 开始识别
    def start(self):
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
        self.ans_key = ans_key
        self.output_ans()


    # 输出结果
    def output_ans(self):
        pw = PopupDialog(self)
        self.wait_window(pw)

        return


if __name__ == '__main__':
    """
           初始化权重
    """
    weight_path = 'best_weights_Alex.h5'
    image_path = 'D:/Deep_learning_design/汉字/毕设重要版本/image_test'
    train_path = "D:/Deep_learning_design/TMD_data/myTrain/train_data"
    model = Alex_model(100)
    model.load_weights(weight_path)
    class_indices = label_of_directory(train_path)  # 获取训练集中标签对应汉字序列

    """
    		界面加载
    """
    app = MyApp()
    app.mainloop()