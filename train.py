#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/2 11:17
# @Author  : shadow
# @Site    : 
# @File    : train.py
# @Software: PyCharm
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Embedding
from keras.layers import Flatten, BatchNormalization, PReLU, Dense, Dropout, Lambda
from keras.models import Model, Input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop
from PIL import Image
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def bn_relu(x):

    """
    Batch Norm正则化
    批量标准层 标准化前一层的激活项
    维持激活项平均值接近0 标准差接近1的转换
    """
    x = BatchNormalization()(x)
    #参数化的ReLU
    x = PReLU()(x)
    return x


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


def my_model(out_dims, input_shape=(128, 128, 1)):
    input_dim = Input(input_shape)                     # 生成一个input_shape的张量

    """
    Conv2D(filters, kernel_size, strides=(1,1), padding='valid')
    滤波器数量，卷积宽度和高度，沿宽度和高度方向步长，padding方法
    返回生成的张量
    """
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input_dim)
    x = bn_relu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    """
    MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_dormat=None)
    沿（垂直，水平）方向缩小比例，步长值 默认pool_size，padding算法，默认channels_last 可选channels_first
    channels_last:(batch_size, rows, cols, channels) channels_first:(batch_size, channels, rows, cols)
    对空间数据最大池化
    """

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
    """
    AveragePooling2D(poolsize=(2,2), strides=None, padding='vaild', data_format=None)
    对空间数据进行平均池化
    """

    """
    Flatten(data_format=None)
    默认channels_last 可选channels_first
    将输入展平 不影响批量大小
    """
    x_flat = Flatten()(x)

    """
    Dense(units, activaction=None, use_bias=True, kernel_initializer='glorot_uniform')
    输出空间维度，激活函数默认a(x)=x，是否使用偏置向量，kernel权值矩阵初始化器 
    全连接层 output = activation(dot(input, kernel) + bias)
    常见输入(batch_size, input_dim) 输出(batch_size, units)
    """
    fc1 = Dense(512)(x_flat)
    fc1 = bn_relu(fc1)
    dp_1 = Dropout(0.3)(fc1)
    """"
    Dropout(rate, noise_shape=None, seed=None)
    rate：在0和1之间浮动 表示需要丢弃的输入比例
    防止过拟合 随机失活
    """

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('softmax')(fc2)
    """
    Activation(activation)
    activation：需要使用的激活函数名称 softmax relu tanh sigmoid linear PreLU LeakyReLI
    输出尺寸与输入尺寸相同
    """

    """
    Model(inputs, outputs)
    模型将计算从inputs到outputs所有网络层
    """
    model = Model(inputs=input_dim, outputs=fc2)
    return model


# 动态调整学习率
def lrschedule(epoch):
    if epoch > 0.9 * max_epochs:
        return 0.00001
    elif epoch > 0.75 * max_epochs:
        return 0.0001
    elif epoch > 0.5 * max_epochs:
        return 0.0005
    else:
        return 0.005
    # return 0.0005



def model_train(model, loadweights):
    lr = LearningRateScheduler(lrschedule)
    """
    LearningRateScheduler(schedule, verbose=0)
    schedule：一个函数，接受轮索引数作为输入（整数，从0开始迭代）
    verbose：整数 0：安静 1：更新信息
    动态设置学习率
    """
    """
    ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, 
    mode='auto',period=1)
    filepath：字符串，保存模型的路径
    monitor：被监测的数据
    verbose：详细信息模式
    save_best_only：True，只保存被监测数据的最佳模型
    mode：[auto,min,max]可选 val_loss->min val_acc->max
    save_weights_only:True，只保存权重，否则保存整个模型
    period：每个检查点之间间隔（训练轮数）
    每个训练期后保存模型
    """
    mdcheck = ModelCheckpoint(weight_path, monitor='val_acc', save_best_only=True)
    td = TensorBoard(log_dir='/home/shallow/PycharmProjects/MyDesign/temsorboard_log/')
    """
    log_dir：用来保存被TensorBoard分析的日志文件的文件名
    为Tensorboard编写一个日志，可以可视化测试和训练的标准评估动态图像，
    也可以可视化模型中不同层的激活值直方图
    """

    # 加载权重等信息
    if loadweights:
        if os.path.isfile(weight_path):             # 判断文件是否存在
            model.load_weights(weight_path)
            print("model load pre weights!")
        else:
            print("model didn't load weights!")
    else:
        print("not load weights")

    """
    SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    学习率，用于加速SGD相关方向前进，学习率衰减值，是否使用Nesterov动量
    随机梯度下降优化器
    """
    # rmsprot = RMSprop(lr=0.001)
    sgd = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    # adam = Adam(lr=0.001)
    print("model compile")
    """
    compile(optimizer, loss=None, metrics=None, loss_weights=None)
    optimizer：优化器名或实例
    loss：目标函数，categorical_crossentropy 目标值是分类格式
    metrics：训练和测试期间的模型评估标准，通常使用['accuracy']
    """
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print("model training")
    """
    fitgenerator(train_generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
    validation_data=None, validation_steps=None)
    generator：一个生成器
    steps_per_epoch：声明一个epoch完成并开始下一个epoch之前从generator产生的总步数
    epochs：训练模型的迭代总轮数
    validation_data：验证数据生成器
    validation_steps：样本批数
    callbacks：实例列表
    返回验证集损失和评估集的记录
    """
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=32000 // batch_size,
                                  epochs=max_epochs,
                                  validation_data=val_generator,
                                  validation_steps=8000 // batch_size,
                                  callbacks=[lr, mdcheck, td])

    return history


def draw_loss_acc(history):
    x_trick = [x + 1 for x in range(max_epochs)]        # 存储1->max_epoch数字
    loss = history.history['loss']                      # 获取history中的数据
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']

    # 使用plt自带样式美化
    plt.style.use('ggplot')

    # 显示loss和val_loss的图像
    plt.figure(figsize=(10, 6))                         # 调整图像尺寸
    plt.title('model = %s, batch_size = %s' % ('losses', batch_size))
    plt.plot(x_trick, loss, 'g-', label='loss')         # 标签
    plt.plot(x_trick, val_loss, 'y-', label='val_loss')
    plt.legend()                                        # 放置标签
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('image/loss.png', format='png', dpi=300)# 先保存图片 dpi设置图像分辨率
    plt.show()

    # 显示acc和val_acc的图像
    plt.figure(figsize=(10, 6))
    plt.title('learninngRate = %s, batch_size = %s' % ('accuracy', batch_size))
    plt.plot(x_trick, val_acc, 'y-', label='val_acc')
    plt.plot(x_trick, acc, 'b-', label='acc')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.savefig('image/acc.png', format='png', dpi=300)
    plt.show()


# 将数据集中的图片地址存入list
def generator_list_of_imagepath(path):
    image_list = []
    for image in os.listdir(path):
        image_list.append(path + image)
    return image_list


# 训练集中汉字对应序号信息
def label_of_directory(directory):
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)

    # 将汉字与序号对应并压缩
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices


# 获取top_k个可能性最高的标签
def get_label_predict_top_k(image, model, top_k):
    predict_proprely = model.predict(image)         # 预测标签值
    predict_list = list(predict_proprely[0])        # 预测值转换成list
    min_label = min(predict_list)                   # 取可能性最小的值
    label_k = []
    for i in range(top_k):
        label = np.argmax(predict_list)             # 获取可能性最大的标签号
        predict_list.remove(predict_list[label])    # 删除当前标签
        predict_list.insert(label, min_label)       # 插入可能性最小的
        label_k.append(label)
    return label_k


# 从序号取得对应的汉字
def get_key_from_value(dict, index):
    for keys, values in dict.items():
        if values == index:
            return keys


# 加载图片
def load_image(image):
    img = Image.open(image)
    img = img.resize((128,128))
    img = np.array(img)
    img = img / 255
    img = img.reshape((1,) + img.shape + (1,))
    return img


def test_predict(model, test_path, directory, top_k = 5):
    model.load_weights(weight_path)
    image_list = generator_list_of_imagepath(test_path)     # 获取的图片地址信息

    predict_label = []
    class_indecs = label_of_directory(directory)
    for image in image_list:
        img = load_image(image)                             # 加载图片
        label_index = get_label_predict_top_k(img, model, top_k)    # 获取可能性最高的top_k个标签
        label_value_dict = []
        for label in label_index:                                   # 将可能性最高的top_k个标签对应的汉字存储
            label_value = get_key_from_value(class_indecs, label)
            label_value_dict.append(str(label_value))
        predict_label.append(label_value_dict)

    return predict_label


def train_list2str(predict_list_label):
    new_label = []
    for row in range(len(predict_list_label)):
        str = ""
        for label in predict_list_label[row]:
            str += label
        new_label.append(str)
    return new_label


def save_csv(test_path, predict_label):
    image_list = generator_list_of_imagepath(test_path)
    save_arr = np.empty((16343, 2), dtype=np.str)
    save_arr = pd.DataFrame(save_arr, columns=['filename', 'label'])
    predict_label = train_list2str(predict_label)
    for i in range(len(image_list)):
        filename = image_list[i].split('/')[-1]
        save_arr.values[i, 0] = filename
        save_arr.values[i, 1] = predict_label[i]
    save_arr.to_csv('submit_test.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
    print("The location of submit_test.csv is :", os.getcwd())



if __name__ == "__main__":
    train_path = "/home/shallow/TMD_data/myTrain/train_data/" # 训练集路径
    val_path = "/home/shallow/TMD_data/myTrain/train_val/"    # 验证集路径
    test_path = "/home/shallow/TMD_data/test2(1)/test2/"       # 测试集路径
    num_calsses = 100                                         # 分类种类
    batch_size = 128
    weight_path = 'best_weights_Alex.h5'                           # 存储最佳权重
    max_epochs = 40

    # 数据集增广
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,                  # 缩放
        horizontal_flip=True,
        rotation_range=20,
        zoom_range = 0.4
    )
    val_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    # 以文件夹路径为参数 生成数据集增强/归一化后的数据
    train_generator = train_datagen.flow_from_directory(
        train_path,                 # 目标路径
        target_size=(128, 128),     # 修改尺寸
        batch_size=batch_size,      # batch数据大小
        color_mode='grayscale',     # 转换成单通道颜色模式 默认"rgb"
        class_mode='categorical'    # 返回2D的one-hot编码标签 在model.predict_generator() or model.evalute_generator()使用
    )
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(128, 128),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )

    model = Alex_model(num_calsses)   # 产生模型
    print(model.summary())          # 输出模型各层的参数状况

    # 训练开始
    print("****start train image of epoch****")
    model_history = model_train(model, False)

    # 输出并保存当前acc和loss图片
    print("****show acc and loss of train and val****")
    draw_loss_acc(model_history)

    print("****test label****")
    model.load_weights(weight_path)
    predict_label = test_predict(model, test_path, train_path, 5)

    print("****csv save****")
    save_csv(test_path, predict_label)