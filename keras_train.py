#coding:utf-8
import keras
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
#
import numpy as np
import cv2
import time
import random
import os
import sys
import glob
import argparse  #这个模块是命令行参数传入，在nb中不需要
import matplotlib.pyplot as plt
#define global variable
IM_WIDTH, IM_HEIGHT = 299, 299    #修正 InceptionV3 的尺寸参数
EPOCHS = 50
WORKERS = 6
BAT_SIZE = 24
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172 #微调需要传递的参数
CLASSES_NUM = 3

#end
imageGenerator = ImageDataGenerator(
                                rotation_range=30, 
                                width_shift_range=0.2, 
                                channel_shift_range=0.2,
                                height_shift_range=0.2, 
                                shear_range=0.1, 
                                zoom_range=0.2,  
                                brightness_range=[0.8, 1.2], 
                                horizontal_flip=True, 
                                vertical_flip=True, 
                                fill_mode="constant", 
                                cval=0.0)

#
def get_all_files(pic_dir, file_type):
    ret_list = []
    for file in os.listdir(pic_dir):
        if not file.lower().endswith(file_type.lower()):
            continue
        ret_list.append(pic_dir + file)
    return ret_list

#定义一个方法——获取训练集和验证集中的样本数量，即nb_train_samples, nb_val_samples
def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))       # glob模块是用来查找匹配文件的，后面接匹配规则。
    return cnt

# 定义迁移学习的函数，不需要训练的部分。
def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
# 定义增加最后一个全连接层的函数
def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet

    Args:
        base_model: keras model excluding top
        nb_classes: # of classes

    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #debug
    x = Dropout(0.5)(x)
    #end
    x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    #predictions = Dense(nb_classes, activation='sigmoid')(x) #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# 定义微调函数
def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

        note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

    Args:
        model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    #categorical_crossentropy-> softmax, binary_crossentropy->sigmoid
    
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()

def st_train(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    print("acc: ", acc)
    print("val_acc: ", val_acc)
    print("loss: ", loss)
    print("val_loss: ", val_loss)


def resize_image_by_ratio(image, resize_w_h):
    actual_h, actual_w = image.shape[0: -1]  #1920 1080
    resize_h, resize_w = resize_w_h
    scale_ratio = min(resize_w/actual_w, resize_h/actual_h)
    scale_w = actual_w*scale_ratio #
    scale_h = actual_h*scale_ratio  #
    image = cv2.resize(image, (int(scale_w), int(scale_h)), interpolation = cv2.INTER_AREA)
    new_image = np.zeros((resize_h, resize_w, 3), dtype='float32') # dtype='float32'uint8
    try:
        if scale_h == resize_h:
            left_w = int((resize_w - int(scale_w))//2)
            right_w = int(left_w + int(scale_w))
            new_image[0:int(scale_h), left_w:right_w,  :] = image
        else:
            up_h = int((resize_h - int(scale_h))//2)
            down_h = int(up_h + int(scale_h))
            new_image[up_h:down_h, 0:int(scale_w),  :] = image
        new_image = new_image.reshape(resize_h, resize_w, 3)
        return new_image
    except Exception as e:
        print(e)
        return None

def load_and_resize_image_(file):
    img = cv2.imread(file) #open cv read as BGR
    if len(img.shape) != 3 and img.shape[2] !=3:
        img = None
        return img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR to RGB
    #img = cv2.resize(img, (IM_WIDTH, IM_HEIGHT), interpolation = cv2.INTER_AREA)
    img = resize_image_by_ratio(img, (IM_WIDTH, IM_HEIGHT))
    return img

def load_and_resize_image_method2(file):
    img1 = image.load_img(file)  # target_size参数前面是高
    try:
        img = img_to_array(img1)
    except Exception as e:
        print(file)
        print(e)
        return None
    img = resize_image_by_ratio(img, (IM_WIDTH, IM_HEIGHT))
    return img
    
def load_and_resize_image(file):
    try:
        img = cv2.imread(file)  # target_size参数前面是高
        img_x1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_x1 = img_x1.astype(np.float32)
        '''img = image.load_img(file, target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]))  # target_size参数前面是高
        img_x1 = img_to_array(img)'''
        cols = img_x1.shape[1]
        rows = img_x1.shape[0]

        if rows > cols:
            crop = cols
            x_bias = 0
            y_bias = int((rows - cols) / 2)
        else:
            crop = rows
            y_bias = 0
            x_bias = int((cols - rows) / 2)   
        img_x2 = img_x1[y_bias:y_bias+crop][x_bias:x_bias+crop]
        img_x2 = cv2.resize(img_x2, (IM_WIDTH, IM_HEIGHT), cv2.INTER_AREA)
    except Exception as e:
        img_x2 = None
    return img_x2
    
def batch_generator(img_list, BAT_SIZE, gen_flag):
    X_batch = []
    Y_batch = []
    Y_batch_onehot = []
    label_int_list = []
    if len(img_list) < BAT_SIZE:
        raise "image not enough"
    while True:
        #print(img_list[0:10])
        random.shuffle(img_list)
        #print(img_list[0:10])
        for img_file in img_list:
            label_list = []
            label_num = None
            img = load_and_resize_image_method2(img_file)
            if img is None:
                continue
            label_char = img_file.split('/')[-2]
            if label_char == "negative":
                label_num = 0
                label_list = [1, 0, 0]
            elif label_char == "violence":
                label_num = 1
                label_list = [0, 1, 0]
            elif label_char == "nsfw":
                label_num = 2
                label_list = [0, 0, 1]
            else:
                continue
            label_int_list.append(label_num)
            X_batch.append(img)
            Y_batch.append(label_list)
            #generator
            if gen_flag:
                if label_num == 1 or label_num == 2:
                    img_transf = imageGenerator.get_random_transform(img.shape)
                    new_img = imageGenerator.apply_transform(img, img_transf)
                    X_batch.append(new_img)
                    Y_batch.append(label_list)
                    label_int_list.append(label_num)
#                print("len: X: ", len(X_batch))
            #Y_batch_return = to_categorical(y_batch, 10) 
            if len(X_batch) >= BAT_SIZE:
                X_batch = np.array(X_batch[0:BAT_SIZE], dtype='float32')/255.0
                Y_batch = np.array(Y_batch[0:BAT_SIZE])
                Y_batch_onehot = keras.utils.np_utils.to_categorical(label_int_list[0:BAT_SIZE], num_classes=CLASSES_NUM)
#                 print(Y_batch)
#                 print("len: Y: ", len(Y_batch_onehot))
#                print(Y_batch_onehot)
#                 print(X_batch.shape)
#                 print(X_batch)
#                 print("center: ", X_batch[:, 100:120, 100:120, :])
                yield (X_batch, Y_batch_onehot) #yield X_batch_return, Y_batch_return
                X_batch = []
                Y_batch = []
                Y_batch_onehot = []
                label_int_list = []
            else:
                continue
                


def train(train_dir, val_dir, epochs=EPOCHS, batch_size=BAT_SIZE, restore_model_file="./weights-008.hdf5"):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
#     nb_train_samples = get_nb_files(train_dir)
#     nb_classes = len(glob.glob(train_dir + "/*"))
    class1_img_dir = train_dir + 'negative/'
    class2_img_dir = train_dir + 'violence/'
    class3_img_dir = train_dir + 'nsfw/'
    train_negatives_img_list = sorted(get_all_files(class1_img_dir, '.jpg'))
    train_negatives_img_list =  train_negatives_img_list + sorted(get_all_files(class1_img_dir, '.JPEG'))
    
    train_positives_vio_img_list = sorted(get_all_files(class2_img_dir, '.jpg'))
    train_positives_vio_img_list = train_positives_vio_img_list + sorted(get_all_files(class2_img_dir, '.JPEG'))
    
    train_positives_nsfw_img_list = sorted(get_all_files(class3_img_dir, '.jpg'))
    train_positives_nsfw_img_list = train_positives_nsfw_img_list + sorted(get_all_files(class3_img_dir, '.JPEG'))
    
    train_img_list = train_positives_vio_img_list + train_negatives_img_list + train_positives_nsfw_img_list
    nb_train_samples = len(train_img_list)
    
    class1_img_dir_val = val_dir + 'negative/'
    class2_img_dir_val = val_dir + 'violence/'
    class3_img_dir_val = val_dir + 'nsfw/'
    test_negatives_img_list = sorted(get_all_files(class1_img_dir_val, '.jpg'))
    test_negatives_img_list =  test_negatives_img_list + sorted(get_all_files(class1_img_dir_val, '.JPEG'))
    
    test_positives_vio_img_list = sorted(get_all_files(class2_img_dir_val, '.jpg'))
    test_positives_vio_img_list = test_positives_vio_img_list + sorted(get_all_files(class2_img_dir_val, '.JPEG'))
    
    test_positives_nsfw_img_list = sorted(get_all_files(class3_img_dir_val, '.jpg'))
    test_positives_nsfw_img_list = test_positives_nsfw_img_list + sorted(get_all_files(class3_img_dir_val, '.JPEG'))
    
    test_img_list = test_positives_vio_img_list + test_negatives_img_list + test_positives_nsfw_img_list



    nb_val_samples = get_nb_files(val_dir)
    print('nb_val_samples: ', nb_val_samples)
    print('total val samples: ', len(test_img_list))
    epochs = int(epochs)
    batch_size = int(batch_size)

    # data prep
    train_datagen =  ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        seed=0
    )

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        seed=0
    )
    print("===val batch index: ", validation_generator.batch_index)
    print("====val calss_indices: ", validation_generator.class_indices)
    # 准备跑起来，首先给 base_model 和 model 赋值，迁移学习和微调都是使用 InceptionV3 的 notop 模型
    #（看 inception_v3.py 源码，此模型是打开了最后一个全连接层），利用 add_new_last_layer 函数增加最后一个全连接层。

    base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
    #base_model = InceptionV3(weights='./inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False) #include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, CLASSES_NUM)

    print('start to transfer learning\n')
    # transfer learning
    setup_to_transfer_learn(model, base_model)
    print('start to fine-tune\n')
    # fine-tuning
    setup_to_finetune(model)
    if os.path.exists(restore_model_file):
        model.load_weights(restore_model_file, by_name=True)
        print('loaded weights')
    
    log_dir = './'
    tensorboard = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + "weights-{epoch:03d}.hdf5",
                                     monitor="val_loss",
                                     mode='min',
                                     save_weights_only=False,
                                     save_best_only=False, 
                                     verbose=1,
                                     period=1)
    callbacks_list = [checkpoint, tensorboard]
    #model.fit()train_positives_img_list + train_negatives_img_list
    t = 1 
    class_weight={
        0: 1,
        1: len(train_negatives_img_list) / len(train_positives_vio_img_list) * t,
        2: 1
    }
    print(class_weight)
    random.seed(int(time.time()))
    history_ft = model.fit_generator(
        batch_generator(train_img_list, BAT_SIZE, True) ,  #batch_generator(train_dir + 'violence/', train_dir + 'no_violence/', BAT_SIZE), method2: train_generator 
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        workers=WORKERS,
        use_multiprocessing=True,
	validation_data = batch_generator(test_img_list, 24, True),
        #validation_data=validation_generator,##batch_generator(val_dir, BAT_SIZE)
        validation_steps=nb_val_samples // batch_size,
        class_weight=class_weight,  #class_weight = auto
        callbacks=callbacks_list)
    #model.save(output_model_file)
    st_train(history_ft)
    #plot_training(history_ft)

def transform_label():
    from sklearn.preprocessing import MultiLabelBinarizer
    from keras.utils.np_utils import to_categorical
    labels = [
                ("blue", "jeans"),
                ("blue", "dress"),
                ("red", "dress"),
                ("red", "shirt"),
                ("blue", "shirt"),
                ("black", "jeans")
            ]
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i + 1, label))
        
    label_one_hot = keras.utils.np_utils.to_categorical([0, 1], num_classes=3)#np_utils.to_categorical() utils.to_categorical
    print(label_one_hot)
    
if __name__ == "__main__":
    #root_dir = 'D:/Workspace/tf_vb/keras/violence/violence/'
    root_dir = '/mnt/sda1/terry/data/'
    train_dir = root_dir + 'train_data/'
    val_dir = root_dir + 'test_data/'
    retore_model_file = 'weights-050.hdf5'
#     class1_img_dir = train_dir + 'violence/'
#     class2_img_dir = train_dir + 'no_violence/'
#     train_img_list =  sorted(get_all_files(class1_img_dir, '.jpg'))
#     train_img_list = train_img_list + sorted(get_all_files(class1_img_dir, '.JPEG'))
#     train_img_list = train_img_list + sorted(get_all_files(class2_img_dir, '.jpg'))
#     train_img_list = train_img_list + sorted(get_all_files(class2_img_dir, '.JPEG'))
#     X, Y = batch_generator(train_img_list, BAT_SIZE)
#     print(Y)
    train(train_dir, val_dir)
