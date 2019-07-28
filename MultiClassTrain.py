import argparse
import os
import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import pool
from keras import regularizers
from random import shuffle
from tqdm import tqdm
from imgaug import augmenters as iaa
from keras.models import Model, load_model
from keras.layers import Input, merge, Dense, Dropout, Flatten, Conv2D, \
    MaxPooling2D, BatchNormalization, AveragePooling2D, Activation, Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau, LambdaCallback

from keras.applications import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception



def AlexNet():
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = Conv2D(96, (11, 11), activation='relu', padding='same', strides=(4, 4),
                name='block1a_conv1', kernel_initializer='glorot_normal')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1a-1_pool')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (11, 11), activation='relu', padding='same', strides=(1, 1),
                name='block2a_conv1', kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2a-1_pool')(x)
    x = BatchNormalization()(x)

    x = Conv2D(384, (3, 3), activation='relu', padding='same', strides=(1, 1),
                name='block3a_conv1', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    x = Conv2D(384, (3, 3), activation='relu', padding='same', strides=(1, 1),
                name='block4a_conv1', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(1, 1),
                name='block5a_conv1', kernel_initializer='glorot_normal')(x)

    x = Flatten(name='flattena')(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.1))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.1))(x)
    x = Dropout(0.5)(x)

    output = Dense(Num_Classes, activation='softmax', name='softmax')(x)

    # Create model.
    return Model(inputs=img_input, outputs=output, name='vgg16')



def VGG16():
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Block 1a
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1a_conv1',
                kernel_initializer='glorot_normal')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1a-1_pool')(x)

    # Block 2a
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2a_conv1',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2a_pool')(x)

    # Block 3a
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv1',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv3',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3a_pool')(x)

    # Block 4a
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv1',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv3',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4a_pool')(x)

    x = Flatten(name='flattena')(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.1))(x)

    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.1))(x)

    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu', name='fc3', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.1))(x)
    x = Dropout(0.5)(x)

    output = Dense(Num_Classes, activation='softmax', name='softmax')(x)

    # Create model.
    return Model(inputs=img_input, outputs=output, name='vgg16')


def VGG19():
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Block 1a
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1a_conv1',
                kernel_initializer='glorot_normal')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1a-1_pool')(x)

    # Block 2a
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2a_conv1',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2a_pool')(x)

    # Block 3a
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv1',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3a_conv3',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3a_pool')(x)

    # Block 4a
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv1',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4a_conv3',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4a_pool')(x)

    # Block 5a
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5a_conv1',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5a_conv2',
                kernel_initializer='glorot_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5a_conv3',
                kernel_initializer='glorot_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5a_pool')(x)

    x = Flatten(name='flattena')(x)
    x = BatchNormalization()(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.1))(x)

    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l2(0.1))(x)

    output = Dense(Num_Classes, activation='softmax', name='softmax')(x)

    # Create model.
    return Model(inputs=img_input, outputs=output, name='vgg19')


def Transfer_Learning(Network):
    shape = (IMG_SIZE, IMG_SIZE, 3)
    if Network == 'NASNetLarge':
        base_model = NASNetLarge(input_shape=shape, weights='imagenet', include_top=False)
    elif Network == 'Inception_Resnet_V2':
        base_model = InceptionResNetV2(input_shape=shape, weights='imagenet', include_top=False)
    elif Network == 'Xception':
        base_model = Xception(input_shape=shape, weights='imagenet', include_top=False)
    elif Network == 'InceptionV3':
        base_model = InceptionV3(input_shape=shape, weights='imagenet', include_top=False)
    elif Network == 'DenseNet201':
        base_model = DenseNet201(input_shape=shape, weights='imagenet', include_top=False)
    elif Network == 'MobileNetV2':
        base_model = MobileNetV2(input_shape=shape, weights='imagenet', include_top=False)
    elif Network == 'NASNetMobile':
        base_model = NASNetMobile(input_shape=shape, weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D(name='ex_Pool')(x)
    x = Dense(1024, activation='relu', name='ex_Dense1', kernel_initializer='glorot_normal')(x)
    x = Dense(1024, activation='relu', name='ex_Dense2', kernel_initializer='glorot_normal')(x)
    x = Dense(512, activation='relu', name='ex_Dense3', kernel_initializer='glorot_normal')(x)
    output = Dense(Num_Classes, activation='softmax', name='softmax')(x)
    for layer in base_model.layers: layer.trainable = train_all_weights

    return Model(inputs=base_model.input, outputs=output)



def Write_Classifications():
    file = open('{}/{}/{}_Class_{}_saved_models/{}_{}_{}_Classes.txt'.format(Network, IMG_SIZE, Num_Classes, starttime,
                                                                          Num_Classes, IMG_SIZE, starttime), 'w')
    file.write('Classifications and Training Numbers:\n\n')
    for animal in Num_Images_Dict: file.write(str(animal) + ':' + str(Num_Images_Dict[animal]) + '\n')
    file.write('\n\nClassifications and Testing Numbers:\n\n')
    for animal in Num_Test_Images_Dict: file.write(str(animal) + ':' + str(Num_Test_Images_Dict[animal]) + '\n')


def Plot_Data_Distribution(image_dict, data_type):
    plt.title('{} Class Training Distribution'.format(Num_Classes))
    plt.ylabel('Num Images')
    plt.bar(image_dict.keys(), image_dict.values(), color='g')
    plt.xticks(rotation=90)
    plt.savefig('{}/{}/{}_Class_{}_saved_models/{}_Distribution.png'.format(Network, IMG_SIZE, Num_Classes,
                                                                            starttime, data_type))
def plot_metrics(data1, data2, IMG_SIZE, metric):
    plt.plot(data1)
    plt.plot(data2)
    plt.title('History of Multiclass Animal Model {} During Training'.format(metric))
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}/{}/{}_Class_{}_saved_models/{}_{}_{}_{}.png'.format(Network, IMG_SIZE, Num_Classes,
                                                            starttime, Num_Classes, IMG_SIZE, starttime, metric))
    plt.clf()


def create_data(DATA_DIR, IMG_SIZE, Train):
    data, classnum = [], -1
    for animal in tqdm(os.listdir(DATA_DIR)[:5]):
        print('\n', animal)
        classnum += 1
        i, imglist = 0, os.listdir(DATA_DIR + animal)
        shuffle(imglist)
        for i in range(len(imglist)-1):
            img = cv2.resize(cv2.imread(DATA_DIR + animal + '/' + imglist[i]), (IMG_SIZE, IMG_SIZE))
            zerolist = [0] * Num_Classes
            zerolist[classnum] += 1
            data.append([np.array(img), zerolist])
            if i >= maxbreak: break
            if Train == True:
                for k in range(max_class_num - len(imglist)):
                    aug_img = train_aug.augment_image(img)
                    zerolist = [0] * Num_Classes
                    zerolist[classnum] += 1
                    data.append([np.array(aug_img), zerolist])
    return data


def generator(x_train, y_train, batch_size):
    batch_x_train = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3))
    batch_y_train = np.zeros((batch_size, Num_Classes))
    index_dict = {}
    for i in range(len(x_train)):
        if y_train[i].argmax() not in index_dict: index_dict[y_train[i].argmax()] = [i]
        else: index_dict[y_train[i].argmax()].append(i)

    while True:
        i = 0
        while i < batch_size - 1:
            index = np.random.randint(0, len(x_train) - 1)
            while True:
                if random.random() > ratio_dict[Class_List[y_train[index].argmax()]]:
                    random_index = random.choice(index_dict[y_train[index].argmax()])
                    batch_x_train[i] = batch_aug.augment_image(x_train[random_index])
                    batch_y_train[i] = y_train[random_index]
                    i += 1
                    if i == batch_size - 1: break
                else:
                    batch_x_train[i] = batch_aug.augment_image(x_train[index])
                    batch_y_train[i] = y_train[index]
                    break
            i += 1
        yield batch_x_train, batch_y_train



if __name__=='__main__':

    batch_aug = iaa.SomeOf((1, 2), [
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Affine: Scale/zoom,                  0.46
                   translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # Translate/move
                   rotate=(-90, 90), shear=(-4, 4)),  # Rotate and Shear
        iaa.PiecewiseAffine(scale=(0, 0.05)),  # Distort Image similar water droplet  1.76
        ], random_order=True)

    train_aug = iaa.SomeOf((1, 3), [  # Random number between 0, 3
        iaa.Fliplr(0.5),  # Horizontal flips                     0.01
        # Random channel increase and rotation 0.03
        iaa.Add((-5, 5)),  # Overall Brightness                   0.04
        iaa.Multiply((0.95, 1.05), per_channel=0.2),  # Brightness multiplier per channel    0.05
        iaa.Sharpen(alpha=(0.1, 0.75), lightness=(0.85, 1.15)),  # Sharpness                            0.05
        iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',  # Random HSV increase                  0.09
                           children=iaa.WithChannels(0, iaa.Add((-30, 30)))),
        iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                           children=iaa.WithChannels(1, iaa.Add((-30, 30)))),
        iaa.WithColorspace(to_colorspace='HSV', from_colorspace='RGB',
                           children=iaa.WithChannels(2, iaa.Add((-30, 30)))),
        iaa.AddElementwise((-10, 10)),  # Per pixel addition                   0.11
        iaa.CoarseDropout((0.0, 0.02), size_percent=(0.02, 0.25)),  # Add large black squares              0.13
        iaa.GaussianBlur(sigma=(0.1, 1.0)),  # GaussianBlur                         0.14
        iaa.Grayscale(alpha=(0.1, 1.0)),  # Random Grayscale conversion          0.17
        iaa.Dropout(p=(0, 0.1), per_channel=0.2),  # Add small black squares              0.17
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Add Gaussian per pixel noise         0.26
        iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.25),  # Distort image by rearranging pixels  0.70
        iaa.ContrastNormalization((0.75, 1.5)),  # Contrast Normalization               0.95
        iaa.weather.Clouds(),
        iaa.weather.Fog(),
        iaa.weather.Snowflakes()
    ], random_order=True)

    #View Network Statistics at https://keras.io/applications/

    parser = argparse.ArgumentParser(description='Directories and Models')
    parser.add_argument('--train_dir', type=str, default='Train/')
    parser.add_argument('--test_dir', type=str, default='Test/')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--network', type=str, default='MobileNetV2')

    args = parser.parse_args()

    #1080 TI
    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir
    IMG_SIZE = args.img_size
    Network = args.network
    batch_size = args.batch_size
    epochs = args.epoch

    train_all_weights = True

    Class_List = os.listdir(TRAIN_DIR)
    Num_Classes = len(Class_List)
    maxbreak = 200000

    Num_Images_Dict, Num_Test_Images_Dict, ratio_dict = {}, {}, {}
    for animal in os.listdir(TRAIN_DIR): Num_Images_Dict[animal] = len(os.listdir(TRAIN_DIR + animal))
    for animal in os.listdir(TEST_DIR): Num_Test_Images_Dict[animal] = len(os.listdir(TEST_DIR + animal))

    for classification in Num_Images_Dict:
        if Num_Images_Dict[classification] > maxbreak: Num_Images_Dict[classification] = maxbreak

    max_class_num = Num_Images_Dict[max(Num_Images_Dict, key=Num_Images_Dict.get)]

    for classification in Num_Images_Dict:
        if Num_Images_Dict[classification] == maxbreak: ratio_dict[classification] = 1
        else: ratio_dict[classification] = Num_Images_Dict[classification] / max_class_num

    train_data = create_data(TRAIN_DIR, IMG_SIZE, Train=True)
    test_data = create_data(TEST_DIR, IMG_SIZE, Train=False)

    print('Train Size: {}'.format(len(train_data)))
    print('Test Size: {}'.format(len(test_data)))

    x_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y_train = np.array([i[1] for i in train_data])

    x_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y_test = np.array([i[1] for i in test_data])

    del train_data
    del test_data

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    starttime = time.strftime("%Y%m%d-%H%M%S")

    if not os.path.exists('{}/{}/{}_Class_{}_saved_models/'.format(Network, IMG_SIZE, Num_Classes, starttime)):
        os.makedirs('{}/{}/{}_Class_{}_saved_models/'.format(Network, IMG_SIZE, Num_Classes, starttime))

    Write_Classifications()
    Plot_Data_Distribution(Num_Images_Dict, 'Training')
    Plot_Data_Distribution(Num_Test_Images_Dict, 'Testing')

    if Network == 'VGG16': model = VGG16()
    elif Network == 'VGG19': model = VGG19()
    else: model = Transfer_Learning(Network)

    # model = load_model('Inception_Resnet_V2/160/55_Class_20190313-125744_saved_models/weights.04-0.20.hdf5')


    # for i,layer in enumerate(model.layers): print(i,layer.name)

    model.summary()
    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6),
                  loss={'softmax': 'categorical_crossentropy'},
                  metrics={'softmax': 'accuracy'})


    # TensorBoard(log_dir='{}_{}TB_Logger./log'.format(Network, starttime))
    csv_logger = CSVLogger('{}/{}/{}_Class_{}_saved_models/{}_{}_{}_Logger.csv'.format(Network, IMG_SIZE, Num_Classes,
                                                                starttime, Num_Classes, IMG_SIZE, starttime), separator=',')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)

    history = model.fit_generator(generator(x_train, y_train, batch_size),
                                  validation_data=(x_test, y_test),
                                  shuffle=True,
                                  steps_per_epoch=x_train.shape[0] / batch_size,
                                  epochs=epochs,
                                  callbacks=[# EarlyStopping(min_delta=0.001, patience=3),
                                      csv_logger,
                                      ModelCheckpoint('%s/%s/%s_Class_%s_saved_models/weights.{epoch:02d}-'
                                                     '{val_acc:.2f}.hdf5' % (Network, IMG_SIZE, Num_Classes, starttime),
                                                      monitor='val_acc', verbose=0, save_best_only=True,
                                                      save_weights_only=False, mode='auto', period=1)])

    plot_metrics(history.history['loss'], history.history['val_loss'], IMG_SIZE, 'Loss')
    plot_metrics(history.history['acc'], history.history['val_acc'], IMG_SIZE, 'Accuracy')







