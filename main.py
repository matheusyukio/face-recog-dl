#!/usr/bin/env python
# coding: utf-8

# imports
import os
import pandas as pd
import numpy as np
# visualizazao
import matplotlib.pyplot as plt
# Mover arquivos
import shutil
from datetime import datetime
# CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
# Pooling layers
from tensorflow.keras.layers import MaxPooling2D
# flatten layers into single vector
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.layers import AveragePooling2D, LocallyConnected2D
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3

from sklearn.model_selection import StratifiedKFold
import datetime


def get_data_transform():
    # base de dados
    lfw_allnames = pd.read_csv("lfw-dataset/lfw_allnames.csv")
    # Monta um DF como caminho para cada imagem da pessoa
    # Por exemplo se há 55 imagens da mesma pessoa é montado o DF dos caminhos para as 55 imagens dela
    image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
    image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
    image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
    image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
    image_paths = image_paths.drop("images", 1)
    return image_paths

def get_min_img(image_paths, min_img):
    ind_counts = image_paths.groupby('name').count().image_path
    image_list = []
    for img in ind_counts[ind_counts >= min_img].iteritems():
        image_list.append(img[0])
    return image_list

def mount_data(pd_df, min_img, sample_size):
    person_list = get_min_img(pd_df, min_img)
    total_filtered = pd_df[pd_df['name'].isin(person_list)]
    sample_list = []
    for img in person_list:
        sample_list.append(total_filtered[total_filtered.name == img].sample(sample_size))
    return pd.concat(sample_list)

def get_mounted_data(min_img, sample_size):
    image_paths = get_data_transform()
    return mount_data(image_paths, min_img, sample_size)

def directory_mover(data, dir_name):
    co = 0
    for image in data.image_path:
        # create top directory
        if not os.path.exists(os.path.join(os.getcwd(), 'new/working/', dir_name)):
            os.makedirs(os.path.join(os.path.join(os.getcwd(), 'new/working/'), dir_name))

        data_type = data[data['image_path'] == image]['name']
        data_type = str(list(data_type)[0])
        if not os.path.exists(os.path.join(os.getcwd(), 'new/working/', dir_name, data_type)):
            os.makedirs(os.path.join(os.path.join(os.getcwd(), 'new/working/', dir_name, data_type)))
        path_from = os.path.join('lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/', image)
        path_to = os.path.join('new/working/', dir_name, data_type)
        shutil.copy(path_from, path_to)
        co += 1

    print('Moved {} images to {} folder.'.format(co, dir_name))

def dataTrainAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

def dataHoldOutAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3
    )

def dataTestAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

def get_model_name(name, k, batch):
    return 'model_' + name + '_' + str(k) + '_' + str(batch) + '.h5'

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

def create_new_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (250, 250, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = num_classes, activation = 'softmax'))
    return model

def DeepFace(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (11, 11), activation='relu', name='C1', input_shape=(250, 250, 3)))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    model.add(Conv2D(16, (9, 9), activation='relu', name='C3'))
    model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
    model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5'))
    model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
    model.add(Flatten(name='F0'))
    model.add(Dense(4096, activation='relu', name='F7'))
    model.add(Dropout(rate=0.5, name='D0'))
    model.add(Dense(num_classes, activation='softmax', name='F8'))
    return model

def LeNet5(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(250, 250, 3)))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def AlexNet(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     input_shape=(250, 250, 3), strides=(4, 4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(4096, input_shape=(250 * 250 * 3,), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model

def write_results(filename, acc, loss, history):
    VALIDATION_ACCURACY = acc
    VALIDATION_LOSS = loss
    HISTORY = history
    file = open(filename, 'a+')
    file.write('VALIDATION_ACCURACY \n')
    file.write(str(VALIDATION_ACCURACY))
    file.write('\n')
    file.write('VALIDATION_ACCURACY mean\n')
    file.write(str(np.mean(VALIDATION_ACCURACY)))
    file.write('\n')
    file.write('VALIDATION_ACCURACY std\n')
    file.write(str(np.std(VALIDATION_ACCURACY)))
    file.write('\n')
    file.write('\n')
    file.write('VALIDATION_LOSS \n')
    file.write(str(VALIDATION_LOSS))
    file.write('\n\n')
    for hist in range(len(HISTORY)):
        file.write('VALIDATION_ACCURACY HISTORY ' + str(hist) + '\n')
        file.write(str(VALIDATION_ACCURACY[hist]))
        file.write('\n')
        file.write('VALIDATION_LOSS HISTORY ' + str(hist) + '\n')
        file.write(str(VALIDATION_LOSS[hist]))
        file.write('\n')
        file.write('HISTORY ' + str(hist) + ' \n')
        file.write(str(HISTORY[hist].history))
        file.write('\n\n')
    file.close()

def plot_train_test_loss(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def run_hold_out(multi_data, X, Y, CLASSES, epoch, MODEL, BATCH_SIZE=32):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    HISTORY = []
    MODEL_NAME = MODEL
    FOLDS = 2
    EPOCHS = epoch
    save_dir = os.path.join(os.getcwd(), 'models/')
    VERBOSE = 1
    fold_var = 7030

    # directory_mover(multi_data,"multi_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var))

    # no hold out da algum erro no flow_from_dataframe
    train_data_generator = dataHoldOutAugmentation().flow_from_dataframe(
        # training_data,
        dataframe=multi_data,
        directory=os.path.join(os.getcwd(), 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'),
        target_size=(250, 250),
        x_col="image_path", y_col="name",
        batch_size=BATCH_SIZE,
        subset="training",
        class_mode="categorical",
        # modificado apenas para verificar o hold out
        shuffle=True
    )

    valid_data_generator = dataHoldOutAugmentation().flow_from_dataframe(
        # training_data,
        dataframe=multi_data,
        directory=os.path.join(os.getcwd(), 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'),
        target_size=(250, 250),
        x_col="image_path", y_col="name",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        # modificado apenas para verificar o hold out
        shuffle=True
    )

    model = get_model(MODEL, CLASSES)
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    # CREATE CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + get_model_name(MODEL_NAME, fold_var, BATCH_SIZE),
                                                    monitor='val_acc', verbose=VERBOSE,
                                                    save_best_only=True, mode='max')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE, patience=500)
    callbacks_list = [checkpoint, earlystopping]
    history = model.fit(train_data_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_data_generator.n // train_data_generator.batch_size,
                        callbacks=callbacks_list,
                        validation_data=valid_data_generator,
                        validation_steps=valid_data_generator.n // valid_data_generator.batch_size,
                        verbose=VERBOSE)

    HISTORY.append(history)

    # LOAD BEST MODEL to evaluate the performance of the model
    model.load_weights(
        os.getcwd() + "/models/model_" + MODEL_NAME + "_" + str(fold_var) + '_' + str(BATCH_SIZE) + ".h5")

    results = model.evaluate(valid_data_generator)
    # results = model.evaluate_generator(valid_data_generator)
    results = dict(zip(model.metrics_names, results))

    VALIDATION_ACCURACY.append(results['acc'])
    VALIDATION_LOSS.append(results['loss'])

    write_results(
        get_current_time_str() + '_holdout_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
            BATCH_SIZE) + '.txt', VALIDATION_ACCURACY, VALIDATION_LOSS, HISTORY)


def run_k_fold(multi_data, X, Y, CLASSES, epoch, MODEL, BATCH_SIZE, num_folds):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    HISTORY = []
    MODEL_NAME = MODEL
    FOLDS = num_folds
    EPOCHS = epoch
    save_dir = os.path.join(os.getcwd(), 'models/')
    VERBOSE = 1

    skf = StratifiedKFold(n_splits=FOLDS, random_state=7, shuffle=True)

    fold_var = 1
    for train_index, val_index in skf.split(X, Y):
        print("=======EPOCHS ", EPOCHS, " Start--k: ", fold_var)

        training_data = multi_data.iloc[train_index]
        validation_data = multi_data.iloc[val_index]

        print(training_data.shape)
        print(validation_data.shape)

        # directory_mover(training_data,"training_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var))
        # directory_mover(validation_data,"validation_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var))
        # flow_from_dataframe
        train_data_generator = dataTrainAugmentation().flow_from_dataframe(
            dataframe=training_data,
            directory=os.path.join(os.getcwd(), 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'),
            target_size=(250, 250),
            x_col="image_path", y_col="name",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False)

        valid_data_generator = dataTrainAugmentation().flow_from_dataframe(
            dataframe=validation_data,
            directory=os.path.join(os.getcwd(), 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'),
            target_size=(250, 250),
            x_col="image_path", y_col="name",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False)
        model = get_model(MODEL, CLASSES)
        # rmsprop = RMSprop(lr=1e-3, decay=1e-6)
        sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        # CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + get_model_name(MODEL_NAME, fold_var, BATCH_SIZE),
                                                        monitor='val_acc', verbose=VERBOSE,
                                                        save_best_only=True, mode='max')
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE, patience=500)

        callbacks_list = [checkpoint, earlystopping]

        '''
        STEP_SIZE_TRAIN=train_data_generator.n//train_data_generator.batch_size
        STEP_SIZE_VALID=valid_data_generator.n//valid_data_generator.batch_size
        print("STEP_SIZE_TRAIN ",STEP_SIZE_TRAIN)
        print("STEP_SIZE_VALID ",STEP_SIZE_VALID)
        history = model.fit_generator(generator=train_data_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    #steps_per_epoch=training_data.shape[0],
                    validation_data=valid_data_generator,
                    validation_steps=STEP_SIZE_VALID,
                    #validation_steps=validation_data.shape[0],
                    epochs=EPOCHS,
                    #callbacks=callbacks_list,
                    verbose=VERBOSE)
        '''
        history = model.fit(train_data_generator,
                            epochs=EPOCHS,
                            steps_per_epoch=train_data_generator.n // train_data_generator.batch_size,
                            callbacks=callbacks_list,
                            validation_data=valid_data_generator,
                            validation_steps=valid_data_generator.n // valid_data_generator.batch_size,
                            verbose=VERBOSE)

        HISTORY.append(history)

        # LOAD BEST MODEL to evaluate the performance of the model model_"+MODEL_NAME+"_"+str(fold_var)+".h5"
        model.load_weights(
            os.getcwd() + "/models/model_" + MODEL_NAME + "_" + str(fold_var) + '_' + str(BATCH_SIZE) + ".h5")

        results = model.evaluate(valid_data_generator)
        # results = model.evaluate_generator(valid_data_generator)
        results = dict(zip(model.metrics_names, results))

        VALIDATION_ACCURACY.append(results['acc'])
        VALIDATION_LOSS.append(results['loss'])

        write_results(
            get_current_time_str() + '_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE) + '.txt', VALIDATION_ACCURACY, VALIDATION_LOSS, HISTORY)

        fold_var += 1


def get_model(model_name, num_classes):
    if model_name == "create_new_model":
        return create_new_model(num_classes)
    elif model_name == "AlexNet":
        return AlexNet(num_classes)
    elif model_name == "LeNet5":
        return LeNet5(num_classes)
    elif model_name == "VGG16":
        return VGG16(num_classes)
    elif model_name == "ResNet50":
        return ResNet50(num_classes)
    elif model_name == "InceptionV3":
        return InceptionV3(num_classes)
    elif model_name == "DeepFace":
        return DeepFace(num_classes)


"""
params = {
    epoch
    min_images_per_person
        number of classes
    model
    batch_size
    hold_out
    k-fold
}
# SINTONIZAR COM LENET ALEXNET e executar com outras arquiteturas
# com todas arquiteturas
"""

def main():
    epoch = 600
    min_images_per_person = [30]  # [25,20]
    models = ["DeepFace"]#["DeepFace",AlexNet","LeNet5"]

    batch_sizes = [2]  # [2,4,8]
    for min_per_person in min_images_per_person:
        for batch in batch_sizes:
            for model in models:
                multi_data = get_mounted_data(min_per_person, min_per_person)
                Y = multi_data[['name']]
                X = multi_data[['image_path']]
                CLASSES = Y.groupby('name').nunique().shape[0]
                # print("### run_hold_out "," epoch ", epoch, " min_per_person ", min_per_person," CLASSES ", CLASSES,"model ",model," batch_size ",batch)
                # run_hold_out(multi_data, X, Y, CLASSES, epoch, model, batch)
                print("### run_k_fold ", " epoch ", epoch, " min_per_person ", min_per_person, " CLASSES ", CLASSES,
                      "model ", model, " batch_size ", batch)
                run_k_fold(multi_data, X, Y, CLASSES, epoch, model, batch, 5)

if __name__ == "__main__":
    main()