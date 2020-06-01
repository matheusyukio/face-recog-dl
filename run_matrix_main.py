#!/usr/bin/env python
# coding: utf-8

# imports
import os
import gc
import numpy as np
# visualizazao
import matplotlib.pyplot as plt
# Mover arquivos
from datetime import datetime
# CNN
import tensorflow as tf

from tensorflow.keras import optimizers

from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3

from sklearn.model_selection import StratifiedKFold
import datetime
from PIL import Image
import pandas as pd
from dataprocess import dataTrainAugmentation, dataHoldOutAugmentation, get_mounted_data

from models import create_new_model, DeepFace, LeNet5, AlexNet

def get_model_name(name, k, batch):
    return 'model_main_matrix' + name + '_' + str(k) + '_' + str(batch) + '.h5'

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

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


def transform_image_dataframe_to_matrix(dataframe, new_width, new_height, from_current_to_images_path):
    dataframe_rows = dataframe.shape[0]
    list_images = np.zeros((dataframe_rows, new_width, new_height, 3), dtype=np.float64)
    count = 0
    for image in dataframe.image_path:
        read = Image.open(os.path.join(os.getcwd(), from_current_to_images_path, image)).resize((new_width, new_height))
        data = np.asarray(read)
        list_images[count] = data
        count += 1

    labels = pd.get_dummies(dataframe["name"]).to_numpy()
    return list_images, labels

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
        data_x, data_y = transform_image_dataframe_to_matrix(multi_data, 250, 250, 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/')
        #training_data = data_x[train_index]
        #training_data_label = data_y[train_index]
        #validation_data = data_x[val_index]
        #validation_data_label = data_y[val_index]

        print(data_x[train_index].shape)
        print(data_x[val_index].shape)
        print('batch')
        print(BATCH_SIZE)

        model = get_model(MODEL, CLASSES)

        sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        # CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + get_model_name(MODEL_NAME, fold_var, BATCH_SIZE),
                                                        monitor='val_acc', verbose=VERBOSE,
                                                        save_best_only=True, mode='max')
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE, patience=400)

        callbacks_list = [checkpoint, earlystopping]

        history = model.fit(data_x[train_index],
                            data_y[train_index],
                            epochs=EPOCHS,
                            batch_size=data_x[train_index].shape[0] // BATCH_SIZE,
                            #steps_per_epoch=data_x[train_index].shape[0] // BATCH_SIZE,
                            callbacks=callbacks_list,
                            validation_data=(data_x[val_index],data_y[val_index]),
                            #validation_steps=data_x[val_index].shape[0] // BATCH_SIZE,
                            verbose=VERBOSE)

        HISTORY.append(history)

        # LOAD BEST MODEL to evaluate the performance of the model model_"+MODEL_NAME+"_"+str(fold_var)+".h5"
        model.load_weights(
            os.getcwd() + "/models/model_main_matrix" + MODEL_NAME + "_" + str(fold_var) + '_' + str(BATCH_SIZE) + ".h5")

        results = model.evaluate(data_x[val_index],data_y[val_index], batch_size=data_x[val_index].shape[0] // BATCH_SIZE)

        results = dict(zip(model.metrics_names, results))

        VALIDATION_ACCURACY.append(results['acc'])
        VALIDATION_LOSS.append(results['loss'])

        write_results(
            get_current_time_str() + 'model_main_matrix_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE) + '.txt', VALIDATION_ACCURACY, VALIDATION_LOSS, HISTORY)
        del history
        del model
        tf.keras.backend.clear_session()
        gc.collect()
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
    epoch = 500
    min_images_per_person = [30]  # [25,20]
    models = ["LeNet5","DeepFace","AlexNet"]#["DeepFace",AlexNet","LeNet5"]
    num_folds = 5

    #aumentando o batch para 30 DeepFace conseguiu bons resultados, testar com outras
    batch_sizes = [30,2]  # [2,4,8]
    for min_per_person in min_images_per_person:
        for batch in batch_sizes:
            for model in models:
                multi_data = get_mounted_data(min_per_person, min_per_person)
                #Y = multi_data[['name']]
                #X = multi_data[['image_path']]
                #CLASSES = Y.groupby('name').nunique().shape[0]
                # print("### run_hold_out "," epoch ", epoch, " min_per_person ", min_per_person," CLASSES ", CLASSES,"model ",model," batch_size ",batch)
                # run_hold_out(multi_data, X, Y, CLASSES, epoch, model, batch)
                print("### run_k_fold ", " epoch ", epoch, " min_per_person ", min_per_person, " CLASSES ", multi_data[['name']].groupby('name').nunique().shape[0],
                      "model ", model, " batch_size ", batch)
                run_k_fold(multi_data, multi_data[['image_path']], multi_data[['name']], multi_data[['name']].groupby('name').nunique().shape[0], epoch, model, batch, num_folds)

if __name__ == "__main__":
    main()