#!/usr/bin/env python
# coding: utf-8

# imports
import os
import gc
import math
# Mover arquivos
from datetime import datetime
# CNN
import tensorflow as tf

from tensorflow.keras import optimizers

from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import datetime
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import itertools

from dataprocess import get_mounted_data, directory_mover, dataTrainAugmentation

from CALFWdataprocess import calfw_mount_data, calfw_directory_mover
#from TFpathUnbatch import transform_image_to_tfrecord_image_path
from models import create_new_model, DeepFace, LeNet5, AlexNet, VGGFace

from write_plot_history import write_results

from write_cm_report import write_cm_report
from plot_cm import plot_confusion_matrix

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import IPython.display as display
from PIL import Image
import pandas as pd
from scipy.stats import mode

global CLASS_NAMES_GLOBAL
CLASS_NAMES_GLOBAL = []

def transform_image_to_tfrecord_image_path(image_path, BATCH_SIZE):
    BATCH_SIZE = BATCH_SIZE
    data_dir = pathlib.Path(image_path)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    IMG_HEIGHT, IMG_WIDTH = 250, 250
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    #print(list_ds)
    CLASS_NAMES_GLOBAL.append(CLASS_NAMES)

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        # normalize
        return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=500):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        #ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    BATCH_SIZE = BATCH_SIZE
    return prepare_for_training(labeled_ds)
    # image_batch, label_batch = next(iter(train_ds))
    # print(image_batch.shape)
    # print(label_batch.shape)

    '''
    data_dir = pathlib.Path(image_path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    list_ds = tf.data.Dataset.list_files(str(data_dir/ '*/*'))
    print(list_ds)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    print(labeled_ds)
    '''

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
    elif model_name == "VGGFace":
        return VGGFace(num_classes)

def get_model_name(name, k, batch):
    return 'model_TFrecord' + name + '_' + str(k) + '_' + str(batch) + '.h5'

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

def run_k_fold(multi_data, X, Y, CLASSES, MODEL, BATCH_SIZE, num_folds, nomes_classes):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    HISTORY = []
    MODEL_NAME = MODEL
    FOLDS = num_folds
    EPOCHS = 0
    save_dir = os.path.join(os.getcwd(), 'models/')
    VERBOSE = 1

    #calfw_directory_mover(multi_data,"validation_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str('fold_var'))


    valid_data_generator = dataTrainAugmentation().flow_from_directory(
            # training_data,
            directory=os.path.join(os.getcwd(), 'new_calfw/working/validation_data_'+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str('fold_var')+'/'),
            target_size=(250, 250),
            # x_col = "image_path", y_col = "name",
            batch_size=1,
            class_mode="categorical",
            #subset="validation",
            shuffle=False)

    model = get_model(MODEL, CLASSES)
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
    model.load_weights("model_TFrecordDeepFace_1_60.h5")
        
    results = model.evaluate(valid_data_generator)
        # results = model.evaluate_generator(valid_data_generator)
    predict = model.predict_generator(valid_data_generator)
    print('predict 1')
    print(predict)
    print(np.argmax(predict, axis=-1))
    classes1 = np.argmax(predict, axis=-1)
    print('results 1')
    print(results)
    results = dict(zip(model.metrics_names, results))

    y_pred = np.argmax(predict, axis=1)
    print(y_pred)
    print(valid_data_generator.classes)

    write_cm_report(Y, valid_data_generator.classes, y_pred, 'calfw_model1_'+get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE)+'.txt')
    cm = confusion_matrix(valid_data_generator.classes, y_pred)
    plot_confusion_matrix(cm, classes=nomes_classes, CLASSES=CLASSES, MODEL_NAME=MODEL_NAME, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, title='Matriz de Confusão CALFW')

    model2 = model
    model2.load_weights("model_TFrecordDeepFace_2_60.h5")
    predict2 = model2.predict(valid_data_generator)
    print('predict 2')
    print(predict2)
    print(np.argmax(predict2, axis=-1))
    classes2 = np.argmax(predict2, axis=-1)
    results2 = model2.evaluate(valid_data_generator)
    print('results 2')
    print(results2)

    model3 = model
    model3.load_weights("model_TFrecordDeepFace_3_60.h5")
    predict3 = model3.predict(valid_data_generator)
    print('predict 3')
    print(predict3)
    print(np.argmax(predict3, axis=-1))  
    classes3 = np.argmax(predict3, axis=-1)
    results3 = model3.evaluate(valid_data_generator)
    print('results 3')
    print(results3)

    print("MEAN ====================")
    final_mean = (predict + predict2 + predict3)/3
    print(final_mean.shape)
    print(final_mean)
    print(np.argmax(final_mean, axis=-1))
    pred_ensemble_media = np.argmax(final_mean, axis=-1)

    write_cm_report(Y, valid_data_generator.classes, pred_ensemble_media, 'ensemble_media_'+get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
        BATCH_SIZE)+'.txt')
    cm2 = confusion_matrix(valid_data_generator.classes, pred_ensemble_media)
    plot_confusion_matrix(cm2, classes=nomes_classes, CLASSES=CLASSES, MODEL_NAME=MODEL_NAME, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, title='Matriz de Confusão')

    print("Voto marjoritario ====================")

    final_pred_mode = np.array([])
        #final_pred_mode = []
    print(multi_data.shape[0])
    print(len(valid_data_generator.classes))
    for i in range(0,len(valid_data_generator.classes)):
        print(classes1[i])
        print(classes2[i])
        print(classes3[i])
        print(mode([classes1[i], classes2[i], classes3[i]]))
        final_pred_mode = np.append(final_pred_mode, mode([classes1[i], classes2[i], classes3[i]])[0][0])
            #final_pred_mode.append(statistics.mode([predict[i], predict2[i], predict3[i]]))
    print('final_pred_mode')
    print(len(final_pred_mode))        
    print(final_pred_mode)
    print(final_pred_mode.astype(int))
    print(type(final_pred_mode[0]))
    print(final_pred_mode[0])
    print(np.argmax(final_pred_mode, axis=-1))
    pred_ensemble_mode = np.argmax(final_pred_mode, axis=-1)

    write_cm_report(Y, valid_data_generator.classes, final_pred_mode.astype(int), 'ensemble_marjoritario_'+get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE)+'.txt')
    cm3 = confusion_matrix(valid_data_generator.classes, final_pred_mode.astype(int))
    plot_confusion_matrix(cm3, classes=nomes_classes, CLASSES=CLASSES, MODEL_NAME=MODEL_NAME, EPOCHS=EPOCHS, BATCH_SIZE=BATCH_SIZE, title='Matriz de Confusão')

    del model
    #del history
    tf.keras.backend.clear_session()
    gc.collect()
    tf.compat.v1.reset_default_graph()

def main():
    min_images_per_person = [30]
    models = ["DeepFace"]
    num_folds = 5
    batch_sizes = [60]

    min_per_person = min_images_per_person[0]

    multi_data = get_mounted_data(min_per_person, min_per_person)
    Y = multi_data[['name']]
    X = multi_data[['image_path']]
    CLASSES = Y.groupby('name').nunique().shape[0]
    nomes_classes = []
    for i in pd.DataFrame(Y.groupby('name')['name'].nunique().reset_index(name="unique"))['name']:
        nomes_classes.append(str(i))
    print(CLASSES)
    print(nomes_classes)

    calfw_df = calfw_mount_data(nomes_classes)
    print(calfw_df.shape)
    calfw_Y = calfw_df[['name']]
    calfw_X = calfw_df[['image_path']]
    calfw_CLASSES = calfw_Y.groupby('name').nunique().shape[0]
    #print(calfw_df.groupby('name')['name'].nunique().reset_index(name="unique")['name'])
    print(calfw_Y)
    print(calfw_X)
    print(calfw_CLASSES)
    ind_counts = calfw_df.groupby('name').count().image_path
    print(ind_counts)
    calfw_classes = []
    for j in pd.DataFrame(calfw_df.groupby('name')['name'].nunique().reset_index(name="unique"))['name']:
        calfw_classes.append(str(j))
    print(calfw_classes)

    print(set(nomes_classes) - set(calfw_classes))
    print(set(calfw_classes) - set(nomes_classes))


    for batch in batch_sizes:
        for model in models:
            print("### run_k_fold ", " min_per_person ", min_per_person, " CLASSES ", CLASSES,
          "model ", model, " batch_size ", batch)
            run_k_fold(calfw_df, X, Y, CLASSES, model, batch, num_folds, nomes_classes)
            tf.keras.backend.clear_session()
            gc.collect()

    #model = get_model("DeepFace", 34)
    #model.load_weights("model_TFrecordDeepFace_1_60.h5")
    #print(model.summary)

if __name__ == "__main__":
    CLASS_NAMES = []
    main()