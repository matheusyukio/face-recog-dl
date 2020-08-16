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

from dataprocess import get_mounted_data, directory_mover
#from TFpathUnbatch import transform_image_to_tfrecord_image_path
from models import create_new_model, DeepFace, LeNet5, AlexNet, VGGFace

from write_plot_history import write_results

AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import IPython.display as display
from PIL import Image
#import statistics
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

def run_k_fold(multi_data, X, Y, CLASSES, MODEL, BATCH_SIZE, num_folds):
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    HISTORY = []
    MODEL_NAME = MODEL
    FOLDS = num_folds
    EPOCHS = 0
    save_dir = os.path.join(os.getcwd(), 'models/')
    VERBOSE = 1

    skf = StratifiedKFold(n_splits=FOLDS, random_state=7, shuffle=True)

    fold_var = 1
    for train_index, val_index in skf.split(X, Y):
        print("=======EPOCHS ", EPOCHS, " Start--k: ", fold_var)

        validation_data = multi_data.iloc[val_index]

        print(validation_data.shape)

        directory_mover(validation_data,"validation_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var))

        # tfrecord
        ds_validation = transform_image_to_tfrecord_image_path(os.path.join(os.getcwd(),"new\working\\","validation_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var)), BATCH_SIZE)

        model = get_model(MODEL, CLASSES)
        sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
        model.load_weights("model_TFrecordDeepFace_1_60.h5")
        
        results = model.evaluate(ds_validation)
        # results = model.evaluate_generator(valid_data_generator)
        predict = model.predict(ds_validation)
        print('predict 1')
        print(predict)
        print(predict.shape)

        print(np.argmax(predict, axis=-1))
        classes1 = np.argmax(predict, axis=-1)
        print('results 1')
        print(results)
        results = dict(zip(model.metrics_names, results))

        model2 = model
        model2.load_weights("model_TFrecordDeepFace_2_60.h5")
        predict2 = model2.predict(ds_validation)
        print('predict 2')
        print(predict2[:3])
        print(predict2.shape)
        print(np.argmax(predict2, axis=-1))
        classes2 = np.argmax(predict2, axis=-1)
        results2 = model2.evaluate(ds_validation)
        print('results 2')
        print(results2)


        model3 = model
        model3.load_weights("model_TFrecordDeepFace_3_60.h5")
        predict3 = model3.predict(ds_validation)
        print('predict 3')
        print(predict3)
        print(np.argmax(predict3, axis=-1))  
        classes3 = np.argmax(predict3, axis=-1)
        results3 = model3.evaluate(ds_validation)
        print('results 3')
        print(results3)

        print("MAX ====================")
        # final_max = np.array([])
        # for i in range(0,validation_data.shape[0]):
        #     predict[i]
        #     final_max = np.append(final_max, max([predict[i], predict2[i]]))
        # print(final_max)

        final_pred = np.array([])
        #final_pred = []
        for i in range(0,validation_data.shape[0]):
            print(classes1[i])
            print(classes2[i])
            print(classes3[i])
            print(mode([classes1[i], classes2[i], classes3[i]]))
            final_pred = np.append(final_pred, mode([classes1[i], classes2[i], classes3[i]]))
            #final_pred.append(statistics.mode([predict[i], predict2[i], predict3[i]]))

        print(final_pred)
        print(np.argmax(final_pred, axis=-1))

        print("MEAN ====================")
        final_mean = (predict + predict2 + predict3)/3
        print(final_mean.shape)
        print(final_mean)
        print(np.argmax(final_mean, axis=-1))  


        """
        [0.1 0.2 0.7] modelo1
        [0.3333 0.333 0.333300001] modelo2
        [0.7 0.3 0.00001] modelo3
        =================== 
        media


        labels_ds = ds_validation.map(lambda image, label: label).unbatch()
        print('labels_ds') #_UnbatchDataset 
        print(labels_ds) #_UnbatchDataset 
        print(dir(labels_ds)) #_UnbatchDataset 
        all_labels = []
        all_labels.append(next(iter(labels_ds.batch(validation_data.shape[0]))).numpy())
        print('all_labels')
        print(all_labels)
        print(len(all_labels))
        cm_correct_labels = np.concatenate(all_labels)
        print(cm_correct_labels)
        print(cm_correct_labels.shape)
        #for i in cm_correct_labels:
        #    print(i)
        label2 = []
        label2.append(np.argmax(cm_correct_labels, axis=-1))
        print('label2')
        print(len(label2))
        print(label2)
        #transform
        label_transf = np.concatenate(label2)
        print('label_transf')
        print(len(label_transf))
        print(label_transf)
        #3for i in cm_correct_labels:
        # 3   print(i) cada imagem tem um array de 34 onde a posição True indica qual o label
        #predict = model.predict()        
        print('predict')
        print(predict.shape)
        print(predict)
        
        all_pred = []
        all_pred.append( np.argmax(predict, axis=-1) )
        print('all_pred')
        print(all_pred)
        print(len(all_pred))

        cm_predictions = np.concatenate(all_pred)
        print('cm_predictions')
        print(cm_predictions.shape)
        print(cm_predictions)

        # results = model.evaluate_generator(valid_data_generator)
        results = dict(zip(model.metrics_names, results))
        print('results 2')
        print(results)
        print('quem nunca eh classificado')
        print(set(label_transf) - set(cm_predictions))
        cmat = confusion_matrix(label_transf, cm_predictions)
        print(cmat)

        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            
            #This function prints and plots the confusion matrix.
            #Normalization can be applied by setting `normalize=True`.
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            plt.figure(figsize=(CLASSES+10, CLASSES+10))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            #plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('Classe Real')
            plt.xlabel('Classe Predita')
            plt.savefig(get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE) + 'CM.png')
            plt.close()

        score = f1_score(label_transf, cm_predictions, average='macro')
        precision = precision_score(label_transf, cm_predictions, average='macro')
        recall = recall_score(label_transf, cm_predictions, average='macro')
        #display_confusion_matrix(cmat, score, precision, recall)

        print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
        print('CLASSES_NAMES GLOBAL')
        print(CLASS_NAMES_GLOBAL[fold_var-1])
        print(CLASS_NAMES_GLOBAL[fold_var-1].shape)
        #VALIDATION_ACCURACY.append(1)
        #VALIDATION_LOSS.append(2)
        plot_confusion_matrix(cmat, classes=CLASS_NAMES_GLOBAL[fold_var-1], title='Matriz de Confusão')
        print(classification_report(label_transf, cm_predictions, target_names=CLASS_NAMES_GLOBAL[fold_var-1]))
        """

        del model
        del history
        tf.keras.backend.clear_session()
        gc.collect()
        tf.compat.v1.reset_default_graph()
        fold_var += 1

def main():
    min_images_per_person = [30]
    models = ["DeepFace"]
    num_folds = 5
    batch_sizes = [60]
    for min_per_person in min_images_per_person:
        for batch in batch_sizes:
            for model in models:
                multi_data = get_mounted_data(min_per_person, min_per_person)
                Y = multi_data[['name']]
                X = multi_data[['image_path']]
                CLASSES = Y.groupby('name').nunique().shape[0]
                print("### run_k_fold ", " min_per_person ", min_per_person, " CLASSES ", CLASSES,
          "model ", model, " batch_size ", batch)
                run_k_fold(multi_data, X, Y, CLASSES, model, batch, num_folds)
                tf.keras.backend.clear_session()
                gc.collect()

    #model = get_model("DeepFace", 34)
    #model.load_weights("model_TFrecordDeepFace_1_60.h5")
    #print(model.summary)

if __name__ == "__main__":
    CLASS_NAMES = []
    main()