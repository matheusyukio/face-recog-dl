#!/usr/bin/env python
# coding: utf-8

# imports
import os
import gc
import numpy as np
import pandas as pd
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

from dataprocess import dataTrainAugmentation, dataTestAugmentation, dataHoldOutAugmentation, get_mounted_data, directory_mover

from models import create_new_model, DeepFace, LeNet5, AlexNet, VGGFace

from write_plot_history import write_results
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def get_model_name(name, k, batch):
    return 'model_main1' + name + '_' + str(k) + '_' + str(batch) + '.h5'

def get_current_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

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

        directory_mover(training_data,"training_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var))

        directory_mover(validation_data,"validation_data_"+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var))

        train_data_generator = dataTrainAugmentation().flow_from_directory(
            # training_data,
            directory=os.path.join(os.getcwd(), 'new/working/training_data_'+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var)+'/'),
            target_size=(250, 250),
            # x_col = "image_path", y_col = "name",
            batch_size=BATCH_SIZE,
            #subset="training",
            class_mode="categorical",
            shuffle=True)

        valid_data_generator = dataTrainAugmentation().flow_from_directory(
            # training_data,
            directory=os.path.join(os.getcwd(), 'new/working/validation_data_'+MODEL_NAME+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(fold_var)+'/'),
            target_size=(250, 250),
            # x_col = "image_path", y_col = "name",
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            #subset="validation",
            shuffle=True)
        '''
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
        '''
        model = get_model(MODEL, CLASSES)
        # rmsprop = RMSprop(lr=1e-3, decay=1e-6)
        sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        # CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + get_model_name(MODEL_NAME, fold_var, BATCH_SIZE),
                                                        monitor='val_acc', verbose=VERBOSE,
                                                        save_best_only=True, mode='max')
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE, patience=400)

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
                            #steps_per_epoch=train_data_generator.n // train_data_generator.batch_size,
                            callbacks=callbacks_list,
                            validation_data=valid_data_generator,
                            #validation_steps=valid_data_generator.n // valid_data_generator.batch_size,
                            verbose=VERBOSE,
                            #GPU Test luisss
                            max_queue_size=BATCH_SIZE,                # maximum size for the generator queue
                            workers=12,                        # maximum number of processes to spin up when using process-based threading
                            use_multiprocessing=False
                            )

        HISTORY.append(history)

        # LOAD BEST MODEL to evaluate the performance of the model model_"+MODEL_NAME+"_"+str(fold_var)+".h5"
        model.load_weights(
            os.getcwd() + "/models/model_main1" + MODEL_NAME + "_" + str(fold_var) + '_' + str(BATCH_SIZE) + ".h5")

        results = model.evaluate(valid_data_generator)
        # results = model.evaluate_generator(valid_data_generator)
        results = dict(zip(model.metrics_names, results))

        VALIDATION_ACCURACY.append(results['acc'])
        VALIDATION_LOSS.append(results['loss'])

        write_results(
            get_current_time_str() + 'main1_k_fold_' + str(CLASSES) + '_' + MODEL_NAME + '_' + str(EPOCHS) + '_' + str(
                BATCH_SIZE) + '.txt', VALIDATION_ACCURACY, VALIDATION_LOSS, HISTORY)

        Y_pred = model.predict_generator(valid_data_generator, validation_data.shape[0]//BATCH_SIZE + 1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        #print(valid_data_generator.classes)
        cm = confusion_matrix(valid_data_generator.classes, y_pred)

        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
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


        nomes_classes = []
        for i in pd.DataFrame(Y.groupby('name')['name'].nunique().reset_index(name="unique"))[
            'name']:  # Y.groupby('name').nunique()['name']:
            nomes_classes.append(str(i))
        plot_confusion_matrix(cm, classes=nomes_classes,
                              title='Matriz de Confusão')
        #print(classification_report(valid_data_generator.classes, y_pred, target_names=nomes_classes))

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
    elif model_name == "VGGFace":
        return VGGFace(num_classes)


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
    min_images_per_person = [30,25]#[30,25]  # [25,20]
    models = ["LeNet5","VGGFace"]#,"AlexNet","DeepFace","VGGFace"]
    num_folds = 5

    #aumentando o batch para 30 DeepFace conseguiu bons resultados, testar com outras
    batch_sizes = [30,60]#[2,4,8,30]
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
                run_k_fold(multi_data, X, Y, CLASSES, epoch, model, batch, num_folds)

if __name__ == "__main__":
    main()