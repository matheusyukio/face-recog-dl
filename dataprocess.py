import os
import pandas as pd
import shutil
import tensorflow as tf

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