def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
 for file in valid_files:
        file_name = os.path.split(file)[1]
        label = int(os.path.basename(os.path.split(file)[0]))

        if images % 1000 == 0:
            index = images // 1000
            if current != index:
                current = index
                record_file = directory + '/' + f'{index:05}' + '.tfrecord'
                if writer:
                    writer.close()
                print('{} images'.format(images))
                print('New file: ', record_file)
                writer = tf.python_io.TFRecordWriter(record_file)
        try:
            img = Image.open(file)
            img = img.resize((image_dim, image_dim), Image.ANTIALIAS).convert('RGB').tobytes()
            if len(img) != image_dim * image_dim * 3:
                print('Something is wrong with this image: {}'.format(file))
                continue

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature(label),  
                        'image_raw': _bytes_feature(img)
                    }))
            writer.write(example.SerializeToString())
            images += 1
        except Exception as e:
            print('Ignored image: ' + file)
            print(e)
if writer:
    writer.close()
