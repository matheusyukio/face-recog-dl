from dataprocess import create_dataset_tfRecord, get_mounted_data

def test_create_dataset_tfRecord():
	min_images_per_person = [100]
	min_per_person = min_images_per_person[0]
	multi_data = get_mounted_data(min_per_person, min_per_person)
	#Y = multi_data[['name']]
	#X = multi_data[['image_path']]
	#print(Y.head(6))
	#print(X.head(6))
	create_dataset_tfRecord(multi_data, 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/')
	#data_x, data_y = transform_image_dataframe_to_matrix(multi_data, 250, 250, 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/')
	#print(multi_data.head(6))
	#print(data_x.shape)
	#print(data_x[:6])
	#print(data_y.shape)
	#print(data_y[:6])

if __name__ == '__main__':
	test_create_dataset_tfRecord()
	print("Pass")