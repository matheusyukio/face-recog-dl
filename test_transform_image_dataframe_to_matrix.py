from dataprocess import transform_image_dataframe_to_matrix, get_mounted_data

def test_transform_image_dataframe_to_matrix():
	min_images_per_person = [5]
	min_per_person = min_images_per_person[0]
	multi_data = get_mounted_data(min_per_person, min_per_person)
	#Y = multi_data[['name']]
	#X = multi_data[['image_path']]
	#print(Y.head(6))
	#print(X.head(6))
	data_x, data_y = transform_image_dataframe_to_matrix(multi_data, 250, 250, 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/')
	#print(multi_data.head(6))
	#print(data_x.shape)
	#print(data_x[:6])
	#print(data_y.shape)
	#print(data_y[:6])

if __name__ == '__main__':
	test_transform_image_dataframe_to_matrix()
	print("Pass")