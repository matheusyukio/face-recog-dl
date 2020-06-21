from models import FaceNet

def test_model():
	print(FaceNet(2).summary())

if __name__ == '__main__':
	test_model()
	print("Pass")