from models import VGGFace

def test_model():
	print(VGGFace(2).summary())

if __name__ == '__main__':
	test_model()
	print("Pass")