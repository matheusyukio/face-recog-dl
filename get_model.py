from models import create_new_model, DeepFace, LeNet5, AlexNet, VGGFace, VGGFaceHalf
from inception import Inception

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
    elif model_name == "VGGFaceHalf":
        return VGGFace(num_classes)
    elif model_name == "Inception":
        return Inception(num_classes)