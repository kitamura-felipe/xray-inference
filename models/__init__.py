import torch.nn as nn

from pretrainedmodels import inceptionv4
from torchvision.models import alexnet, densenet201, resnet101


class ModelLoaders:
    @classmethod
    def load_alexnet(cls, num_classes):
        model = alexnet(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    @classmethod
    def load_densenet(cls, num_classes):
        model = densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    @classmethod
    def load_inceptionv4(cls, num_classes):
        model = inceptionv4(num_classes=1000, pretrained="imagenet")
        model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
        return model

    @classmethod
    def load_resnet101(cls, num_classes):
        model = resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


def load_model(model_name, n_classes, state_dict=None, **kwargs):
    try:
        model_loader = getattr(ModelLoaders, "load_{}".format(model_name))
    except AttributeError:
        raise Exception("Unavailable model '{}'".format(model_name))

    model = model_loader(n_classes, **kwargs)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model
