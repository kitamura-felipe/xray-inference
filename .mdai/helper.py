from torchvision import transforms
import torch.nn as nn
from torchvision.models import densenet201


def load_model(n_classes, state_dict=None, **kwargs):
    model = densenet201(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def transform_image(img):

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(img).unsqueeze(0)
