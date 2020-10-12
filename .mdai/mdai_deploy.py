import os
import sys
from io import BytesIO
import pydicom
import torch
import numpy as np
from PIL import Image
from helper import load_model, transform_image

MODEL_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, MODEL_PATH)
from vis.gradcam import GradCam


class MDAIModel:
    def __init__(self):
        root_path = os.path.dirname(os.path.dirname(__file__))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            gpu_ids = []
        state_dict = torch.load(
            os.path.join(root_path, "models", "trained_models", "densenet201.pth")
        )
        self.model = load_model(25, state_dict)
        self.model = self.model.to(self.device)

    def predict(self, data):
        """
        The input data has the following schema:

        {
            "instances": [
                {
                    "file": "bytes"
                    "tags": {
                        "StudyInstanceUID": "str",
                        "SeriesInstanceUID": "str",
                        "SOPInstanceUID": "str",
                        ...
                    }
                },
                ...
            ],
            "args": {
                "arg1": "str",
                "arg2": "str",
                ...
            }
        }

        Model scope specifies whether an entire study, series, or instance is given to the model.
        If the model scope is 'INSTANCE', then `instances` will be a single instance (list length of 1).
        If the model scope is 'SERIES', then `instances` will be a list of all instances in a series.
        If the model scope is 'STUDY', then `instances` will be a list of all instances in a study.

        The additional `args` dict supply values that may be used in a given run.

        For a single instance dict, `files` is the raw binary data representing a DICOM file, and
        can be loaded using: `ds = pydicom.dcmread(BytesIO(instance["file"]))`.

        The results returned by this function should have the following schema:

        [
            {
                "type": "str", // 'NONE', 'ANNOTATION', 'IMAGE', 'DICOM', 'TEXT'
                "study_uid": "str",
                "series_uid": "str",
                "instance_uid": "str",
                "frame_number": "int",
                "class_index": "int",
                "data": {},
                "probability": "float",
                "explanations": [
                    {
                        "name": "str",
                        "description": "str",
                        "content": "bytes",
                        "content_type": "str",
                    },
                    ...
                ],
            },
            ...
        ]

        The DICOM UIDs must be supplied based on the scope of the label attached to `class_index`.
        """
        input_instances = data["instances"]
        results = []
        self.model.eval()
        for instance in input_instances:
            tags = instance["tags"]
            try:
                ds = pydicom.dcmread(BytesIO(instance["file"]))
                arr = ds.pixel_array
            except:
                continue

            if arr.dtype != "uint8":
                arr = np.uint8((arr - arr.min()) * (255 / (arr.max() - arr.min())))

            if ds.PhotometricInterpretation == "MONOCHROME1":
                arr = 255 - arr

            img = Image.fromarray(arr).convert("RGB")
            img = transform_image(img)

            with torch.set_grad_enabled(False):
                outputs = self.model(img.to(self.device))
                preds = torch.sigmoid(outputs)
            y_prob = preds.cpu().numpy()
            y_classes = y_prob >= 0.5
            class_indices = np.where(y_classes.astype("bool"))[1]
            if len(class_indices) == 0:
                # no outputs, return 'NONE' output type
                result = {
                    "type": "NONE",
                    "study_uid": tags["StudyInstanceUID"],
                    "series_uid": tags["SeriesInstanceUID"],
                    "instance_uid": tags["SOPInstanceUID"],
                    "frame_number": None,
                }
                results.append(result)
            else:
                for class_index in class_indices:
                    probability = y_prob[0][class_index]

                    gradcam = GradCam(self.model)
                    gradcam_output = gradcam.generate_cam(
                        img.to(self.device), arr, class_index
                    )
                    gradcam_output_buffer = BytesIO()
                    gradcam_output.save(gradcam_output_buffer, format="PNG")

                    result = {
                        "type": "ANNOTATION",
                        "study_uid": tags["StudyInstanceUID"],
                        "series_uid": tags["SeriesInstanceUID"],
                        "instance_uid": tags["SOPInstanceUID"],
                        "frame_number": None,
                        "class_index": int(class_index),
                        "data": None,
                        "probability": float(probability),
                        "explanations": [
                            {
                                "name": "Grad-CAM",
                                "content": gradcam_output_buffer.getvalue(),
                                "content_type": "image/png",
                            },
                        ],
                    }
                    results.append(result)
            return results
