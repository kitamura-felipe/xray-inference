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
        See https://github.com/mdai/model-deploy/blob/master/mdai/server.py for details on the
        schema of `data` and the required schema of the outputs returned by this function.
        """
        input_files = data["files"]

        self.model.eval()

        outputs = []

        for file in input_files:
            if file["content_type"] != "application/dicom":
                continue

            try:
                ds = pydicom.dcmread(BytesIO(file["content"]))
                arr = ds.pixel_array
            except:
                continue

            if arr.dtype != "uint8":
                arr = np.uint8((arr - arr.min()) * (255 / (arr.max() - arr.min())))

            if ds.PhotometricInterpretation == "MONOCHROME1":
                arr = 255 - arr

            dicom_uids = {
                "study_uid": ds.StudyInstanceUID,
                "series_uid": ds.SeriesInstanceUID,
                "instance_uid": ds.SOPInstanceUID,
            }

            # Handle multi-frame instances
            try:
                num_frames = int(ds.NumberOfFrames)
            except:
                num_frames = 0
            is_multi_frame = num_frames > 1 and arr.shape[0] == num_frames

            if is_multi_frame:
                for frame_number in range(num_frames):
                    arr_outputs = self.predict_on_arr(
                        arr[frame_number], dicom_uids, frame_number=frame_number
                    )
                    outputs.extend(arr_outputs)
            else:
                arr_outputs = self.predict_on_arr(arr, dicom_uids)
                outputs.extend(arr_outputs)

            return outputs

    def predict_on_arr(self, arr, dicom_uids, frame_number=None):
        img = Image.fromarray(arr).convert("RGB")
        img = transform_image(img)

        outputs = []

        with torch.set_grad_enabled(False):
            preds = torch.sigmoid(self.model(img.to(self.device)))

        y_prob = preds.cpu().numpy()
        y_classes = y_prob >= 0.5
        class_indices = np.where(y_classes.astype("bool"))[1]

        if len(class_indices) == 0:
            # no outputs, return 'NONE' output type
            output = {
                "type": "NONE",
                "study_uid": dicom_uids.get("study_uid"),
                "series_uid": dicom_uids.get("series_uid"),
                "instance_uid": dicom_uids.get("instance_uid"),
                "frame_number": frame_number,
            }
            outputs.append(output)
        else:
            for class_index in class_indices:
                probability = y_prob[0][class_index]

                gradcam = GradCam(self.model)
                gradcam_output = gradcam.generate_cam(
                    img.to(self.device), arr, class_index
                )
                gradcam_output_buffer = BytesIO()
                gradcam_output.save(gradcam_output_buffer, format="PNG")

                output = {
                    "type": "ANNOTATION",
                    "study_uid": dicom_uids.get("study_uid"),
                    "series_uid": dicom_uids.get("series_uid"),
                    "instance_uid": dicom_uids.get("instance_uid"),
                    "frame_number": frame_number,
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
                outputs.append(output)

        return outputs
