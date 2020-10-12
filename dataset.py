import pydicom
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InferenceXRayDataset(Dataset):
    classes = {
        "Abdomen": 0,
        "Ankle": 1,
        "Arm": 2,
        "Cavum": 3,
        "Cervical Spine": 4,
        "Chest": 5,
        "Clavicles": 6,
        "Elbow": 7,
        "Feet": 8,
        "Finger": 9,
        "Forearm": 10,
        "Hand": 11,
        "Hip": 12,
        "Knee": 13,
        "Lower Leg": 14,
        "Lumbar Spine": 15,
        "Others": 16,
        "Pelvis": 17,
        "Sacroiliac": 18,
        "Shoulder": 19,
        "Sinus": 20,
        "Skull": 21,
        "Thigh": 22,
        "Thoracic Spine": 23,
        "Wrist": 24
    }

    def __init__(self, dcm_fpaths, transform):
        self.dcm_fpaths = dcm_fpaths
        self.transform = transform

    @staticmethod
    def load_input_image(fpath):
        ds = pydicom.dcmread(fpath)
        arr = ds.pixel_array

        # Normalize images not having values in [0,255]
        if arr.dtype != "uint8":
            arr = np.uint8((arr - arr.min()) * (255 / (arr.max() - arr.min())))

        if ds.PhotometricInterpretation == "MONOCHROME1":
            arr = 255 - arr
            
        img = Image.fromarray(arr).convert("RGB")
        return ds.SOPInstanceUID, img

    def __len__(self):
        return len(self.dcm_fpaths)

    def __getitem__(self, item):
        fpath = self.dcm_fpaths[item]
        sop_instance_uid, img = self.load_input_image(fpath)
        input_img = self.transform(img)
        return sop_instance_uid, input_img
