import argparse
import numpy as np
import os
import pandas as pd
import random
import torch
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import InferenceXRayDataset
from models import load_model
from tqdm import tqdm


def set_seeds(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed + 1)
    random.seed(seed + 2)


def load_config(fpath, assert_keys=None):
    with open(fpath) as f:
        config = yaml.safe_load(f)

    if assert_keys is not None:
        unavailable_keys = list(filter(lambda k: k not in config, assert_keys))
        if len(unavailable_keys) > 0:
            raise KeyError("The following keys are missing in config file: '{}'".format(unavailable_keys))

    return config


def load_dcm_fpaths(data_path):
    if not os.path.isdir(data_path):
        raise NotADirectoryError("Directory does not exist: '{}'".format(data_path))

    fpaths = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith(".dcm"):
                fpaths.append(os.path.join(dirpath, filename))

    return fpaths


def run_inference(model, dataloader, device):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    progress_bar = tqdm(dataloader, desc="Running inference")
    model.eval()

    predictions = []
    for i, (sop_instance_uids, inputs, _) in enumerate(progress_bar):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            preds = torch.sigmoid(outputs)
            predictions.extend(zip(list(sop_instance_uids), list(preds)))

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_fpath", required=True)
    args = parser.parse_args()

    assert_keys = [
        "model_name",
        "model_state_dict",
        "input_size",
        "data_path"
        "batch_size",
        "save_path"
    ]
    config = load_config(args.config_fpath, assert_keys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(config["model_state_dict"])
    model = load_model(config["model_name"], state_dict)

    transform = transforms.Compose([
        transforms.Resize(config["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dcm_fpaths = load_dcm_fpaths(config["data_path"])
    dataset = InferenceXRayDataset(dcm_fpaths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, worker_init_fn=set_seeds)

    predictions = run_inference(model, dataloader, device)

    data = [[id_] + [p.item() for p in preds] for id_, preds in predictions]
    columns = ["sop_instance_uid"] + [cls for cls, i in dataset.classes.items()]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(config["save_path"], "test_predictions.csv"), index=False)
