from jhammer.io import read_txt_2_list
from cavasspy.ops import read_cavass_file
import os
import pandas as pd

from tqdm import tqdm

data_path = "/W-DS1"
label = "LCR"

samples = read_txt_2_list(f"/{label.lower()}_samples")
missing_label_samples = {}

labels = ["LCR"]

subject_lst, inferior_lst, superior_lst = [], [], []
for ct_name in tqdm(samples):
    for label in labels:
        missing_label_samples[label] = []
        label_file = os.path.join(data_path, ct_name, f"{ct_name}_{label}.BIM")
        if not os.path.exists(label_file):
            print(f"label: {label}. label file is missing: {label_file}")
            missing_label_samples[label].append(ct_name)
        label_data = read_cavass_file(label_file).astype(bool)
        inferior, superior = None, None
        for i in range(0, label_data.shape[2]):
            if inferior is None and label_data[...,i].sum() > 0:
                inferior = i
                break

        for i in range(label_data.shape[2] - 1, 0, -1):
            if superior is None and label_data[...,i].sum() > 0:
                superior = i
                break
        subject_lst.append(ct_name)
        inferior_lst.append(inferior)
        superior_lst.append(superior)
    data = {"subject": subject_lst, "inferior": inferior_lst, "superior": superior_lst}

    df = pd.DataFrame(data)
    df.to_csv(f"/{label}_boundary.csv", index=False)