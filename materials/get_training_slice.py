import os

from jhammer.io import write_list_2_txt


label = "AxR"
path = f"/slices/{label}"


write_list_2_txt(f"/dataset/{label.lower()}_training_samples.txt", os.listdir(path))