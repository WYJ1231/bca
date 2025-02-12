from jhammer.io import read_txt_2_list, write_list_2_txt
import random


label = "LCR"
all_data = read_txt_2_list(f"/{label}_samples")
random.shuffle(all_data)

n_val_samples = 10

val_samples = all_data[:n_val_samples]

training_samples = all_data[n_val_samples:]

write_list_2_txt(f"/{label}_training_samples.txt", training_samples)
write_list_2_txt(f"/{label}_val_samples.txt", val_samples)