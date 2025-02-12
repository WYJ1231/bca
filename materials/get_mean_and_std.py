import os

from cavasspy.ops import read_cavass_file
from jhammer.statistics import aggregate_mean_std
from tqdm import tqdm

data_dir = "/im0_images"
skn_dir = "/Skn"

mean_list, std_list, num_list = [], [], []

data_lst = []
for each in tqdm(os.listdir(data_dir)):
    each = each[:-4]
    image = read_cavass_file(f"{data_dir}/{each}.IM0")
    skn_label = read_cavass_file(f"{skn_dir}/{each}.BIM").astype(bool)

    skn_region = image[skn_label]
    mean = skn_region.mean()
    std = skn_region.std(ddof=1)
    mean_list.append(mean)
    std_list.append(std)
    num_list.append(skn_region.size)

mean1, std1, num1 = mean_list[0], std_list[0], num_list[0]

for mean, std, num in zip(mean_list[1:], std_list[1:], num_list[1:]):
    mean1, std1 = aggregate_mean_std(mean1, std1, num1, mean, std, num)
    num1 += num

print(mean1, std1)
