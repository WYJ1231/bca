# !/bin/bash -e  --nproc_per_node=8
nohup python infer.py --cfg /cfgs/inference/5hao.toml >> "/nohup_vit_LCR.log" &
