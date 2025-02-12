# !/bin/bash
CONFIG_FILE="/cfgs/training/5hao.toml"
LABEL=$(python -c "import toml; print(toml.load('$CONFIG_FILE')['label'])")
TASK=$(python -c "import toml; print(toml.load('$CONFIG_FILE')['task'])")
SNAPSHOT_DIR="/running/bca${TASK}"

mkdir -p ${SNAPSHOT_DIR}

nohup python train.py --cfg "${CONFIG_FILE}" >> "${SNAPSHOT_DIR}/train_output_${LABEL}_C.log" &
