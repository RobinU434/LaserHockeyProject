#!/bin/bash

config2class stop-all

config2class to-code --input config/training/train_sac.yaml --output project/utils/configs/train_sac_config.py --init-none
config2class to-code --input config/training/train_dyna.yaml --output project/utils/configs/train_dyna_config.py --init-none
config2class to-code --input config/training/train_md_dyna.yaml --output project/utils/configs/train_md_dyna_config.py --init-none
