#!/bin/env bash
# ReconFormer x4
python main_recon.py --phase train --model ReconFormer --epochs 50 --challenge singlecoil --bs 4 --F_path 'path to fastMRI dataset' --train_dataset F --test_dataset F --sequence PD --accelerations 4 --center-fractions 0.08 --lr 0.0002 --lr-step-size 5 --lr-gamma 0.9 --save_dir /home/pengfei/F_ReconRelease --verbose
# ReconFormer x8
#python main_recon.py --phase train --model ReconFormer --epochs 50 --challenge singlecoil --bs 4 --F_path 'path to fastMRI dataset' --train_dataset F --test_dataset F --sequence PD --accelerations 8 --center-fractions 0.04 --lr 0.0002 --lr-step-size 5 --lr-gamma 0.9 --save_dir /home/pengfei/F_ReconRelease --verbose
