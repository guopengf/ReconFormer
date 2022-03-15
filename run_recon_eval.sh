#!/bin/env bash
# ReconFormer Evaluation
python main_recon_test.py --phase test --model ReconFormer --challenge singlecoil --F_path /mnt/2t/MR_data/fastMRI_preprocessed --test_dataset F --sequence PD --accelerations 8 --center-fractions 0.04 --checkpoint /home/pengfei/code/exp_recon_seg/media_release/F_RTransv2_1206_ms_recatt_8/checkpoint.pth --verbose
