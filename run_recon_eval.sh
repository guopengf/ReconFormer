#!/bin/env bash
# ReconFormer Evaluation
python main_recon_test.py --phase test --model ReconFormer --challenge singlecoil --F_path 'path to fastMRI dataset' --test_dataset F --sequence PD --accelerations 8 --center-fractions 0.04 --checkpoint 'path to checkpoint'--verbose
