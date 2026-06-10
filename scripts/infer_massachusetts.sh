#!/usr/bin/env bash
python tools/infer_rtrnet.py --config configs/config_massachusetts.yaml --checkpoint checkpoints/rtrnet.pth --save_dir outputs/massachusetts_rtr
python tools/eval_qatf.py --config configs/config_massachusetts.yaml --rtr_dir outputs/massachusetts_rtr --save_dir outputs/massachusetts_qatf
