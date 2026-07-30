# Garment Inertial Denoiser: Endowing Accurate Motion Capture via Loose IMU Denoiser

This repository contains the implementation of our paper, *Garment Inertial Denoiser: Endowing Accurate Motion Capture via Loose IMU Denoiser*, including pretrained weights, training scripts, and evaluation code.

[Paper]() | [Project Page]()

- [train_per_imu.py](./train_per_imu.py): trains Location-Specific Denoiser expert modules, each specialized for one IMU placement.
- [train_fuse.py](./train_fuse.py): trains the Cross-wear Fusion model, initialized from the retained Location-Specific Denoiser checkpoints.
- [denoise.py](./denoise.py): generates denoised IMU sequences with the trained model.
- [eval.py](./eval.py): evaluates saved denoised IMU data with MAE.

## Dataset

The GID dataset is available at:

Google Drive: https://drive.google.com/drive/folders/1tp6yjy3AbLiOAsud96pyHmdJqPSqFo6R?usp=sharing

Baidu Cloud: https://pan.baidu.com/s/1LDw5Z8BhQLjhqbE4RepKvg?pwd=7hap


<!-- ## Citation

```bibtex
@article{fang2026garment,
  title={Garment Inertial Denoiser (GID): Endowing Accurate Motion Capture via Loose IMU Denoiser},
  author={Fang, Jiawei and Zheng, Ruonan and Gao, Xiaoxia and Jiang, Shifan and Chen, Anjun and Ye, Qi and Guo, Shihui},
  journal={arXiv preprint arXiv:2601.01360},
  year={2026}
}
``` -->
