# WDTFuSR

<p align="center">
  <b>A Fusion-Guided Infrared Image Super-Resolution Network with Wavelet Modulation and Dense Transformer</b>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/task-guided%20infrared%20SR-2f6f9f"></a>
  <a href="#"><img src="https://img.shields.io/badge/backbone-Swin%20Transformer-4c7c59"></a>
  <a href="#"><img src="https://img.shields.io/badge/module-wavelet%20modulation-8b5cf6"></a>
  <a href="#"><img src="https://img.shields.io/badge/framework-PyTorch-ee4c2c"></a>
</p>

WDTFuSR is a fusion-guided infrared image super-resolution framework designed to recover high-resolution thermal details from a low-resolution infrared input with the help of a registered high-resolution visible image. The project targets the practical bottlenecks of infrared imaging: weak texture, low contrast, sensor noise, cross-modal discrepancy, and information loss in deep reconstruction networks.

This repository is organized from the WDTFuSR research materials of Shaohua Yang, Xingang Mou, Xiao Zhou, and Dongming Wang. It extends the SwinFuSR-style guided SR codebase with wavelet feature modulation and dense attention-driven feature preservation for infrared-visible reconstruction.

## Overview

![WDTFuSR architecture](images/wdtfusr_architecture.png)

Given a low-resolution infrared image and a high-resolution visible guidance image, WDTFuSR first purifies and enhances infrared features in the frequency domain, then performs dual-stream deep feature extraction and cross-domain fusion, and finally reconstructs the high-resolution infrared result.

Core ideas:

- Wavelet Transform Feature Modulation Block (WTFMB): decomposes shallow infrared features into frequency sub-bands, suppresses low-frequency non-uniform noise, and enhances mid/high-frequency target edges.
- Residual Dense Channel Attention Group (RDCAG): keeps sparse infrared structure flowing through deeper layers with dense feature reuse and attention-based recalibration.
- Attention-guided Cross-domain Fusion Module (ACFM): adaptively transfers useful visible textures into the infrared stream while reducing cross-modal contamination.
- Random modality dropout: improves robustness when the visible guidance image is missing, occluded, or unreliable.

## WTFMB Detail

![WTFMB detail](images/wtfmb_detail.png)

The WTFMB uses local detail and structural guidance branches before the DWT/IDWT modulation path. The frequency-domain branch strengthens directional structures and suppresses noisy low-frequency interference before cross-modal fusion.

## Wavelet Band Enhancement

![Wavelet band enhancement](images/wavelet_bands.png)

The wavelet modulation branch explicitly separates approximation and directional detail components. This makes the network less dependent on raw visible textures and helps the infrared stream enter fusion with cleaner structural features.

## Quantitative Results

The following results are from the CIDIS-Test setting in the manuscript materials at x4 magnification.

| Method | Scale | PSNR | SSIM |
| --- | ---: | ---: | ---: |
| Bicubic | x4 | 31.76 | 0.8971 |
| SRCNN | x4 | 33.17 | 0.9161 |
| FSRCNN | x4 | 33.14 | 0.9115 |
| VDSR | x4 | 34.56 | 0.9365 |
| EDSR | x4 | 35.28 | 0.9453 |
| SwinIR | x4 | 34.66 | 0.9410 |
| DRCT | x4 | 35.71 | 0.9477 |
| SwinFuSR | x4 | 35.92 | 0.9512 |
| **WDTFuSR (Ours)** | **x4** | **36.49** | **0.9552** |

Ablation on CIDIS-Test:

| WTFMB | RDCAG | PSNR |
| :---: | :---: | ---: |
| - | - | 35.92 |
| yes | - | 36.12 |
| - | yes | 36.28 |
| yes | yes | **36.49** |

## Dataset Notes

The manuscript experiments use CIDIS with the following split:

| Split | Number of pairs/images | Usage |
| --- | ---: | --- |
| Train | 700 pairs | model optimization |
| Validation | 200 pairs | model selection and validation |
| Test | 100 images | final evaluation |

Additional tests on RoadScene and TNO are used to examine generalization under different infrared-visible scene distributions.

For the released training scripts, update the dataset paths in `options/*.json` before running:

- `datasets/train/dataroot_lr`
- `datasets/train/dataroot_guide`
- `datasets/train/dataroot_gt`
- `datasets/validation/dataroot_lr`
- `datasets/validation/dataroot_guide`
- `datasets/validation/dataroot_gt`

## Environment

```bash
conda create --name WDTFuSR python=3.9
conda activate WDTFuSR

# Install PyTorch according to your CUDA version.
# Example for CUDA 11.7:
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt
```

## Training

Small x8 guided SR configuration:

```bash
python main_train_SwinFuSR.py --opt options/train_baseline.json
```

Large x8 guided SR configuration:

```bash
python main_train_SwinFuSR.py --opt options/train_final.json
```

Random visible-modality dropout for robustness:

```bash
python main_train_SwinFuSR.py --opt options/train_augmentation.json
```

x16 guided SR configuration:

```bash
python main_train_SwinFuSR.py --opt options/train_x16.json
```

Fine-tuning from pretrained weights:

```bash
python main_train_SwinFuSR.py --opt options/train_baseline_finetune.json
```

Set `path/pretrained_netG` in the corresponding JSON file before fine-tuning or testing.

## Testing

```bash
python test_SwinFuSR.py --opt options/test_swinFuSR.json
```

Important testing fields:

- Set `path/pretrained_netG` to the checkpoint path.
- Use `without_GT: false` when ground-truth infrared images are available and metrics should be computed.
- Use `without_GT: true` when only inference outputs are needed.

## Repository Structure

```text
WDTFuSR/
├── data/                    # guided infrared-visible dataset loader
├── images/                  # architecture and README figures
├── models/
│   ├── network_swinfusionSR.py
│   ├── model_plain.py
│   └── loss*.py
├── options/                 # training, testing, fine-tuning configs
├── utils/                   # image, logging, option, metric utilities
├── main_train_SwinFuSR.py
├── test_SwinFuSR.py
└── requirements.txt
```

## Key Configuration Fields

| Field | Meaning |
| --- | --- |
| `scale` | Super-resolution factor configured by the experiment. |
| `n_channels_lr` | Number of infrared input channels. |
| `n_channels_guide` | Number of visible guidance channels. |
| `netG/net_type` | Network entry, currently `swinfusionSR`. |
| `netG/embed_dim` | Feature embedding dimension. |
| `netG/Ex_depths` | Depth of the infrared/visible extraction stage. |
| `netG/Fusion_depths` | Depth of the cross-domain fusion stage. |
| `netG/Re_depths` | Depth of the reconstruction stage. |
| `datasets/train/proba_without_rgb` | Probability of dropping visible guidance during training. |
| `train/G_optimizer_lr` | Initial learning rate. |
| `train/G_scheduler_milestones` | Learning-rate decay milestones. |

## Citation

If this repository is useful for your work, please cite the WDTFuSR manuscript when it becomes available.

```bibtex
@misc{yang2026wdtfusr,
  title  = {WDTFuSR: A Fusion-Guided Infrared Image Super-Resolution Network with Wavelet Modulation and Dense Transformer},
  author = {Yang, Shaohua and Mou, Xingang and Zhou, Xiao and Wang, Dongming},
  year   = {2026}
}
```

## Acknowledgements

This project is built on the guided thermal image super-resolution implementation style of SwinFuSR and is also inspired by prior work on SwinFusion and CoReFusion. We thank the authors of these open-source projects and related infrared-visible fusion studies for their valuable contributions to the community.
