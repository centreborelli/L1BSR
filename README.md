# L1BSR: Exploiting Detector Overlap for Self-Supervised SISR of Sentinel-2 L1B Imagery

[Ngoc Long Nguyen](https://ngoclongct.github.io/), [Jérémy Anger](https://github.com/kidanger/), [Axel Davy](http://dev.ipol.im/~adavy/), [Pablo Arias](http://dev.ipol.im/~pariasm/), [Gabriele Facciolo](http://gfacciol.github.io/)

Centre Borelli, ENS Paris-Saclay

---

[![arXiv](https://img.shields.io/badge/paper-arxiv-brightgreen)](https://arxiv.org/pdf/2304.06871.pdf)
[![Zenodo](https://img.shields.io/badge/L1BSR%20dataset-Zenodo-9cf)](https://zenodo.org/record/7826696)
[![IPOL Demo](https://img.shields.io/badge/demo-IPOL-blueviolet)](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000471)
[![Project](https://img.shields.io/badge/project%20web-github.io-red)](https://centreborelli.github.io/L1BSR/)

This repository is the official PyTorch implementation of L1BSR: Exploiting Detector Overlap for Self-Supervised SISR of Sentinel-2 L1B Imagery (**Best Student Paper at EarthVision 2023**).

L1BSR produces a 5m high-resolution (HR) output with all bands correctly registered from a single 10m low-resolution (LR) Sentinel-2 L1B image with misaligned bands. Note that L1BSR is trained on real data with self-supervision, i.e. without any ground truth HR targets.

![](https://github.com/centreborelli/L1BSR/blob/docs/docs/resources/L1BSR_teaser.png)

## Contents

1. [Overview](#Overview)
1. [Testing](#Testing)
1. [Training](#Training)
1. [Citation](#Citation)
1. [License and Acknowledgement](#License-and-Acknowledgement)

### Overview

There are two key modules integral to the training of the L1BSR:

1. The REConstruction (**REC**) module: performs joint super-resolution and band-alignment for the L1B BGRN data.
1. The Cross-Spectral Registration (**CSR**) module: produces a dense flow between 2 images of different spectral bands.

Both modules are trained with self-supervision. Note that the **CSR** is used only during the training of L1BSR, whereas at inference, only the **REC** is needed.

![](https://github.com/centreborelli/L1BSR/blob/docs/docs/resources/L1BSR_framework.png)

### Testing

For your convenience we provide some test BGRN images (~10Mb) in `/examples`.

If you want a quick inspection of our two key modules **REC** and **CSR**, checkout our IPOL demo [![IPOL Demo](https://img.shields.io/badge/demo-IPOL-blueviolet)](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000471)

We also provide the testing code `main.py`. Like in the demo, you can choose the task (super-resolution or cross-spectral registration) for our networks (**REC** or **CSR**, respectively) to perform.

Examples:

```bash
# Super-resolution: This code below super-resolves (x2) the image in "examples/00.tif"
# and saves it in "output.tif".
python main.py examples/00.tif output.tif --device cuda --task superresolution
# Cross-spectral registration: This code below aligns the bands Blue, Red, and NIR of
# the image in "examples/00.tif" to its Green band and saves the output in "output.tif".
python main.py examples/00.tif output.tif --device cuda --task registration
```

### Training

The training codes for both the **CSR** and **REC** modules will be soon available. Stay tuned!

### Citation

```
@inproceedings{nguyen2023l1bsr,
    title={L1BSR: Exploiting Detector Overlap for Self-Supervised Single-Image Super-Resolution of Sentinel-2 L1B Imagery},
    author={Nguyen, Ngoc Long and Anger, J{\'e}r{\'e}my and Davy, Axel and Arias, Pablo and Facciolo, Gabriele},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={2012--2022},
    year={2023}
    }
```

### License and Acknowledgement

This project is released under the GPL-3.0 license. The codes are based on [RCAN](https://github.com/yulunzhang/RCAN). Please also follow their licenses. Thanks for their awesome works.
