# DeepJSCC‑L++

**Source code for [DeepJSCC‑L++: Robust and Bandwidth‑Adaptive Wireless Image Transmission”](https://ieeexplore.ieee.org/document/10436878), IEEE Globecom 2023, by Chenghong Bian, Yulin Shao, and Deniz Gündüz.

---

## 🚀 Overview

DeepJSCC‑L++ is a vision-transformer-based (**Swin Transformer**) deep joint source-channel coding (JSCC) scheme for **adaptive wireless image transmission**. Notably, it supports multiple **bandwidth ratios** and **channel SNRs** using a single model, achieving near-optimal performance under varying conditions using:

- **Side information** (bandwidth & SNR) fed into the encoder and decoder
- **Dynamic Weight Assignment (DWA)** to balance losses across conditions
- A **Swin Transformer** backbone for robust feature learning.

See the slides for the Globecom conference, `globecom pre.pdf` for more details of the paper.
---

## Training

To train a model, run

```python run_swin_adapt.py --resume False```.

See `get_args.py` file for more hyper parameters. 

The model will be stored in the `models/` folder.




#### Evaluation

To evaluate the performance of a given setting

```python run_swin_adapt.py --resume True```.
