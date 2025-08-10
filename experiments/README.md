<h1 align="center">
Experiments – Transformer Architectures Comparative Configurations
</h1>

<p align="center">
This folder contains the configuration files used in the comparative analysis of nine transformer-based and hybrid architectures implemented in our semantic segmentation framework.  
All experiments were conducted using the RELLIS-3D dataset, following the same pipeline, training parameters, and evaluation procedures.
</p>

<p align="center">
  <a href="#1-overview-">Overview</a> ·
  <a href="#2-usage-">Usage</a> ·
  <a href="#3-configuration-mapping-">Configuration Mapping</a>
</p>

---

## 1. Overview 📊

Each `.ini` file corresponds to a **single backbone configuration** used for benchmarking.
They define all necessary parameters for training or evaluating the model, including backbone type, decoder, attention module, loss function, optimizer settings, and dataset paths.

To switch between **training** and **testing**, simply change the `mode` parameter within the file:

```ini
mode = train   # for training the model  
mode = test    # for evaluating the model
```

This approach ensures reproducibility and keeps configurations modular and adaptable.

---

## 2. Usage ⚙️

Example: training using the **MobileViT** configuration:

```bash
python run.py --cfg experiments/rellis3d_5090_mobilevit.ini
```

Example: evaluating the same model after training:

```bash
python run.py --cfg experiments/rellis3d_5090_mobilevit.ini
```

> Just make sure `mode = test` is set within the `.ini` file.

---

## 3. Configuration Mapping 🗂️

The table below maps each configuration file to its corresponding model and includes links to the original paper presentations:

| **Configuration file**              | **Model** (Paper Link)                                                     |
| ----------------------------------- | -------------------------------------------------------------------------- |
| `rellis3d_5090_mobilevit.ini`       | [MobileViT](https://arxiv.org/abs/2110.02178)                              |
| `rellis3d_5090_maxxvitv2.ini`       | [MaxViT](https://arxiv.org/abs/2204.01697)                                 |
| `rellis3d_5090_efficientformer.ini` | [EfficientFormer](https://arxiv.org/abs/2206.01191)                        |
| `rellis3d_5090_tinyvit.ini`         | [TinyViT](https://arxiv.org/abs/2207.10666)                                |
| `rellis3d_5090_segformerb0.ini`     | [SegFormer](https://arxiv.org/abs/2105.15203)                              |
| `rellis3d_5090_pitxs.ini`           | [PiT (Pooling-based Vision Transformer)](https://arxiv.org/abs/2103.16302) |
| `rellis3d_5090_sam2_hiera.ini`      | [SAM 2 (Segment Anything Model 2)](https://arxiv.org/abs/2408.00714)       |
| `rellis3d_5090_fastvit.ini`         | [FastViT](https://arxiv.org/abs/2303.14189)                                |
| `rellis3d_5090_edgenext.ini`        | [EdgeNeXt](https://arxiv.org/abs/2206.10589)                               |

