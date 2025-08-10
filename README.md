
<h1 align="center">
Segment net - 2D Image Segmentation Framework with Custom Architectures
</h1>
  
<p align="center">
This repository presents a flexible and modular framework for <strong>semantic segmentation</strong> using various <strong>transformer-based</strong>, <strong>convolutional and hybrid backbones</strong>.  
It supports multiple attention mechanisms, decoder designs, loss functions, and real-world datasets (urban and off-road), enabling thorough experimentation and benchmarking.
</p>

<p align="center">
  <a href="#1-introduction-and-goals-">Introduction and Goals</a> ·
  <a href="#2-project-structure-">Project Structure</a> ·
  <a href="#3-architecture-details-">Architecture Details</a> ·
  <a href="#4-datasets-">Datasets</a> ·
  <a href="#5-installation-and-usage-">Installation and Usage</a> ·
  <a href="#6-model-attributes-overview-">Model Attributes Overview</a> ·
  <a href="#7-results-">Results</a> ·
  <a href="#8-contribution-">Contribution</a> ·
  <a href="#9-license-">License</a>
</p>

<p align="center">
  <img src="images/model.svg" alt="Architecture Diagram" width="600">
</p>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/cuda-12.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![Torch](https://img.shields.io/badge/pytorch-2.4.1-orange)](https://pytorch.org/)
[![License](https://img.shields.io/github/license/iag0g0mes/segment_net.svg)](LICENSE)

</div>

---

## 1. Introduction and Goals 📊

Semantic segmentation is a core task for autonomous vehicle perception systems. In real-world applications, the ability to understand and classify each region of an image is crucial for safe and autonomous navigation, especially in unstructured environments.

In off-road scenarios, perception systems must deal with:

🔹 Irregular terrains and unstructured surfaces (mud, grass, rubble).  
🔹 Ambiguous class boundaries and visual noise.  
🔹 High intra-class variability and illumination changes.  
🔹 Severe class imbalance in available datasets (e.g., RELLIS-3D, RUGD).  

Transformers have emerged as powerful alternatives to traditional CNNs, offering better global context modeling and higher segmentation accuracy. However, their behavior in off-road segmentation tasks is still underexplored.

To validate the effectiveness of the proposed framework and architectures in real-world off-road scenarios, we conducted extensive experiments using the **RELLIS-3D dataset**. Quantitative and qualitative results, including per-class performance metrics and visual segmentation outputs, are presented in the *Results* section below.

---

### 🎯 Objectives of This Work

🔹 Evaluate **transformer-based segmentation backbones** under off-road conditions.  
🔹 Build a **modular and configurable framework** for segmentation research.  
🔹 Allow easy switching between **backbones, decoders, losses, and attention types**.  
🔹 Provide a **reproducible baseline** using the RELLIS-3D dataset.  
🔹 Facilitate **benchmarking and experimentation** for off-road autonomous navigation.  
🔹 Report and analyze experimental results obtained using the **RELLIS-3D dataset**, providing both quantitative benchmarks and qualitative visual comparisons.

---

## 2. Project Structure 📂

The repository is organized as follows:

```
SEGMENT_NET/
│
├── cfg/              # Configuration files (.ini) for training/testing
├── env/              # Environment setup scripts
├── images/           # Architecture diagrams
├── logs/             # Training logs
├── params/           # Model parameters, checkpoints, or derived configs
├── utils/            # Helper functions and support scripts
│
├── class_weights.py  # Class balancing script
├── run.py            # Main pipeline script
├── run_all.sh        # Shell script to enable automatic training sequences
├── .gitignore
└── README.md        
```

---

## 3. Architecture Details 🧠

<p align="center">
  <img src="https://github.com/iag0g0mes/segment_net/blob/main/images/model.svg" alt="Model Architecture" width="600">
</p>

### 🔩 Supported Backbones

<div align="justify">
The proposed segmentation framework was designed to be modular, extensible, and experiment-friendly. It supports a broad range of architectural components, enabling researchers and developers to combine different backbones, decoders, attention mechanisms, feature aggregation strategies, and loss functions with ease. This flexibility allows for systematic exploration of design choices and facilitates fair comparisons across models and datasets, particularly in challenging off-road scenarios.

</div>

#### 🧱 Convolutional Backbones

* `resnet18`
* `mobilenetv3`
* `efficientnetb0`
* `deeplabv3_mobilenetv3`
* `convnextv2`

#### 🧠 Transformer-based Backbones

* `deit3_small`
* `sam2_hiera`
* `pitxs`
* `segformerb0`
* `levit`
* `tinyvit`
* `fastvit`

#### ⚡ Hybrid Backbones (Conv + Transformer)

* `mobilevit`
* `efficientformer`
* `maxxvitv2`
* `edgenext`
  
---

### 🧱 Feature Pyramid Aggregation (FPN)

* `sum`
* `concat`
* `weighted_sum`
* `max_pool`

---

### 🎯 Class-wise Attention Mechanisms

* `none`
* `spatial`
* `query`
* `class_channel`
* `se_channel`

---

### 🧠 Decoder Variants

* `se_conv_interp`
* `depthwise_nn`
* `transformer`

---

### 📉 Supported Loss Functions

* `dice`
* `focal_dice`
* `cross_entropy`
* `focal_cross_entropy`
* `lovasz_softmax`
* `boundary_dice`
* `hausdorff_dt_dice`

---

## 4. Datasets 🌍

This segmentation framework offers built-in support for several standard public datasets, covering both urban and off-road scenarios, enabling fair comparison and flexible experimentation. Configuration files (e.g. in `cfg/`) can simply reference any of these datasets to run training, validation, or testing.

| Dataset                                                                            | Environment                                        | Annotations                                    | Key Features                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[A2D2 (Audi Autonomous Driving Dataset)](https://a2d2.audi/)**                   | Diverse urban + highway (Germany)                  | RGB + LiDAR + 2D/3D semantic masks             | \~41 k segmented images (41 labels); includes both semantic and 3D box annotations. Multi-sensor platform with 5 LiDARs and 6 cameras                                                         |
| **[RELLIS‑3D](https://github.com/unmannedlab/RELLIS-3D)**                          | Off‑road tracks and terrain                        | RGB images + LiDAR scans with per‑pixel labels | 6 235 labeled frames (from \~13 k synchronized LiDAR+camera); 19 semantic classes including grass, sky, rubble, and vehicle. Auto‑focus on class‑imbalance and irregular terrain         |
| **[RUGD (Robot Unstructured Ground Driving)](https://rugd.vision/)**               | Natural off‑road (trails, parks, creeks, villages) | RGB images with semantic masks                 | 7 436 images and 24 classes (e.g., tree, fence, vehicle, puddle, gravel, concrete). Split: \~4,779 train / 1,964 test / 733 val                                                       |
| **[GOOSE (German Outdoor and Offroad Dataset)](https://goose-dataset.de/)**        | Unstructured outdoor robotics environments         | RGB + NIR images and annotated point clouds    | 10 000+ paired image and LiDAR frames; supports fine-grained class ontology over unstructured terrain. Includes open-source tools and evaluation challenges                          |
| **[BDD100K (Berkeley DeepDrive)](https://bair.berkeley.edu/blog/2018/05/30/bdd/)** | Diverse urban drives (USA)                         | 1280×720 RGB with pixel-level segmentation     | 10 semantic instance segmentation classes (e.g., car, pedestrian, truck). \~10K annotated images; diverse weather, lighting, and traffic scenes                           |

---

## 5. Installation and Usage ⚙️

### 📋 Requirements

This project requires a **Python 3.8** environment with **PyTorch 2.4.1** and **CUDA 12.1**. All core dependencies and libraries are installed via a single automated setup script. The main packages used include:

* `torch==2.4.1` · `torchvision==0.19.1` · `torchaudio==2.4.1`
* `albumentations`, `transformers`, `monai`, `timm`
* `OpenCV`, `Matplotlib`, `Scikit-learn`, `Plotly`, `Dash`
* `PyTorch Geometric` (with CUDA wheels)

Ensure you have **Anaconda/Miniconda** installed and a compatible NVIDIA GPU with CUDA 12.1 support for full functionality.

---

### 📦 Environment Setup (via shell script)

To automatically install the required packages and create the environment:

```bash
bash env/create_env.sh
```

This will:

✅ Create a conda environment named **`pytorch-env`**  
✅ Install **Python 3.8.20**, **PyTorch 2.4.1**, **TorchVision**, **TorchAudio** (CUDA 12.1)  
✅ Install required libraries via `pip` (Albumentations, MONAI, Transformers, etc.)  
✅ Install **PyTorch Geometric** with precompiled CUDA wheels  

Then activate the environment:

```bash
conda activate pytorch-env
```

---

### 🚀 Running the Pipeline

1. **Choose a config file**
   Use one of the `.ini` files in the `cfg/` folder (e.g., `rellis3d_dev.ini`).

2. **Edit configuration**
   Update the following parameters inside the file:

   * `type` (backbone type, e.g., `segformerb0`)
   * `mode` (`train`, `resume`, or `test`)
   * `dataset` (e.g., `rellis3d`, `rugd`, `bdd100k`)
   * Optionally: loss function, learning rate, attention, batch size, etc.

#### ▶ Training

```bash
python run.py --cfg cfg/rellis3d_dev.ini
```

Ensure `mode = train` in the `.ini` file.

#### ✅ Resume Training

```bash
python run.py --cfg cfg/rellis3d_dev.ini
```

Ensure `mode = resume` to continue from the last checkpoint.

#### 🧪 Testing

```bash
python run.py --cfg cfg/rellis3d_dev.ini
```

Ensure `mode = test` in the configuration file.

---

## 6. Model Attributes Overview 🧠

| **Model**       | **Batch size** | **# Params (MM)** | **Inference time (ms)** | **Features vector (len)** |
|-----------------|---------------:|------------------:|------------------------:|--------------------------:|
| MobileViT       | 32             | 19.22             | 0.0245                  | 6                         |
| MaxVit          | 16             | 116.62            | 0.0031                  | 5                         |
| EfficientFormer | 16             | 5.26              | 0.0132                  | 4                         |
| TinyViT         | 32             | 12.41             | 0.0007                  | 5                         |
| SegFormer       | 32             | 4.53              | 0.0029                  | 4                         |
| PiT             | 64             | 12.30             | 0.0015                  | 4                         |
| SAM 2           | 32             | 28.37             | 0.0280                  | 5                         |
| FastVit         | 32             | 11.84             | 0.0007                  | 5                         |
| EdgeNeXt        | 32             | 38.99             | 0.0008                  | 5                         |

---

## 7. Results 🚀

### 7.1 Quantitative Results 📊

| **Models**          |        *sky*       |       *grass*      |       *tree*       |       *bush*       |     *concrete*     |        *mud*       |      *person*      |      *puddle*      |      *rubble*      |      *barrier*     | *log* |       *fence*      |      *vehicle*     |      *object*      |       *pole*       |       *water*      |      *asphalt*     |     *building*     |      **mIoU**      |     **Acc (%)**    |
| ------------------- | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :---: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: | :----------------: |
| **MobileViT**       |  0.961 **\[best]** |        0.862       |        0.695       |        0.675       |        0.753       |        0.300       |  0.771 **\[best]** |        0.595       |        0.297       |        0.214       | 0.000 |  0.297 **\[best]** |  0.355 **\[best]** |  0.245 **\[best]** |  0.106 **\[best]** | 0.000 **\[worst]** | 0.000 **\[worst]** |  0.104 **\[best]** |        0.402       |        89.68       |
| **MaxViT**          |        0.961       |  0.879 **\[best]** |        0.736       |        0.727       |        0.764       |  0.343 **\[best]** |        0.639       |        0.664       |  0.498 **\[best]** |        0.204       | 0.000 |        0.205       |        0.244       |        0.121       |        0.106       |  0.230 **\[best]** |        0.006       |        0.032       |  0.409 **\[best]** |  91.02 **\[best]** |
| **EfficientFormer** |        0.954       |        0.854       |        0.708       |        0.674       |        0.745       |        0.319       |        0.561       |        0.560       |        0.088       |        0.183       | 0.000 |        0.185       |        0.179       |        0.003       |        0.061       |        0.000       |        0.004       |        0.044       |        0.340       |        89.42       |
| **TinyViT**         |        0.953       |        0.871       |        0.706       |        0.698       |        0.739       |        0.333       | 0.423 **\[worst]** |  0.710 **\[best]** |        0.345       | 0.150 **\[worst]** | 0.000 | 0.005 **\[worst]** |        0.112       | 0.000 **\[worst]** |        0.064       |        0.000       |        0.000       | 0.007 **\[worst]** |        0.340       |        90.08       |
| **SegFormer**       |        0.961       |        0.846       |  0.748 **\[best]** |        0.639       |        0.753       |        0.295       |        0.626       |        0.633       |        0.210       |        0.164       | 0.000 |        0.173       |        0.239       |        0.002       |        0.083       |        0.000       |        0.019       |        0.042       |        0.357       |        89.38       |
| **PiT**             | 0.952 **\[worst]** |        0.864       |        0.697       |        0.678       | 0.700 **\[worst]** | 0.291 **\[worst]** |        0.505       |        0.625       |        0.136       |        0.181       | 0.000 |        0.144       | 0.104 **\[worst]** |        0.018       |        0.038       |        0.000       |        0.003       |        0.029       |        0.331       |        89.40       |
| **SAM 2**           |        0.960       | 0.719 **\[worst]** | 0.666 **\[worst]** | 0.545 **\[worst]** |  0.783 **\[best]** |        0.302       |        0.540       | 0.443 **\[worst]** |        0.319       |        0.207       | 0.000 |        0.098       |        0.128       |        0.000       | 0.000 **\[worst]** |        0.000       |  0.067 **\[best]** |        0.026       | 0.322 **\[worst]** | 83.90 **\[worst]** |
| **FastViT**         |        0.956       |        0.873       |        0.723       |        0.709       |        0.716       |        0.322       |        0.502       |        0.646       | 0.027 **\[worst]** |        0.191       | 0.000 |        0.127       |        0.128       |        0.038       |        0.000       |        0.000       |        0.000       |        0.012       |        0.332       |        90.38       |
| **EdgeNeXt**        |        0.955       |        0.878       |        0.743       |  0.732 **\[best]** |        0.762       |        0.311       |        0.582       |        0.708       |        0.373       |  0.229 **\[best]** | 0.000 |        0.203       |        0.212       |        0.127       |        0.074       |        0.000       |        0.002       |        0.016       |        0.384       |        91.01       |

---

## 7.2 Qualitative Results 🌟

Visual inspection of segmentation outputs is crucial to understanding how each model handles complex scenes, boundaries, and class ambiguity—especially in challenging terrain.

<div align="justify"> The mosaic below shows representative <strong>input frames</strong>, <strong>ground-truth masks</strong>, and <strong>model predictions</strong> side by side for a range of architectural backbones. This qualitative comparison highlights typical failure modes: boundary misclassifications, small-object omission, and visual artifacts. Mosaic-style layouts are common in semantic segmentation literature, such as those exploring textured mosaic analysis, serving as effective summary visuals for multi-class environments. 
</div>

<p align="center">
  <img src="images/results/mosaic.png" alt="Qualitative segmentation results mosaic" width="800">
</p>

---

## 8. Contribution 🤝

Contributions are welcome!
If you have suggestions, feature requests, or improvements, feel free to:

* Open an [Issue](https://github.com/your-repo/issues)
* Submit a [Pull Request](https://github.com/your-repo/pulls)

---

## 9. License 📜

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.
