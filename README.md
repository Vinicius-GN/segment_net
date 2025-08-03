
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
  <a href="#6-parameters-table-">Parameters Table</a> ·
  <a href="#7-results-and-comparison-">Results and Comparison</a> ·
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

---

### 🎯 Objectives of This Work

🔹 Evaluate **transformer-based segmentation backbones** under off-road conditions.  
🔹 Build a **modular and configurable framework** for segmentation research.  
🔹 Allow easy switching between **backbones, decoders, losses, and attention types**.  
🔹 Provide a **reproducible baseline** using the RELLIS-3D dataset.  
🔹 Facilitate **benchmarking and experimentation** for off-road autonomous navigation.  

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

### 📦 Environment Setup (via shell script)

To automatically install the required packages and create the conda environment:

```bash
bash env/create_env.sh
```

This will:

* Create a `pytorch-env` environment with **Python 3.8.20**
* Install **PyTorch 2.4.1**, **TorchVision**, **TorchAudio**, with **CUDA 12.1**
* Install key libraries (Albumentations, OpenCV, MONAI, Transformers, etc.)
* Install **PyTorch Geometric** and CUDA-compatible wheels

Then activate the environment:

```bash
conda activate pytorch-env
```

---

### 🚀 Running the Pipeline

1. **Choose a config file**
   Use one of the `.ini` files in the `cfg/` folder (e.g., `rellis3d_dev.ini`).

2. **Edit configuration**
   Open the file and update it to accomplish your desired setup:

   * `type` em [BACKBONE] (e.g., segformerb0)
   * `mode` (`train`, `resume`, or `test`)
   * `dataset` (e.g., `rellis3d`, `rugd`, `bdd100k`, etc.)
   * Optional: learning rate, loss function, attention module, batch size, etc.

#### ▶ Training

To train any network, after fulfilling the project's requirements and editing the configuration file, you just need to run the following shell command

```bash
python run.py --cfg cfg/rellis3d_dev.ini
```

Ensure the `.ini` file has `mode = train`.

#### ✅ Resume

In case you stop your network's training and need to recover it where it stopped, just run:
```bash
python run.py --cfg cfg/rellis3d_dev.ini
```

Ensure the `.ini` file has `mode = resume`.

#### 🧪 Testing

For testing any supported network, again, just run:

```bash
python run.py --cfg cfg/rellis3d_dev.ini
```

Ensure the `.ini` file has `mode = test`.

---

## 6. Model Attributes Overview 🧠

Before diving into the performance results, we provide an overview of each model’s computational characteristics. These attributes help assess trade-offs between speed, memory usage, and scalability across real-world deployments (e.g., embedded systems, real-time constraints).

| **Attributes**          | **MobileViT** | **DeiT** | **EfficientFormer** | **LeViT** | **SegFormer** | **PiT** | **SAM 2** |
| ----------------------- | ------------- | -------- | ------------------- | --------- | ------------- | ------- | --------- |
| **Batch Size**          | —             | —        | —                   | —         | —             | —       | —         |
| **# Parameters (M)**    | —             | —        | —                   | —         | —             | —       | —         |
| **Inference Time (ms)** | —             | —        | —                   | —         | —             | —       | —         |

---

---

## 7. Results 🚀

### 7.1 Quantitative Results 📊

Below are the quantitative benchmarks collected across all tested models using our unified framework in Rellis3D datasey. We report per-class IoU scores and overall mean IoU, as well as resource usage metrics to highlight architecture trade‑offs in speed and size.

| **Models**          | *sky* | *grass* | *tree* | *bush* | *concrete* | *mud* | *person* | *puddle* | *rubble* | *barrier* | *log* | *fence* | *vehicle* | *object* | *pole* | *water* | *asphalt* | *building* | **mean** |
|--------------------|:-----:|:------:|:-----:|:-----:|:---------:|:----:|:-------:|:--------:|:--------:|:-------:|:---:|:--------:|:--------:|:--------:|:-----:|:------:|:--------:|:----------:|:--------:|
| **MobileViT**       |       |        |       |       |           |       |         |          |          |         |     |         |          |          |       |        |          |            |          |
| **DeiT**            |       |        |       |       |           |       |         |          |          |         |     |         |          |          |       |        |          |            |          |
| **EfficientFormer** |       |        |       |       |           |       |         |          |          |         |     |         |          |          |       |        |          |            |          |
| **LeViT**           |       |        |       |       |           |       |         |          |          |         |     |         |          |          |       |        |          |            |          |
| **SegFormer**       |       |        |       |       |           |       |         |          |          |         |     |         |          |          |       |        |          |            |          |
| **PiT**             |       |        |       |       |           |       |         |          |          |         |     |         |          |          |       |        |          |            |          |
| **SAM 2**           |       |        |       |       |           |       |         |          |          |         |     |         |          |          |       |        |          |            |          |



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
