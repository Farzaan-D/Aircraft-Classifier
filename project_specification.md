# Aircraft Classifier: Technical Project Specification & Architecture Document

---

## 1. Problem Statement

### 1.1 The Challenge of Fine-Grained Aircraft Classification
In standard computer vision, image classification models are tasked with distinguishing coarse, highly distinct categories (e.g., distinguishing a *cat* from a *dog*, or a *car* from an *airplane*). In contrast, **Fine-Grained Visual Categorization (FGVC)** focuses on distinguishing between highly similar sub-classes belonging to the exact same super-category.

For aircraft, this task presents distinct real-world challenges:
* **High Intra-Class Variance:** The exact same aircraft model (e.g., *Boeing 737-800*) can appear in hundreds of different commercial airline liveries, varied paint schemes, day/night lighting, extreme angles, airport tarmacs, or in-flight skies.
* **Low Inter-Class Variance:** Different aircraft models often share near-identical airframes. For instance, distinguishing an *Airbus A319* from an *Airbus A320*, or a *Boeing 737-700* from a *Boeing 737-800*, relies on subtle geometric differences such as fuselage length, the count and placement of overwing emergency exit doors, engine cowling contours, or wingtip winglet designs.
* **Complex Background Noise:** In unconstrained real-world imagery, aircraft are surrounded by busy airport infrastructure—runway markings, baggage loaders, pushback tugs, boarding bridges, and hangar buildings—which can mislead standard neural networks.

### 1.2 Practical Real-World Importance
* **Air Traffic Control (ATC) & Smart Airports:** Automating apron management, taxiway tracking, gate scheduling, and verifying wake-turbulence separation categories (which depend on exact aircraft weight and variant).
* **Defense & Airspace Sovereignty:** Automated, instant identification of aircraft manufacturer, model, and operational capabilities from ground cameras, radar-slaved optical systems, and aerial reconnaissance drones.
* **Aviation Maintenance & Logistics:** Automated visual inventory tracking and inspection logs during airline turnaround operations.

---

## 2. Dataset & Data Engineering

### 2.1 Benchmark Dataset: Oxford FGVC-Aircraft (`fgvc-aircraft-2013b`)
The model was trained on the benchmark **FGVC-Aircraft** dataset curated by the Visual Geometry Group (VGG) at the University of Oxford.

* **Total Dataset Size:** ~10,000 images (~2.7 GB).
* **Class Uniformity:** Exactly 100 images per variant across 100 variants, ensuring no class-imbalance bias.
* **Annotation Depth:** Every sample contains:
  1. High-resolution RGB photograph.
  2. Bounding box coordinates: `[xmin, ymin, xmax, ymax]`.
  3. 3-tier hierarchical taxonomy annotations.

### 2.2 Hierarchical Data Structure
Aircraft naturally organize into a 3-tier taxonomic hierarchy:
```
Manufacturer (Maker)   -->  30 Classes  (e.g., Boeing, Airbus, Embraer, Bombardier)
      └── Model Family -->  70 Classes  (e.g., Boeing 737, Airbus A320, CRJ)
            └── Model Variant --> 100 Classes (e.g., 737-700, 737-800, A319, A320)
```

### 2.3 Dataset Split Strategy
The dataset was split using the official Oxford benchmark partition:
* **Training Set (`train`):** **3,334 images** (~33.34%) — ~33 images per variant.
* **Validation Set (`val`):** **3,333 images** (~33.33%) — ~33 images per variant.
* **Test Set (`test`):** **3,333 images** (~33.33%) — ~34 images per variant.

### 2.4 Preprocessing & Normalization Pipeline
* **Input Spatial Dimensions:** $224 \times 224$ pixels.
* **Format:** 3-channel RGB float tensors normalized using standard ImageNet parameters:
  $$\text{Mean } \mu = [0.485, 0.456, 0.406], \quad \text{Std } \sigma = [0.229, 0.224, 0.225]$$

---

## 3. System Architecture & Model Design

The project uses a **multi-stage, multi-task deep learning architecture** designed for localization, fine-grained hierarchical classification, and visual interpretability.

```
┌─────────────────────────────────────────────────────────────┐
│                        Raw Input Image                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Localization & Cropping                            │
│ Model: Ultralytics YOLOv8n (COCO Class 4: Airplane)        │
│ Purpose: Isolate airframe and discard background clutter   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Deep Feature Extraction Backbone                   │
│ Model: ConvNeXt-Tiny (Pretrained on ImageNet-1K via timm)   │
│ Latent Output: 768-dimensional visual feature vector        │
└──────────────────────────────┬──────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Stage 3a:        │ │ Stage 3b:        │ │ Stage 3c:        │
│ Maker Head       │ │ Family Head      │ │ Variant Head     │
│ Linear(768, 30)  │ │ Linear(768, 70)  │ │ Linear(768, 100) │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Explainability Engine (Grad-CAM)                   │
│ Target Layer: ConvNeXt Stage 4 final convolutional block    │
│ Purpose: Visualize key features (engines, winglets, tail)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Interactive Presentation Layer: Gradio Web UI (Port 7860)   │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Component Breakdown

#### A. Localization Engine: YOLOv8-Nano
* **Role:** Acts as an automated attention crop. Instead of passing an entire busy airport image to the classifier, YOLOv8n identifies the plane boundary box and crops it.
* **Parameter Footprint:** 3.2 Million parameters.

#### B. Classification Backbone: ConvNeXt-Tiny
* **Role:** Modern pure convolutional neural network incorporating Vision Transformer advancements (depthwise separable convolutions, $7 \times 7$ kernels, inverted bottleneck design, and GELU activations).
* **Feature Dimension:** 768 latent dimensions.
* **Parameter Footprint:** 27.97 Million parameters.

#### C. Multi-Task Classification Heads
Three parallel linear heads branch from the shared 768-dimensional feature representation:
1. `head_maker`: `nn.Linear(768, 30)`
2. `head_family`: `nn.Linear(768, 70)`
3. `head_variant`: `nn.Linear(768, 100)`

#### D. Explainability Engine: Grad-CAM
* **Role:** Gradient-weighted Class Activation Mapping computes gradients of the target class score flowing into the final convolutional feature maps (`classifier.backbone.stages[-1].blocks[-1]`), producing a 2D attention heatmap showing which physical airframe components drove the prediction.

### 3.2 CNN vs. Vision Transformer (ViT) Benchmark
A comparative hardware benchmark was performed on Apple M1 Pro silicon to evaluate whether a CNN or Transformer should serve as the primary classifier:

| Backbone Model | Architecture Type | Parameters | Inference Speed (Batch 16) | Latency per Image | Architectural Selection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ConvNeXt-Tiny** | Modern CNN | **27.97 M** | **76.82 FPS** | ~13.0 ms | **Selected** (Strong spatial inductive bias preserves localized structural edges) |
| **ViT-Tiny** | Vision Transformer | **5.56 M** | **266.24 FPS** | ~3.75 ms | High speed, but requires larger pretraining volume for subtle geometry |

---

## 4. End-to-End Image Pipeline Walkthrough

When a user uploads an image into the web interface, the application executes the following deterministic 6-step pipeline:

```
[Upload Image] ──> [YOLOv8 Detection] ──> [BBox Crop] ──> [Tensor Transform] ──> [ConvNeXt Inference] ──> [Grad-CAM & Results]
```

1. **Image Ingestion:** The user uploads any arbitrary image format (JPEG, PNG, WebP) via the Gradio interface.
2. **Object Detection & Localization:**
   * The raw PIL image is passed through `yolov8n.pt`.
   * The model filters for COCO Class index `4` (*airplane*).
   * If multiple aircraft are detected, it selects the bounding box with the highest confidence score.
   * *Graceful Fallback:* If no aircraft is detected (e.g., an extreme close-up of a cockpit or tail), the system logs a warning and processes the entire original image.
3. **Region-of-Interest (ROI) Cropping:** The image is cropped strictly around the airframe coordinates `(x1, y1, x2, y2)`, eliminating background tarmac, sky, and airport equipment.
4. **Tensor Preprocessing:**
   * The cropped image is resized to $224 \times 224$ pixels.
   * Converted to a PyTorch float tensor and normalized using ImageNet channel mean and standard deviation.
   * Moved to the active acceleration device (Apple Silicon `mps` or NVIDIA `cuda`).
5. **Hierarchical Forward Pass:**
   * The tensor passes through `ConvNeXt-Tiny`, generating a 768-dimensional feature embedding.
   * The embedding feeds simultaneously into `head_variant`, `head_family`, and `head_maker`.
   * A Softmax function converts raw logits into calibrated probability distributions.
   * Class index mappings resolve the top prediction and confidence score for all three tiers.
6. **Grad-CAM Attention Map Synthesis:**
   * Gradients for the predicted variant are computed with respect to the last convolutional block.
   * Activations are averaged and mapped via OpenCV Jet colormap (`cv2.applyColorMap`).
   * The colored heatmap is overlaid on the original cropped image at 50% opacity.
7. **UI Presentation:** The web interface renders:
   * Cropped aircraft image.
   * Text card: Manufacturer, Family, and Variant with percentage confidences.
   * Visual Grad-CAM heatmap highlighting attention points.

---

## 5. Training Details, Hyperparameters & Results

### 5.1 Training Configuration
* **Total Epochs Trained:** **10 Epochs**
* **Batch Size:** 32
* **Optimizer:** AdamW (`lr = 1e-4`, `weight_decay = 1e-4`)
* **Hardware:** Apple M1 Pro (Metal Performance Shaders - `mps`)
* **Model Checkpointing:** Evaluated at the end of each epoch; checkpoints saved only when validation variant accuracy improved (`best_model.pth`).

### 5.2 Hierarchical Loss Formulation
Instead of standard single-label Cross-Entropy, the model was trained using a **hierarchical multi-task loss function**:

$$\mathcal{L}_{\text{total}} = \gamma \mathcal{L}_{\text{variant}} + \beta \mathcal{L}_{\text{family}} + \alpha \mathcal{L}_{\text{maker}}$$

Where:
* $\mathcal{L}_{\text{variant}} = \text{CrossEntropy}(\text{pred}_{\text{variant}}, y_{\text{variant}})$
* $\mathcal{L}_{\text{family}} = \text{CrossEntropy}(\text{pred}_{\text{family}}, y_{\text{family}})$
* $\mathcal{L}_{\text{maker}} = \text{CrossEntropy}(\text{pred}_{\text{maker}}, y_{\text{maker}})$
* **Loss Weights:** $\gamma = 1.0$ (fine-grained priority), $\beta = 0.5$ (family constraint), $\alpha = 0.5$ (maker constraint).

*Why this works:* Penalizing maker and family classification errors forces the shared feature extractor to learn broader aerodynamic and brand features before attempting to separate fine variants.

### 5.3 Training & Validation Results

| Epoch | Train Loss | Train Variant Acc | Val Loss | Val Variant Acc | Val Family Acc | Val Maker Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 8.1980 | 0.90% | 8.1028 | 1.02% | 7.98% | 21.99% |
| 3 | 4.2150 | 42.10% | 4.1200 | 38.50% | 52.40% | 68.20% |
| 5 | 1.0540 | 85.20% | 2.1104 | 67.80% | 78.40% | 85.10% |
| 8 | 0.0820 | 99.10% | 1.3200 | 76.50% | 85.20% | 90.50% |
| **10** | **0.0041** | **100.00%** | **1.2250** | **77.92%** | **86.41%** | **91.78%** |

* **Final Maker Accuracy:** **91.78%**
* **Final Family Accuracy:** **86.41%**
* **Final Variant Accuracy:** **77.92%**

---

## 6. Future Scalability & Recommended Architectural Changes

If this system is scaled into an industrial or enterprise-grade deployment (e.g., nationwide defense network, commercial airport surveillance, or global airline operations), the following architectural changes are recommended:

### 6.1 Architectural Change 1: Cascaded / Conditional Hierarchical Heads
* **Current Limitation:** The 3 heads currently operate independently in parallel from the same 768-dim vector. This allows mathematically inconsistent predictions (e.g., predicting Maker: *Airbus*, but Variant: *Boeing 737-800*).
* **Future Architecture:** Implement **Conditional Probability Tree Conditioning**:
  $$P(\text{Variant} \mid x) = P(\text{Maker} \mid x) \times P(\text{Family} \mid \text{Maker}, x) \times P(\text{Variant} \mid \text{Family}, x)$$
  The output embedding of the Maker head can be concatenated or cross-attended with the Family head, guaranteeing 100% taxonomic consistency.

### 6.2 Architectural Change 2: Domain-Specific Fine-Tuned Instance Segmentation
* **Current Limitation:** The current YOLOv8n model is pretrained on standard COCO classes and produces rectangular bounding boxes that still include background runway asphalt.
* **Future Architecture:**
  * Fine-tune **YOLOv8-Seg** or **Mask R-CNN** directly on aircraft polygonal masks.
  * Extract pixel-perfect airframe segmentation masks and zero out (blacken) all tarmac/sky background pixels, feeding pure aircraft silhouettes into the classifier.

### 6.3 Architectural Change 3: Part-Based Attention & Multi-Crop Feature Fusion
* **Current Limitation:** The model currently consumes a single downscaled $224 \times 224$ global view of the aircraft. Crucial details (e.g., number of pitot tubes, exact engine blade chevrons, cockpit window angles) can be blurred.
* **Future Architecture:** Adopt a **Part-Based Attention Network (P-Net)**:
  * A detection network identifies specific aircraft components: (1) Cockpit Nose, (2) Engine Nacelles, (3) Wingtips, (4) Tailfin.
  * High-resolution crops of each component are passed into specialized sub-networks, and their feature vectors are concatenated via cross-attention.

### 6.4 Architectural Change 4: Transition to Self-Supervised Vision Foundation Models
* **Current Limitation:** ConvNeXt-Tiny was initialized with ImageNet-1K supervised weights.
* **Future Architecture:**
  * Upgrade to **DINOv2 (ViT-Base or ViT-Large)** or **OpenCLIP**.
  * DINOv2 learns self-supervised visual features with spatial understanding of object parts and depth without requiring human labels, making it suitable for fine-grained aerospace recognition.

### 6.5 Architectural Change 5: Production Serving & Edge Infrastructure
* **Inference Optimization:**
  * Export PyTorch models to **ONNX Runtime** and **TensorRT** (for NVIDIA GPU servers) or **CoreML** (for Apple Silicon devices).
  * Apply INT8 / FP16 quantization to reduce model footprint from 120 MB to < 35 MB, driving latency down from 13 ms to < 3 ms per frame.
* **Serving Stack:**
  * Decouple the UI from the compute engine: Replace monolithic Gradio with a high-performance **FastAPI / Triton Inference Server** backend and a reactive React/Next.js frontend.
  * Deploy horizontally across Kubernetes clusters with GPU auto-scaling for live airport camera RTSP video streaming.

---

## 7. Summary Table of Specifications

| Category | Specification Details |
| :--- | :--- |
| **Project Name** | Aircraft Classifier AI with Hierarchical Multi-Task Learning & XAI |
| **Dataset** | Oxford FGVC-Aircraft (`fgvc-aircraft-2013b`) — ~10,000 images, ~2.7 GB |
| **Taxonomy Levels** | 30 Makers $\rightarrow$ 70 Families $\rightarrow$ 100 Variants |
| **Dataset Split** | Train: 3,334 (33.3%) \| Val: 3,333 (33.3%) \| Test: 3,333 (33.4%) |
| **Detection Backbone** | Ultralytics YOLOv8n (3.2M parameters) |
| **Classification Backbone** | ConvNeXt-Tiny via `timm` (27.97M parameters, 768-dim features) |
| **Explainable AI (XAI)** | Grad-CAM on ConvNeXt Stage 4 final convolutional block |
| **Training Duration** | 10 Epochs |
| **Optimization** | AdamW (`lr = 1e-4`, `weight_decay = 1e-4`), Batch Size: 32 |
| **Loss Function** | Hierarchical Loss: $1.0 \times \mathcal{L}_{\text{variant}} + 0.5 \times \mathcal{L}_{\text{family}} + 0.5 \times \mathcal{L}_{\text{maker}}$ |
| **Final Performance** | Maker Acc: **91.78%** \| Family Acc: **86.41%** \| Variant Acc: **77.92%** |
| **UI & Tech Stack** | Python, Gradio, PyTorch, torchvision, timm, OpenCV, Pillow |
