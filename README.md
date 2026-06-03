![SoFT logo](https://raw.githubusercontent.com/mib-unitn/SoFT/master/logo.png)

<a href="https://ascl.net/2508.008"><img src="https://img.shields.io/badge/ascl-2508.008-blue.svg?colorB=262255" alt="ascl:2508.008" /></a>  ![PyPI - Version](https://img.shields.io/pypi/v/solar_ft?style=plastic)  [![DOI](https://zenodo.org/badge/DOI/p687.svg)](https://doi.org/p687)

[cite_start]Solar granulation plays a fundamental role in convective energy transport and the organization of magnetic fields in the photosphere[cite: 27, 28, 29]. [cite_start]Analyzing these dynamic structures at high resolution requires efficient and adaptable processing[cite: 54]. [cite_start]**SoFTg (Solar Feature Tracking: granules)** is a fast, multi-instrument hybrid segmentation algorithm within the SOFT library designed to isolate solar granules automatically[cite: 14, 15, 60].

### Pre-processing & Contrast Enhancement

[cite_start]To optimize solar images for segmentation, a custom pipeline is applied[cite: 84]:
1. [cite_start]**Temporal Stacking**: Applies a median stack of $N=5$ contiguous frames to eliminate atmospheric noise without blurring boundaries[cite: 85, 88, 89].
2. [cite_start]**CLAHE**: Enhances local feature contrast[cite: 85].
3. [cite_start]**Unsharp Masking**: Subtracts a blurred background to sharpen structural edges[cite: 86].

### Segmentation Phase: Hybrid Approach

[cite_start]SoFTg combines classical heuristic control with deep-learning efficiency[cite: 15]:

1. [cite_start]**Adaptive Thresholding**: Computes a local rolling Gaussian threshold based on regional median and standard deviation metrics[cite: 91, 92, 93]:
   $$\mathcal{T}=\tilde{I}-k\sqrt{\mu_{sq}-\mu^{2}}$$
2. [cite_start]**Morphological Filtering**: Cleans intergranular lanes using erosion, dilation, and skeletonization to capture complex or exploding granules[cite: 107, 108, 109].
3. [cite_start]**Watershed Segmentation**: Identifies basins with a Euclidean Distance Transform (EDT) and floods them to create preliminary labels[cite: 110].
4. [cite_start]**UNet++ Deep Learning Refinement**: Passes the frames through a UNet++ architecture built with dense skip connections[cite: 81, 118]. [cite_start]Trained with a differentiable **Soft-IoU loss function**, the network resolves complex structures, avoids artificial internal cavities, and outputs solid, contiguous granular cells[cite: 123, 181, 219, 220].

### Network Evaluation Metrics

[cite_start]The pipeline evaluates structural network topology by extracting morphological skeletons to mirror true solar configurations[cite: 196]:
- [cite_start]**Connected Components**: Discrete count of isolated individual granular cells[cite: 197].
- [cite_start]**Network Length**: Cumulative pixel perimeter of structural boundaries[cite: 198].
- [cite_start]**Junctions**: Intersection nodes serving as a direct proxy for lane interconnectivity[cite: 199].
- [cite_start]**Fragmentation Index**: Normalised unconnected line terminations tracking network splitting[cite: 200].

Further details regarding the SoFTg algorithm can be found in [Berretti et al. [cite_start]2026](DRAFT VERSION JUNE 3, 2026).

## Installation

[cite_start]Clone the repository and install the framework (switch to the `granules` branch for this tool)[cite: 225]:

```sh
git clone [https://github.com/mib-unitn/SoFT.git](https://github.com/mib-unitn/SoFT.git)
cd SoFT
git checkout granules
pip install .
```

Usage
Please make sure to cite the corresponding paper if you use SoFTg in your research: [Berretti et al. 2026](DRAFT VERSION JUNE 3, 2026).
