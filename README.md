![SoFT logo](https://raw.githubusercontent.com/mib-unitn/SoFT/master/logo.png)

<a href="https://ascl.net/2508.008"><img src="https://img.shields.io/badge/ascl-2508.008-blue.svg?colorB=262255" alt="ascl:2508.008" /></a>  ![PyPI - Version](https://img.shields.io/pypi/v/solar_ft?style=plastic)  [![DOI](https://zenodo.org/badge/DOI/p687.svg)](https://doi.org/p687)

Solar granulation plays a fundamental role in convective energy transport and the organization of magnetic fields in the photosphere. Analyzing these dynamic structures at high resolution requires efficient and adaptable processing. **SoFTg (Solar Feature Tracking: granules)** is a fast, multi-instrument hybrid segmentation algorithm within the SOFT library designed to isolate solar granules automatically.

### Pre-processing & Contrast Enhancement

To optimize solar images for segmentation, a custom pipeline is applied:
1. **Temporal Stacking**: Applies a median stack of $N=5$ contiguous frames to eliminate atmospheric noise without blurring boundaries.
2. **CLAHE**: Enhances local feature contrast.
3. **Unsharp Masking**: Subtracts a blurred background to sharpen structural edges.

### Segmentation Phase: Hybrid Approach

SoFTg combines classical heuristic control with deep-learning efficiency:

1. **Adaptive Thresholding**: Computes a local rolling Gaussian threshold based on regional median and standard deviation metrics:
   $$\mathcal{T}=\tilde{I}-k\sqrt{\mu_{sq}-\mu^{2}}$$
2. **Morphological Filtering**: Cleans intergranular lanes using erosion, dilation, and skeletonization to capture complex or exploding granules.
3. **Watershed Segmentation**: Identifies basins with a Euclidean Distance Transform (EDT) and floods them to create preliminary labels.
4. **UNet++ Deep Learning Refinement**: Passes the frames through a UNet++ architecture built with dense skip connections. Trained with a differentiable **Soft-IoU loss function**, the network resolves complex structures, avoids artificial internal cavities, and outputs solid, contiguous granular cells.

### Network Evaluation Metrics

The pipeline evaluates structural network topology by extracting morphological skeletons to mirror true solar configurations:
- **Connected Components**: Discrete count of isolated individual granular cells.
- **Network Length**: Cumulative pixel perimeter of structural boundaries.
- **Junctions**: Intersection nodes serving as a direct proxy for lane interconnectivity.
- **Fragmentation Index**: Normalised unconnected line terminations tracking network splitting.

Further details regarding the SoFTg algorithm can be found in Berretti et al. 2026.

## Installation

Clone the repository and install the framework (switch to the `granules` branch for this tool):

```sh
git clone [https://github.com/mib-unitn/SoFT.git](https://github.com/mib-unitn/SoFT.git)
cd SoFT
git checkout granules
pip install .
```

Usage
Please make sure to cite the corresponding paper if you use SoFTg in your research: Berretti et al. 2026.
