# Open Area Segmentation in CT Images Based on Pixel Displacement and Multi-View Ensemble with Application in the Axillary and Lower Cervical Regions
## Introduction
This study focuses on the segmentation of open regions. Unlike organs with well-defined boundaries, open regions typically lack clear boundaries, making the segmentation process more complex. Therefore, this study aims to propose a new segmentation algorithm to improve the accuracy and reliability of open region segmentation.
## Methodology
This study proposes an algorithm that combines pixel displacement techniques with a multi-view integration framework. By leveraging the strengths of multiple views, it accurately captures the complex structures of open regions, thereby achieving more accurate and robust segmentation results than existing methods.
## Result
We selected the best metrics for each category of methods (based on Dice scores) to compare with our method. Since the results of the 2D methods were relatively low, they are not shown here.
| Task | Method | Parameters(MB) | DSC | HD95 | ASD |
|:---:|:---:|:---:|:---:|:---:|:---:|
|the Axillary Region| 2.5D | 132.43 | 0.8864 | 9.05 | 2.51 |
|the Axillary Region| 3D | 333.95 | 0.8869 | 7.71 | 1.83 |
|the Axillary Region| Ours | 42.84 | 0.8999 | 5.49 | 1.61 
|the Lower Cervical Region| 2.5D | 298.19  | 0.7907 | 8.98 | 2.41 |
|the Lower Cervical Region| 3D | 333.95 | 0.7443 | 11.10 | 2.44 |
|the Lower Cervical Region| Ours | 42.84 | 0.8114 | 8.10 | 1.30 |
## Environmrnt 
- Ubuntu 20.04.6
- Python 3.8
- Pytorch 1.13.1+cu116
- CUDA 10.1
- GCC 7.5.0
- G++ 7.5.0
## Usage
The relevant model code is located in the file ./jhammer/models/unet.py.
