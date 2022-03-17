# Nuclei Segmentation Model (Python-DL)
Nuclei segmentation performed on the digitized H&amp;E-stained images of whole slide images (WSI). It is a deep learning model (DL) based on a GAN architecture.

## Pre-requsites
- Linux (Tested on windows10, CUDA10.2)
- NVIDIA GPU (Tested on Nvidia TianX 1080 x 12 on local workstations)
- Python (3.5+), matplotlib (3.1.1), numpy (1.17.3), opencv-python (4.1.1.26), pillow (6.2.1), PyTorch (1.5+), scikit-learn (0.22.1), scipy (1.3.1), torchvision (0.4.2), tensorboardx (optional)

All directly running to automatically install all depending libraries.
```bash
pip install -r requirement.txt 
```

## How to run
Model checkpoint is save in model folder. To run the nuclei pixel prediction, simply run

```python
python main_PredviaTrainGenerator.py
```

The input image for nuclei pixel prediction should be put into data folder. Different image format handling wrappers were also provided by main_PredviaTrainGenerator_IMAGEEXTENSION.py, change the IMAGEEXTENSION with the provided extension names and place the corresponding images into data folder.
