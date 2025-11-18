Detection of Large Woody Debris
Deep learning – U-Net/Deeplabv3+
Author: QIHAN
Created: 18 November 2025
Email: qi.han@polito.it
Aim: Detection of Large Woody Debris (LWD) in braided rivers using RGB-UAV dataset.

Dependencies
1. ArcPy (ArcGIS Pro Python API). Used for image processing, data preprocessing and model inference.
2. RGB-UAV images. High-resolution UAV-acquired images for model training and inference.
3. Annotated samples. Field-survey or manually labeled large woody debris (LWD) samples for training or validation. Annotation formats can be used Shapefile.
4.Pre-trained model. The model_available folder contains a pre-trained model trained on the RGB-UAV dataset. Can be used directly for inference or for transfer learning.
The model and additional image datasets are hosted on Zenodo (https://zenodo.org/uploads/17640738) to avoid GitHub file size limits.
Please refer to the Zenodo link to access all required model files and supplementary data.

Method:
1. Prepare your RGB-UAV images and annotation samples.
2. Use ArcGIS Pro and the provided scripts to run the model.
3. Apply the model from the `model_available` folder: 
   - Direct inference: use the model to detect large woody debris on new RGB images. 
   - Transfer learning: leveraged as a pre-trained model for further training of another model.  And it is recommended to reduce the learning rate to ensure stable convergence.

Notes:
- Make sure to follow the dependencies and requirements before running the scripts.
- The pre-trained model is not included directly in this repository to avoid large file limits; use the Zenodo link provided.
