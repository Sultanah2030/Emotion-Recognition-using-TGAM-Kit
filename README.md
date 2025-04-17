# Emotion-Recognition-using-TGAM-Kit
This project recognizes emotions (happiness and sadness) using EEG signals and machine learning. The main goal of this project is to classify emotions using a single-channel EEG device (TGAM). The project uses signal processing techniques to clean the data and machine learning (SVM classifier) to classify emotions.

## Files Included

1. **NeuroPy Library**:
- This project uses the NeuroPy library. The library is licensed under the Copyright (c) 2013, sahil singh. All rights reserved.
Used to collect and process EEG signals. 
   - [NeuroPy GitHub Repository](https://github.com/lihas/NeuroPy)

2. **Jupyter Notebook for Data Representation (SVM Classifier)**: 
   - The main notebook for running the SVM classifier and training the model using the features extracted from EEG data.
   
3. **Training File (SVM Classifier)**: 
   - Contains the code for training the emotion classification model (SVM) with the preprocessed EEG data.

4. **Preprocessing and Feature Extraction (EEG_processor)**: 
   - This file handles the preprocessing of raw EEG signals (including median filtering, notch filtering, thresholding, and bandpass filtering).
   - It also extracts features such as Power Spectral Density (PSD), Standard Deviation, Mean, and Entropy from the cleaned data for training the model.

5. **Project Report**: 
   - The detailed report that explains the project, the methods used, results, and challenges faced.

6. **Datasets for Training**: 
   - The EEG datasets used to train and test the model, including both raw and filtered EEG data files.

7. **Raw vs Filtered Files**: 
   - Files showing raw and filtered EEG signals to highlight the importance of preprocessing.

# Dataset Information
The dataset used for training is based on EEG signals collected using the TGAM device. It contains two types of emotional states:
- Happy
  
- Sad

# Results
- The trained model achieved an accuracy of 88.64%.

- The happiness state was classified with 96% accuracy.

- The sadness state was classified with 84% accuracy.

- The results are based on a trade-off between the low-cost single-channel EEG device and classification performance.















NeuroPy Library
This project uses the [NeuroPy library](https://github.com/lihas/NeuroPy). The library is licensed under the Copyright (c) 2013, sahil singh. All rights reserved.


