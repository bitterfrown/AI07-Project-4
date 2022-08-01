# Project-4: Semantic Segmentation for Images Containing Cell Nuclei

## 1.0 Introduction
This project is carried out to create a computer model that can identify a range of nuclei across varied conditions. The data has been obtained from [here](https://www.kaggle.com/competitions/data-science-bowl-2018/code)

## 2.0 Requirements
The project is built with Spyder. Multiple libraries are imported such as Tensorflow, Numpy, Matplotlib, OpenCv and Scikit Learn

## 3.0 Methodology
### 3.1 Data Loading
The data has folders which contains a train folder for training data and test folder for test data. In these folders, there are images of the cell used for inputs and masks used for labels. The inputs are preprocessed with feature scaling while the labels are encoded to a numeric coding with binary values. The train and test data is split to a ratio of 80:20. 

### 3.2 Model Building
A pre-trained model, MobileNetV2 is imported and applied to the model as a feature extraction layer. The model is also designed according to the U-Net architecture. It has two parts known as the downward stack and the upsampling path. The downward stack functions as a feature extractor whereas the upsampling path will double the dimension of inputs.

### 3.3 Model training
The model has been trained with a batch size of 16 for 100 epochs. Earlystopping is also applied in training the model. 
