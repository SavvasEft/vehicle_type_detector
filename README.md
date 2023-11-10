# Identifying Vehicle Types

This project focuses on evaluating an image dataset, and on how to train a Neural Network to identify accurately different types of vehicles, such as Vans, Cabs, SUVs, Sedans etc.
## Demonstrates:

Data-set analysis / Feature engineering / Data cleaning / Automating checks on images 
NN training / Object detection / Model evaluation 


## Short description:

For this project, I modified a dataset (found [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)). This was used to train a NN (YoloV4) to identify car types. The original dataset consisted of images with car models as classes. The three scripts in the repository do the following: 

### 'data_prep. ...'
- download dataset (!!!! DOWNLOADS ~4Gb of data !!!!)
- check and correct duplicates
- correct errors from manual inspection (hard-coded)
- prepare images with bounding boxes for manual inspection
(- analyse pre-defined dataset)
- define new class (vehicle-type)
- prepare test / train datasets (20/80)
- prepare format for training YOLOv5s
    (folder structure / txt files / yaml file / transformed bb coordinates)

### 'veh_type_yolo_training.ipynb ...' (in progress...)
    - train YOLOv4 (using pretrained weights)
    - evaluate results

### 'vehicl_type_predictor.ipynb  ...' (in progress...)
    - accepts images, detects vehicles and identifies vehicle type
