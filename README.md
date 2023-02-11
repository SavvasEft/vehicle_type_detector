# Identifying Vehicle Types

## Demonstrates:

Object detection / NN training / Data cleaning / Feature engineering / Data-set analysis / Automated checks 

## Objective: 

Develop a model that could accurately identify different types of cars, such as Vans, Cabs, SUVs, Sedans etc.

## Short description:

For this project, I modified a dataset (found ([here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)) to have the desired classes. I used this to train a NN (YoloV4) to identify car types. The original dataset consisted of images with car models as classes. The two scripts in the repository do the following: 

### 'data_prep. ...'
- download dataset (!!!! DOWNLOADS ~4Gb of data !!!!)
- check and correct duplicates
- correct errors from manual inspection (hard-coded)
- prepare images with bounding boxes for manual inspection
(- analyse pre-defined dataset)
- define new class (vehicle-type)
- prepare test / train datasets (20/80)
- prepare format for training YOLOv4
    (folder structure / txt files / yaml file / transformed bb coordinates)

### 'veh_type_yolo_training.ipynb ...' (in progress...)
    - train YOLOv4
    - evaluate results

### 'vehicl_type_predictor.ipynb  ...' (in progress...)
    - accepts images, detects vehicles and idetifies vehicle type
































# Car type identifier

## Project goal:





## Outline:

### Part A: Data exploration

- 1) Manual inspection - Dataset distribution.
- 2) Analysis of datasets.
- 3) Bounding box inspection.
- 4) Data modifications: Correcting identified errors

### Part B: Preparing the new dataset

- 1) Define new classes
- 2) Re-separate examples and prepare the dataset for Yolo v5.

### Part C: Train a neural network

- 1) Train Yolov5s (using pretrained weights)
</br>
</br>

!!! data.prep.ipynb downloads ~4gb of data


