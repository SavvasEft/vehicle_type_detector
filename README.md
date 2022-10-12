# Car type identifier

## Project Description:

- This project aims in training a neural network to identify car types. To achieve this, we used the Cars dataset by Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei ([here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)). After analysis of the dataset, we used examples from it to create a new dataset with new classes. Then we trained Yolo on our custom data to identify car types.

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


Parts A and B: data_preparation.ipynb (Jupyter ntb run on laptop) </br>
</br>
</br>
</br>
In order to prepare the dataset (Parts A and B)follow these steps:

- 1) From the webside of the data [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), download these:</br>
 - [Train dataset](http://ai.stanford.edu/~jkrause/car196/cars_train.tgz)</br>
 - [Test dataset](http://ai.stanford.edu/~jkrause/car196/cars_test.tgz)</br>
 - [Test annotations](http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat) to be used</br>
 - All examples
   - [Images](http://ai.stanford.edu/~jkrause/car196/car_ims.tgz)</br>
   - [Bounding Boxes and Labels](http://ai.stanford.edu/~jkrause/car196/cars_annos.mat)</br>

        
- 2) Prepare the following folder structure:</br>

Data</br>
&emsp; |\-----test</br>
&emsp; |&emsp;&emsp;|</br>
&emsp; |&emsp;(test images) </br>
&emsp; |</br> 
&emsp; |\-----train</br>
&emsp; |&emsp;&emsp;|</br>
&emsp; |&emsp;(train images)</br> 
&emsp; |</br> 
&emsp; |\-----car_ims</br>
&emsp; |&emsp;&emsp;|</br>
&emsp; |&emsp;(all examples images) </br>
&emsp; |</br>
&emsp; |\-----cars_test_annos_withlabels.mat</br>
&emsp; |\-----cars_train_annos.mat</br>
&emsp; |\-----cars_meta.mat</br>
&emsp; |\-----cars_annos.mat</br>

 
 
- 3) Manualy do the changes described in Section "Part A: Data exploration" Subsection 4) Data modifications: Correcting identified errors


- 4) Run data_preparation.ipynb</br>
