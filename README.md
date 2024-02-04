# FaceNet
### Overview
Welcome to the FaceNet project! This project aims to generate embedding vectors for given faces as input using a custom PyTorch implementation. The repository is organized into three main folders: data, models, and tests. Please note that this project is a work in progress, and the code may be incomplete.

### FaceNet Algorithm
FaceNet is a face recognition system that maps facial features into a high-dimensional space where the distance between the points directly corresponds to the similarity between the faces. This is achieved by training a neural network to generate embeddings for facial images. These embeddings can then be compared to determine the similarity or dissimilarity between different faces.

### Project Structure
**data**:  
generate_dataset.py: This script contains code to generate the dataset dataframe.  
dataset.py: Implementation of a customized PyTorch dataset object for handling the face dataset.

**models**:  
This folder contains the PyTorch implementations of the FaceNet model architecture for face embedding.

**tests**:  
This folder contains code to test and evaluate the model's performance. It includes various scripts for assessing the accuracy and efficiency of the FaceNet model.
Setup


### To set up and run the FaceNet project, follow these steps:

Clone the repository to your local machine:  
`$ git clone https://github.com/ponakilan/FaceNet.git`  
`$ cd FaceNet`

Install the required dependencies:  
`$ pip install -r requirements.txt`

### Known Issues
The project is currently not completed, and some parts of the code may be incomplete or under development.
