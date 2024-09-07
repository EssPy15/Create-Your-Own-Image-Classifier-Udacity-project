### Overview

Code for project in Udacity's "AI Programming with Python" Nanodegree program. In this project, we first develop an image classifier using PyTorch pre-trained model from trochvision, then convert it into a command line application. By default pretrained 'Resnet50' model is used but it can changed to other models like vgg16, densenet121. vgg16 has generally more accuracy but is comparitively slower due to more hidden layers and so resnet50 is used as default as it has satisfactory accuracy as well runs comparitively faster.

This project is made in Python 3.12 and the following libraries.
- PyTorch
- ArgParse
- PIL
- NumPy
- Pandas
- matplotlib
- scikit-learn
All the above libraries were installed using Anaconda.

### Run 

In a terminal or command window and run the following commands:

- Train a network on a data set with train.py
  - Basic usage: python train.py data_directory
  - prints out training loss, validation loss, and validation accuracy as the network trains
  - Hyperparameters:
    - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    - Choose architecture: python train.py data_dir --arch "vgg13"
    - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    - Use GPU for training: python train.py data_dir --gpu

- Predict the breed of the flower in image with predict.py
  - Basic usage: python predict.py /path/to/image checkpoint
  - Hyperparameters:
    - Use GPU for inference: python predict.py input checkpoint --gpu
    - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    - Return top K(=n) most likely classes: python predict.py input checkpoint --top_k 3
