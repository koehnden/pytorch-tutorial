# pytorch-tutorial
Code for a tutorial in pyTorch given at Berlin School of Economic and Law (HWR)

## Setup

The Project uses python 2.7. It is best practice to use a clean python version by using a virtual enviroment.
In order to install the needed libraries, I recommend also install pip
1. Installing pip:
``` $ sudo apt-get install python-pip ```
2. Install virtualenv: 
``` $ pip install virtualenv ```

#### 1. Clone the project (git needs to be installed):
1. install git if you have not done yet
``` $ apt-get install git-core ```
2. clone the repo
``` $ git clone https://github.com/koehnden/pytorch-tutorial.git ```

#### 2. Create and initialize virtualenv for the project:
1. cd into the project folder and create a virtualenviroment called venv:
``` $ virtualenv venv ```
2. acitivate virtualenv running:
``` $ source venv/bin/activate ``` 
    
##### 2.2 Install python packages
1. install [pytorch](https://pytorch.org/) by generating the suitable pip command, i.e. choose your OS and python version and the select CUDA None
2. to install the remaining libraries run:
``` $ pip install -r requirements.txt ```
