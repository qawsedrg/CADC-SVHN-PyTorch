# CADC-SVHN-PyTorch

Model designed to recognize a number with two digits, require by the CADC competition, trained on SVHN dataset, implemented using PyTorch.

## Requirements

* torch 1.0.1

* torchvision 0.2.1

* Pillow
  
* protobuf
  
* lmdb

* h5py
    ```
    $ sudo apt-get install libhdf5-dev
    $ pip install h5py
    ```

*   [NNI](https://github.com/microsoft/nni) (for AutoML)

## Setup

2. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

3. Extract to data folder, now your folder structure should be like below:
    ```
    SVHNClassifier
        - data
            - extra
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
    ```

## Data pre-process

1. Convert to LMDB format

    ```
    $ python convert_to_lmdb.py --data_dir ./data
    ```

1. (Optional) Test for reading LMDBs

    ```
    Open `read_lmdb_sample.ipynb` in Jupyter
    ```

## Train, test and convert to .pt

    Open `mobilenet.ipynb` in Jupyter
## AutoML

1.  (Optional) modify parameters in `search_space.json` and in `config.yml`
2.  `$ nnictl create --config ./config.yml`

