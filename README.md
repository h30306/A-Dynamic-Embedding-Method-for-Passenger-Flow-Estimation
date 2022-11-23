# A Dynamic Embedding Method for Passenger Flow Estimation

## Introduction
This repository contains the code for replicating results from

* [A Dynamic Embedding Method for Passenger Flow Estimation](https://doi.org/10.1109/IIAI-AAI53430.2021.00070)
* Wei-Yi Chung; Yen-Nan Ho; Yu-Hsuan Wu; Jheng-Long Wu
* In 2021 10th International Congress on Advanced Applied Informatics (IIAI-AAI)
* [slide](https://github.com/h30306/A-Dynamic-Embedding-Method-for-Passenger-Flow-Estimation/blob/main/Conference_Howard_20210706.pdf) | [Competitive Paper Award](https://github.com/h30306/A-Dynamic-Embedding-Method-for-Passenger-Flow-Estimation/blob/main/Competitive%20Paper%20Award.pdf)
<br>

## Getting Started

* Build a new virtual environment 
* Install python3 requirements: `pip install -r requirements.txt`
* Run `cd ./Project/model` to the model folder
* Use your own dataset (optional)
  * Construct the longtitude and latitude information of station in to [mrt_vd.csv](./Project/data/mrt_vd.csv)
  * Adjustment the format of passanger flow data to the [demo input format](./Project/data/data_2019.csv)
  * Run `python distance_matrix` to create the distance matrix
* Build training data, run `python Data_preparing_threeloss` to generate the training data
* Train your own models of pretrained stage
* repace the station feature from Node2Vec to BERT output in [GMAN](https://github.com/zhengchuanpan/GMAN)

## Training Insturctions

* Experiment configurations are found in `experiments.conf`
* Choose an experiment that you would like to run, e.g. `best`
* Training: `python train.py <experiment>`
* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* Evaluation: `python evaluate.py <experiment>`

