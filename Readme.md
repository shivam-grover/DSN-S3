# Dynamic Sparse Network + S3


This repository presents an example usage of S3 with a time series classification model DSN ([link](https://github.com/QiaoXiao7282/DSN) to the baseline's repository).

## Dataset
Download the EEG2 Dataset from UCI [here](https://github.com/titu1994/MLSTM-FCN/releases/download/v1.0/eeg2-20180328T234701Z-001.zip) and extract it inside the data folder. 

## Training the baseline alone

Set the --enable_S3 flag to 0 to disable S3
```
python trainer_DSN.py --sparse True --density 0.2 --sparse_init remain_sort --fix False --growth random --depth 4 --ch_size 47 --c_size 3 --k_size 39 --enable_S3 0

```

## Training the baseline with S3

Set the --enable_S3 flag to 1, and other S3 hyperparameter values.
```
python trainer_DSN.py --sparse True --density 0.2 --sparse_init remain_sort --fix False --growth random --depth 4 --ch_size 47 --c_size 3 --k_size 39 --epochs 100 --datalist 'eeg2' --enable_S3 1 --initial_num_segments 16 --num_layers 1 --segment_multiplier 2 --shuffle_vector_dim 1

```

## Performing grid search

In the file trainer_DSN_grid_search.py, we use optuna for performing grid search.

Our search space is

```
"initial_num_segments": [2, 4, 8, 16, 24],
"num_layers": [1, 2, 3],
"segment_multiplier": [0.5, 1, 2],
"shuffle_vector_dim": [1, 2, 3]
```

To run the grid search code, use the following command:

```
python trainer_DSN_grid_search.py --sparse True --density 0.2 --sparse_init remain_sort --fix False --growth random --depth 4 --ch_size 47 --c_size 3 --k_size 39 --epochs 100 --datalist 'eeg2' --enable_S3 1
```

**Note**: The original DSN code only used a train and test set. However, it is recommended to use a separate validation set for performing grid search. In our code, we have modified the data loading step to create three separate sets, and we optimise our hyperparameters on the best validation accuracy.

### Viewing Results with Optuna Dashboard
Optuna provides a useful dashboard to monitor the optimization process and visualize the trials. In the image below, you can see the values (validation accuracies) for different sets of hyperparameters. It offers a few other tools that you can explore when you open the dashboard locally.

![image](https://github.com/user-attachments/assets/a3d85ea5-7187-4bbd-a77c-c5e326eb5367)

To launch the dashboard, run:

```bash
optuna-dashboard sqlite:///optuna.db
```

If you are running your experiments on a remote server and would like to visualise the optuna study on your local machine then run the command above on your server, and then on your local machine run the following command

```bash
ssh -L 8080:localhost:8080 username@remote_server_ip
```

This will forward the remote server's port 8080 to your local machine. Now, you can open a browser on your local machine and navigate to http://localhost:8080 to access the Optuna dashboard and visualize the study's progress, best trials, and hyperparameter tuning details.

### Viewing the Results in the saved CSV files

Apart from optuna, we also save the results from each model in the folder "grid-search-results" in the form of csv files (one file per dataset).

![image](https://github.com/user-attachments/assets/8a1ce229-5eed-405b-9548-088b203600d1)

I have added the simple code for this in the trainer_DSN_grid_search.py file.

## Citation

```
@inproceedings{
grover2024segment,
title={Segment, Shuffle, and Stitch: A Simple Layer for Improving Time-Series Representations},
author={Shivam Grover and Amin Jalali and Ali Etemad},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=zm1LcgRpHm}
}
```
