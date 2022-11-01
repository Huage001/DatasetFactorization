# Dataset Factorization

This is the pytorch implementation of the following NeurIPS 2022 paper:

**[Dataset Distillation via Factorization](https://arxiv.org/abs/2210.16774)**

*Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, and Xinchao Wang.*

<img src="https://github.com/Huage001/DatasetFactorization/blob/main/teaser.png" width="1024px"/>

## Installation

* Create a new environment if you want:

  ```bash
  conda create -n HaBa python=3.8
  conda activate HaBa
  ```

* Clone the repo and install the required packages:

    ```bash
    git clone https://github.com/Huage001/DatasetFactorization.git
    cd DatasetFactorization
    pip install -r requirements.txt
    ```

## Dataset Distillation

* Install required packages:

  ```bash
  pip install -r requirements.txt
  ```

* First, generate buffer of training trajectories using:

  ```bash
  python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca --buffer_path={path_to_buffer_storage} --data_path={path_to_dataset}
  ```

* Then, edit *run_cifar10_ipc[xx]_style5.sh*. Change *{path_to_buffer_storage}*  to your path of buffers and *{path_to_dataset}* to your path of datasets.

* Run:

  ```bash
  bash run_cifar10_ipc[xx]_style5.sh
  ```

  *[xx]* can be 1, 10, or 50.

* Most of hyper-parameters are following [the baseline repo](https://github.com/GeorgeCazenavette/mtt-distillation). You may also try other configurations of arguments in the .sh files freely.

* *distill.py* contains the original implementation of the baseline method [MTT](https://github.com/GeorgeCazenavette/mtt-distillation) for comparison.

## Acknowledgement

This code borrows heavily from [mtt-distillation](https://github.com/GeorgeCazenavette/mtt-distillation) and [DatasetCondensation](https://github.com/VICO-UoE/DatasetCondensation).

## Citation

If you find this project useful in your research, please consider cite our paper and [the default baseline method](https://arxiv.org/abs/2203.11932):

```latex
@article{liu2022dataset,
    author    = {Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, Xinchao Wang},
    title     = {Dataset Distillation via Factorization},
    journal   = {NeurIPS},
    year      = {2022},
}
```

```latex
@inproceedings{
cazenavette2022distillation,
title={Dataset Distillation by Matching Training Trajectories},
author={George Cazenavette and Tongzhou Wang and Antonio Torralba and Alexei A. Efros and Jun-Yan Zhu},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2022}
}
```

