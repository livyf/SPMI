# SPMI

This is a PyTorch implementation of **ICML 2024** paper [Learning with Partial-Label and Unlabeled Data: A Uniform Treatment for Supervision Redundancy and Insufficiency](https://proceedings.mlr.press/v235/liu24ar.html)

## Quick Start

We provide sample shell scripts for model training.

```shell
python -u main.py --exp_type rand --exp_dir ./experiment/fmnist_rand0.3_4000 --dataset fmnist --data_dir ./data --num_class 10 --cuda_VISIBLE_DEVICES 0 --seed 123 --arch lenet --workers 0 --lr 0.05 --wd 1e-3 --epochs 500 --batch_size 256 --optimizer sgd --momentum 0.9 --cosine --partial_rate 0.3 --labeled_num 4000 --warm_up 10 --kl_theta_labeled 3 --kl_theta_unlabeled 2 --ema_theta 0.999

python -u main.py --exp_type rand --exp_dir ./experiment/cifar10_rand0.3_4000 --dataset cifar10 --data_dir ./data --num_class 10 --cuda_VISIBLE_DEVICES 0 --seed 123 --arch WRN_28_2 --workers 0 --lr 0.05 --wd 1e-3 --epochs 500 --batch_size 256 --optimizer sgd --momentum 0.9 --cosine --partial_rate 0.3 --labeled_num 4000 --warm_up 10 --kl_theta_labeled 3 --kl_theta_unlabeled 2 --ema_theta 0.999

python -u main.py --exp_type rand --exp_dir ./experiment/cifar100_rand0.05_10000 --dataset cifar100 --data_dir ./data --num_class 100 --cuda_VISIBLE_DEVICES 0 --seed 123 --arch WRN_28_8 --workers 0 --lr 0.05 --wd 1e-3 --epochs 500 --batch_size 256 --optimizer sgd --momentum 0.9 --cosine --partial_rate 0.05 --labeled_num 10000 --warm_up 20 --kl_theta_labeled 3 --kl_theta_unlabeled 2  --ema_theta 0.999

python -u main.py --exp_type rand --exp_dir ./experiment/svhn_rand0.3_1000 --dataset svhn --data_dir ./data --num_class 10 --cuda_VISIBLE_DEVICES 0 --seed 123 --arch WRN_28_2 --workers 0 --lr 0.05 --wd 1e-3 --epochs 500 --batch_size 256 --optimizer sgd --momentum 0.9 --cosine --partial_rate 0.3 --labeled_num 1000 --warm_up 10 --kl_theta_labeled 3 --kl_theta_unlabeled 2 --ema_theta 0.999
```


## Citation

If this work is helpful to you, please cite:

```
@inproceedings{
  liu2024spmi,
  title={Learning with Partial-Label and Unlabeled Data: A Uniform Treatment for Supervision Redundancy and Insufficiency},
  author={Liu, Yangfan and Lv, Jiaqi and Geng, Xin and Xu, Ning},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}

```

If you have any questions, please contact: Yangfan Liu (liuyangfan@seu.edu.cn).
