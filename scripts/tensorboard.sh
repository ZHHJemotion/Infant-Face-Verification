#!/usr/bin/env bash

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

log_dir=/home/pingguo/PycharmProject/Weights/logs/DS-Net-2018-06-09-14-47-59/
tb_dir=/usr/local/lib/python2.7/dist-packages/tensorboard/
python ${tb_dir}main.py --logdir=${log_dir} --port=9009