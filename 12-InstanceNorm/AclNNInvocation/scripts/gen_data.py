#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data_simple():
    x = np.random.uniform(1, 10, [3,1024, 1024,3]).astype(np.float16)
    gamma = np.random.uniform(1, 10, [3,1024, 1024,3]).astype(np.float16)
    beta = np.random.uniform(1, 10, [3,1024, 1024,3]).astype(np.float16)
    data_format = "ND"
    epsilon = 0
    shape_x = x.shape
    axis = []
    if data_format in ("NDHWC",):
        axis = [1, 2, 3]
    elif data_format in ("NCDHW",):
        axis = [2, 3, 4]
    elif data_format in ("NHWC",):
        axis = [1, 2]
    elif data_format in ("NCHW",):
        axis = [2, 3]
    elif data_format in ("ND",):
        axis = list(range(2, len(shape_x)))


    mean = np.mean(x, tuple(axis), keepdims=True)
    variance = np.mean(np.power((x - mean),2), tuple(axis), keepdims=True)
    result = gamma*((x - mean) / np.sqrt(variance + epsilon)) + beta
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x.tofile("./input/input_x.bin")
    gamma.tofile("./input/input_gamma.bin")
    beta.tofile("./input/input_beta.bin")
    result.tofile("./output/golden_y.bin")
    mean.tofile("./output/golden_mean.bin")
    variance.tofile("./output/golden_variance.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
