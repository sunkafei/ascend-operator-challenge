#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import torch
import os

def gen_golden_data_simple():
    x1_tensor = np.random.uniform(-4, 4, [4, 1, 6, 3, 7, 8, 2]).astype(np.float16)
    x2_tensor = np.random.uniform(-4, 4, [1, 5, 6, 3, 7, 1, 2]).astype(np.float16)
    x1 = torch.tensor(x1_tensor.astype(np.float32))
    x2 = torch.tensor(x2_tensor.astype(np.float32))
    dim = 3
    res_tensor = torch.cross(x1,x2 , dim=dim)
    golden = res_tensor.numpy().astype(np.float16)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_tensor.tofile("./input/input_x1.bin")
    x2_tensor.tofile("./input/input_x2.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
