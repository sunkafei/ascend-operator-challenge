#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os

def gen_golden_data_simple():
    input_x = np.random.uniform(-100, 100, [4095, 4095])
    input_y = np.random.uniform(-100, 100, [4095, 4095])
    #input_y = input_x * np.random.uniform(1-1e-8, 1+1e-8, [4096, 4096])
    
    input_x = input_x.astype(np.int8)
    input_y = input_y.astype(np.int8)
    print((input_x - input_y)[:10])
    
    golden = np.less_equal(input_x, input_y)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
