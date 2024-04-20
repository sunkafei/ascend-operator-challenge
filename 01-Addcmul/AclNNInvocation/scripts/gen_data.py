#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy
import os
def gen_golden_data_simple():
    dtype = numpy.float16
    output_shape = [133, 4095]
    input_x = numpy.random.uniform(1, 4, [133, 4095]).astype(dtype)
    input_y = numpy.random.uniform(1, 4, [133, 4095]).astype(dtype)
    input_z = numpy.random.uniform(1, 4, [133, 4095]).astype(dtype)
    input_value = numpy.random.uniform(1, 4, [1]).astype(dtype)

    golden = (input_x + input_y*input_z*input_value).astype(dtype)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    input_z.tofile("./input/input_z.bin")
    input_value.tofile("./input/input_value.bin")
    golden.tofile("./output/golden.bin")
    print(input_x.shape)
    with open("./output/meta", "w") as fp:
        print(str(dtype).split("'")[1], file=fp)
        print(*input_x.shape, file=fp)
        print(*input_y.shape, file=fp)
        print(*input_z.shape, file=fp)
        print(*input_value.shape, file=fp)
        print(*output_shape, file=fp)

if __name__ == "__main__":
    gen_golden_data_simple()
