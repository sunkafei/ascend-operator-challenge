/**
* @file main.cpp
*
* Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "acl/acl.h"
#include "op_runner.h"
#include "common.h"
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>

bool g_isDevice = false;
int deviceId = 0;
std::string dtype;
std::vector<int64_t> read_shape(std::fstream &meta) {
    std::string line;
    getline(meta, line);
    std::istringstream stream(line);
    std::vector<int64_t> shape;
    int64_t dim;
    while (stream >> dim) {
        shape.push_back(dim);
    }
    return shape;
}
OperatorDesc CreateOpDesc() {
    std::fstream meta("../output/meta");
    aclDataType dataType;
    getline(meta, dtype);
    while (std::isspace(dtype.back())) {
        dtype.pop_back();
    }
    if (dtype == "numpy.int8") {
        dataType = ACL_INT8;
    }
    else if (dtype == "numpy.int32") {
        dataType = ACL_INT32;
    }
    else if (dtype == "numpy.float32") {
        dataType = ACL_FLOAT;
    }
    else if (dtype == "numpy.float16") {
        dataType = ACL_FLOAT16;
    }
    std::vector<int64_t> shape_x = read_shape(meta);
    std::vector<int64_t> shape_y = read_shape(meta);
    std::vector<int64_t> shape_z = read_shape(meta);
    std::vector<int64_t> shape_value = read_shape(meta);
    std::vector<int64_t> shape_output = read_shape(meta);
    
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataType, shape_x.size(), shape_x.data(), format);
    opDesc.AddInputTensorDesc(dataType, shape_y.size(), shape_y.data(), format);
    opDesc.AddInputTensorDesc(dataType, shape_z.size(), shape_z.data(), format);
    opDesc.AddInputTensorDesc(dataType, shape_value.size(), shape_value.data(), format);
    opDesc.AddOutputTensorDesc(dataType, shape_output.size(), shape_output.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_x.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_y.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("../input/input_z.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2));
    ReadFile("../input/input_value.bin", fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    WriteFile("../output/output.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

void DestoryResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destory resource failed");
    } else {
        INFO_LOG("Destory resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        }
        else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit("../scripts/acl.json") != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestoryResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }
    
    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestoryResource();
        return FAILED;
    }

    DestoryResource();

    return SUCCESS;
}
