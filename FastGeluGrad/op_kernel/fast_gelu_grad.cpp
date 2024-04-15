#include "kernel_operator.h"

extern "C" __global__ __aicore__ void fast_gelu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}