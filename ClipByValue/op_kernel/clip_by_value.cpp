#include "kernel_operator.h"

extern "C" __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}