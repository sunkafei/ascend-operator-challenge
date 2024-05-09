#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T> class KernelCross {
public:
    __aicore__ inline KernelCross() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t totalSize, uint64_t batchSize, uint64_t stepSize) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->totalSize = totalSize;
        this->batchSize = batchSize;
        this->stepSize = stepSize;
        Gm_x1.SetGlobalBuffer((__gm__ T*)x1, totalSize);
        Gm_x2.SetGlobalBuffer((__gm__ T*)x2, totalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, totalSize);
    }
    __aicore__ inline void Process() {
        for (uint64_t i = 0; i < batchSize; ++i) {
            for (uint64_t j = 0; j < stepSize; ++j) {
                auto index1 = i * 3 * stepSize + 0 * stepSize + j;
                auto index2 = i * 3 * stepSize + 1 * stepSize + j;
                auto index3 = i * 3 * stepSize + 2 * stepSize + j;
                float a1 = Gm_x1.GetValue(index1);
                float a2 = Gm_x1.GetValue(index2);
                float a3 = Gm_x1.GetValue(index3);
                float b1 = Gm_x2.GetValue(index1);
                float b2 = Gm_x2.GetValue(index2);
                float b3 = Gm_x2.GetValue(index3);
                auto result1 = a2 * b3 - a3 * b2;
                auto result2 = a3 * b1 - a1 * b3;
                auto result3 = a1 * b2 - a2 * b1;
                Gm_y.SetValue(index1, (T)result1);
                Gm_y.SetValue(index2, (T)result2);
                Gm_y.SetValue(index3, (T)result3);
            }
        }
    }
private:
    GlobalTensor<T> Gm_x1, Gm_x2, Gm_y;
    uint64_t totalSize, batchSize, stepSize;
};
extern "C" __global__ __aicore__ void cross(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCross<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize);
    op.Process();
}