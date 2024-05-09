#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T> class KernelCross {
public:
    __aicore__ inline KernelCross() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t totalSize[], uint64_t batchSize[], uint64_t stepSize[]) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->maxbatchSize = 0;
        this->maxstepSize = 0;
        this->maxtotalSize = 0;
        for (int i = 0; i < 2; ++i) {
            if (batchSize[i] > this->maxbatchSize)
                this->maxbatchSize = batchSize[i];
            if (stepSize[i] > this->maxstepSize)
                this->maxstepSize = stepSize[i]; 
            if (totalSize[i] > this->maxtotalSize)
                this->maxtotalSize = totalSize[i];  
            this->batchSize[i] = batchSize[i];
            this->stepSize[i] = stepSize[i];
            //this->batchOffset[i] = totalSize[i] / batchSize[i];
        }

        Gm_x1.SetGlobalBuffer((__gm__ T*)x1, totalSize[0]);
        Gm_x2.SetGlobalBuffer((__gm__ T*)x2, totalSize[1]);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, maxtotalSize);
    }
    __aicore__ inline void Process() {
        for (uint64_t i = 0; i < maxbatchSize; ++i) {
            for (uint64_t j = 0; j < maxstepSize; ++j) {
                float a1 = Gm_x1.GetValue(i % batchSize[0] * 3 * stepSize[0] + 0 * stepSize[0] + j % stepSize[0]);
                float a2 = Gm_x1.GetValue(i % batchSize[0] * 3 * stepSize[0] + 1 * stepSize[0] + j % stepSize[0]);
                float a3 = Gm_x1.GetValue(i % batchSize[0] * 3 * stepSize[0] + 2 * stepSize[0] + j % stepSize[0]);
                float b1 = Gm_x2.GetValue(i % batchSize[1] * 3 * stepSize[1] + 0 * stepSize[1] + j % stepSize[1]);
                float b2 = Gm_x2.GetValue(i % batchSize[1] * 3 * stepSize[1] + 1 * stepSize[1] + j % stepSize[1]);
                float b3 = Gm_x2.GetValue(i % batchSize[1] * 3 * stepSize[1] + 2 * stepSize[1] + j % stepSize[1]);
                auto result1 = a2 * b3 - a3 * b2;
                auto result2 = a3 * b1 - a1 * b3;
                auto result3 = a1 * b2 - a2 * b1;
                Gm_y.SetValue(i * 3 * maxstepSize + 0 * maxstepSize + j, (T)result1);
                Gm_y.SetValue(i * 3 * maxstepSize + 1 * maxstepSize + j, (T)result2);
                Gm_y.SetValue(i * 3 * maxstepSize + 2 * maxstepSize + j, (T)result3);
            }
        }
    }
private:
    GlobalTensor<T> Gm_x1, Gm_x2, Gm_y;
    uint64_t maxtotalSize, maxbatchSize, maxstepSize;
    uint64_t batchSize[3], squareSize[3], stepSize[3];
};
extern "C" __global__ __aicore__ void cross(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCross<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize);
    op.Process();
}