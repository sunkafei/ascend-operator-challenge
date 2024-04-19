#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T> class KernelInstanceNorm {
public:
    __aicore__ inline KernelInstanceNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, uint64_t totalSize, uint64_t batchSize, uint64_t stepSize) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->totalSize = totalSize;
        this->batchSize = batchSize;
        this->stepSize = stepSize;
        this->squareSize = totalSize / batchSize / stepSize;
        Gm_x.SetGlobalBuffer((__gm__ T*)x, totalSize);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, totalSize);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, totalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, totalSize);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, batchSize * stepSize);
        Gm_variance.SetGlobalBuffer((__gm__ T*)variance, batchSize * stepSize);

        //this->clip_value_max = Gm_clip_value_max.GetValue(0);
    }
    __aicore__ inline void Process() {
        for (uint64_t i = 0; i < batchSize; ++i) {
            for (uint64_t j = 0; j < stepSize; ++j) {
                float sum = 0.0;
                for (uint64_t k = 0; k < squareSize; ++k) {
                    float val = Gm_x.GetValue(i * squareSize * stepSize + k * stepSize + j);
                    sum += val;
                }
                float avg = sum / squareSize;
                Gm_mean.SetValue(i * stepSize + j, (T)avg);
            }
        }
        for (uint64_t i = 0; i < batchSize; ++i) {
            for (uint64_t j = 0; j < stepSize; ++j) {
                float avg = Gm_mean.GetValue(i * stepSize + j);
                float sum = 0.0;
                for (uint64_t k = 0; k < squareSize; ++k) {
                    float val = Gm_x.GetValue(i * squareSize * stepSize + k * stepSize + j);
                    sum += (val - avg) * (val - avg);
                }
                float var = sum / squareSize;
                Gm_variance.SetValue(i * stepSize + j, (T)var);
            }
        }
    }
private:
    GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_variance;
    uint64_t totalSize, batchSize, stepSize, squareSize;
};
extern "C" __global__ __aicore__ void instance_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelInstanceNorm<DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, variance, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize);
    op.Process();
}