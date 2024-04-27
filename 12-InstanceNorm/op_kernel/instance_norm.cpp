#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T> class KernelInstanceNorm {
public:
    __aicore__ inline KernelInstanceNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, uint64_t totalSize[], uint64_t batchSize[], uint64_t stepSize[], float epsilon) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->maxbatchSize = 0;
        this->maxstepSize = 0;
        this->maxtotalSize = 0;
        for (int i = 0; i < 3; ++i) {
            if (batchSize[i] > this->maxbatchSize)
                this->maxbatchSize = batchSize[i];
            if (stepSize[i] > this->maxstepSize)
                this->maxstepSize = stepSize[i]; 
            if (totalSize[i] > this->maxtotalSize)
                this->maxtotalSize = totalSize[i];  
            this->batchSize[i] = batchSize[i];
            this->stepSize[i] = stepSize[i];
            this->squareSize[i] = totalSize[i] / batchSize[i] / stepSize[i];
            this->batchOffset[i] = totalSize[i] / batchSize[i];
        }
        this->epsilon = epsilon;
        this->maxsquareSize = maxtotalSize / maxbatchSize / maxstepSize;
        Gm_x.SetGlobalBuffer((__gm__ T*)x, maxtotalSize);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, maxtotalSize);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, maxtotalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, maxtotalSize);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, maxbatchSize * maxstepSize);
        Gm_variance.SetGlobalBuffer((__gm__ T*)variance, maxbatchSize * maxstepSize);
    }
    __aicore__ inline void Process() {
        for (uint64_t i = 0; i < maxbatchSize; ++i) {
            for (uint64_t j = 0; j < maxstepSize; ++j) {
                float sum = 0.0;
                for (uint64_t k = 0; k < maxsquareSize; ++k) {
                    float val = Gm_x.GetValue(i * maxsquareSize * maxstepSize + k * maxstepSize + j);
                    sum += val;
                }
                float avg = sum / maxsquareSize;
                Gm_mean.SetValue(i * maxstepSize + j, (T)avg);
            }
        }
        for (uint64_t i = 0; i < maxbatchSize; ++i) {
            for (uint64_t j = 0; j < maxstepSize; ++j) {
                float avg = Gm_mean.GetValue(i * maxstepSize + j);
                float sum = 0.0;
                for (uint64_t k = 0; k < maxsquareSize; ++k) {
                    float val = Gm_x.GetValue(i * maxsquareSize * maxstepSize + k * maxstepSize + j);
                    sum += (val - avg) * (val - avg);
                }
                float var = sum / maxsquareSize;
                Gm_variance.SetValue(i * maxstepSize + j, (T)var);
            }
        }
        for (uint64_t i = 0; i < maxbatchSize; ++i) {
            for (uint64_t j = 0; j < maxstepSize; ++j) {
                float mean = Gm_mean.GetValue(i * maxstepSize + j);
                float variance = Gm_variance.GetValue(i * maxstepSize + j);
                float sum = 0.0;
                for (uint64_t k = 0; k < maxsquareSize; ++k) {
                    auto index = i * maxsquareSize * maxstepSize + k * maxstepSize + j;
                    float x = Gm_x.GetValue(index);
                    float gamma = Gm_gamma.GetValue(i % batchSize[1] * batchOffset[1] + k % squareSize[1] * stepSize[1] + j % stepSize[1]);
                    float beta = Gm_beta.GetValue(i % batchSize[2] * batchOffset[2] + k % squareSize[2] * stepSize[2] + j % stepSize[2]);
                    float result = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta;
                    Gm_y.SetValue(index, (T)result);
                }
            }
        }
    }
private:
    GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_variance;
    uint64_t maxtotalSize, maxbatchSize, maxstepSize, maxsquareSize;
    uint64_t batchSize[3], squareSize[3], stepSize[3], batchOffset[3];
    float epsilon;
};
extern "C" __global__ __aicore__ void instance_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelInstanceNorm<DTYPE_X> op;
    op.Init(x, gamma, beta, y, mean, variance, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize, tiling_data.epsilon);
    op.Process();
}