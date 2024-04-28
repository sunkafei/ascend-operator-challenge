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

/*template<typename T, typename A, typename C> __aicore__ inline void Wait(A&& arg, C&& callable) {
    LocalTensor<T> x1 = arg.template AllocTensor<T>();
    LocalTensor<T> x2 = arg.template AllocTensor<T>();
    arg.EnQue(x1);
    arg.EnQue(x2);
    LocalTensor<T> y1 = arg.template DeQue<T>();
    LocalTensor<T> y2 = arg.template DeQue<T>();
    callable();
    arg.FreeTensor(y1);
    arg.FreeTensor(y2);
}*/
template<typename T> __aicore__ inline void GroupReduce(const LocalTensor<T> &y, const LocalTensor<T> &x, uint32_t group_size, uint32_t group_count) {
    static constexpr int32_t SIZE = sizeof(T);
    static constexpr int32_t ALIGN = 32 / sizeof(T);
    int32_t reduceCount = 256 / sizeof(T);
    int32_t new_size = group_size / reduceCount;
    if (new_size % ALIGN) {
        reduceCount /= ALIGN / new_size;
    }
    int32_t repeatTimes = group_count * group_size / reduceCount;
    int32_t repStride = (reduceCount * SIZE - 1) / 32 + 1;
    WholeReduceSum(x, x, reduceCount, repeatTimes, 1, 1, repStride);
    group_size /= reduceCount;
    WholeReduceSum(y, x, group_size, group_count, 1, 1, group_size * SIZE / 32);
}
template<typename T> class KernelInstanceNorm_Fast {
public:
    __aicore__ inline KernelInstanceNorm_Fast() {}
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
        this->maxsquareSize = maxtotalSize / maxbatchSize;
        this->packNumber = 4;
        this->tileLength = this->packNumber * this->maxsquareSize;
        Gm_x.SetGlobalBuffer((__gm__ T*)x, maxtotalSize);
        Gm_gamma.SetGlobalBuffer((__gm__ T*)gamma, maxtotalSize);
        Gm_beta.SetGlobalBuffer((__gm__ T*)beta, maxtotalSize);
        Gm_y.SetGlobalBuffer((__gm__ T*)y, maxtotalSize);
        Gm_mean.SetGlobalBuffer((__gm__ T*)mean, maxbatchSize);
        Gm_variance.SetGlobalBuffer((__gm__ T*)variance, maxbatchSize);
        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_gamma, BUFFER_NUM, this->tileLength * sizeof(T)); //todo
        pipe.InitBuffer(Q_beta, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(Q_mean, 1, this->maxbatchSize * sizeof(T));
        pipe.InitBuffer(Q_variance, 1, this->maxbatchSize * sizeof(T));
    }
    __aicore__ inline void Process() { // 2281
        auto cof = T(1.0f / maxsquareSize);
        LocalTensor<T> mean = Q_mean.AllocTensor<T>();
        for (uint64_t i = 0; i < maxbatchSize; i += packNumber) {
            {
                LocalTensor<T> x = Q_x.AllocTensor<T>();
                DataCopy(x, Gm_x[i * maxsquareSize], tileLength);
                Q_x.EnQue(x);
            }
            {
                LocalTensor<T> x = Q_x.DeQue<T>();
                Muls(x, x, cof, tileLength);
                GroupReduce(mean[i], x, maxsquareSize, packNumber);
                Q_x.FreeTensor(x);
            }
        }
        Q_mean.EnQue(mean);
        LocalTensor<T> mean_out = Q_mean.DeQue<T>();
        DataCopy(Gm_mean, mean_out, maxbatchSize);
        Q_mean.FreeTensor(mean_out);

        LocalTensor<T> variance = Q_variance.AllocTensor<T>();
        for (uint64_t i = 0; i < maxbatchSize; i += packNumber) {
            {
                LocalTensor<T> x = Q_x.AllocTensor<T>();
                DataCopy(x, Gm_x[i * maxsquareSize], tileLength);
                Q_x.EnQue(x);
            }
            {
                LocalTensor<T> x = Q_x.DeQue<T>();
                for (int j = 0; j < packNumber; ++j) {
                    float avg = mean.GetValue(i + j);
                    Adds(x[j * maxsquareSize], x[j * maxsquareSize], T(-avg), maxsquareSize);
                }
                Mul(x, x, x, tileLength);
                Muls(x, x, cof, tileLength);
                GroupReduce(variance[i], x, maxsquareSize, packNumber);
                Q_x.FreeTensor(x);
            }
        }
        Q_variance.EnQue(variance);
        LocalTensor<T> variance_out = Q_variance.DeQue<T>();
        DataCopy(Gm_variance, variance_out, maxbatchSize);
        Q_variance.FreeTensor(variance_out);

        for (uint64_t i = 0; i < maxbatchSize; i += packNumber) {
            {
                LocalTensor<T> x = Q_x.AllocTensor<T>();
                LocalTensor<T> gamma = Q_gamma.AllocTensor<T>();
                LocalTensor<T> beta = Q_beta.AllocTensor<T>();
                DataCopy(x, Gm_x[i * maxsquareSize], tileLength);
                for (int j = 0; j < packNumber; ++j) {
                    int k = i + j;
                    DataCopy(gamma[j * maxsquareSize], Gm_gamma[k % batchSize[1] * maxsquareSize], maxsquareSize);
                    DataCopy(beta[j * maxsquareSize], Gm_beta[k % batchSize[2] * maxsquareSize], maxsquareSize);
                }
                Q_x.EnQue(x);
                Q_gamma.EnQue(gamma);
                Q_beta.EnQue(beta);
            }
            {
                LocalTensor<T> y = Q_y.AllocTensor<T>();
                LocalTensor<T> x = Q_x.DeQue<T>();
                LocalTensor<T> gamma = Q_gamma.DeQue<T>();
                LocalTensor<T> beta = Q_beta.DeQue<T>();
                for (int j = 0; j < packNumber; ++j) {
                    float avg = mean.GetValue(i + j);
                    float var = variance.GetValue(i + j);
                    float deno = 1.0f / sqrt(var + epsilon);
                    Adds(x[j * maxsquareSize], x[j * maxsquareSize], T(-avg), maxsquareSize);
                    Muls(x[j * maxsquareSize], x[j * maxsquareSize], T(deno), maxsquareSize);
                }
                //Adds(x, x, T(-avg), tileLength); // (x - mean)
                //Muls(x, x, T(deno), tileLength); // (x - mean) / sqrt(variance + epsilon)
                Mul(x, x, gamma, tileLength); // gamma * (x - mean) / sqrt(variance + epsilon)
                Add(y, x, beta, tileLength); // gamma * (x - mean) / sqrt(variance + epsilon) + beta

                Q_y.EnQue<T>(y);
                Q_x.FreeTensor(x);
                Q_gamma.FreeTensor(gamma);
                Q_beta.FreeTensor(beta);
            }
            {
                LocalTensor<T> y = Q_y.DeQue<T>();
                DataCopy(Gm_y[i * maxsquareSize], y, tileLength);
                Q_y.FreeTensor(y);
            }
        }
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x, Q_gamma, Q_beta;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y, Q_mean, Q_variance;
    GlobalTensor<T> Gm_x, Gm_gamma, Gm_beta, Gm_y, Gm_mean, Gm_variance;
    uint64_t tileLength, packNumber;
    uint64_t maxtotalSize, maxbatchSize, maxstepSize, maxsquareSize;
    uint64_t batchSize[3], squareSize[3], stepSize[3], batchOffset[3];
    float epsilon;
};
extern "C" __global__ __aicore__ void instance_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.stepSize[0] > 1 || tiling_data.totalSize[0] / tiling_data.batchSize[0] * sizeof(DTYPE_X) % 256 != 0) { //todo?
        KernelInstanceNorm<DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize, tiling_data.epsilon);
        op.Process();
    }
    else {
        KernelInstanceNorm_Fast<DTYPE_X> op;
        op.Init(x, gamma, beta, y, mean, variance, tiling_data.totalSize, tiling_data.batchSize, tiling_data.stepSize, tiling_data.epsilon);
        op.Process();
    }
}