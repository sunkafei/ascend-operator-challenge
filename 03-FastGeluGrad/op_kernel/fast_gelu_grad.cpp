#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelFastGeluGrad {
public:
    __aicore__ inline KernelFastGeluGrad() {}
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        dyGm.SetGlobalBuffer((__gm__ DTYPE_DY*)dy + startPointer, bufferlength);
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_DY));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(signbitBuffer, this->tileLength * sizeof(DTYPE_X));
        this->signbit = signbitBuffer.Get<DTYPE_X>();
        if constexpr (std::is_same_v<DTYPE_X, float>) {
            Duplicate(signbit.ReinterpretCast<uint32_t>(), uint32_t(2147483648u), this->tileLength);
        }
        else {
            Duplicate(signbit.ReinterpretCast<uint16_t>(), uint16_t(32768u), this->tileLength);
        }
    }
    __aicore__ inline void Process()
    {
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<DTYPE_DY> dyLocal = inQueueDY.AllocTensor<DTYPE_DY>();
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(dyLocal, dyGm[progress * this->tileLength], length);
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        // enque input tensors to VECIN queue
        inQueueDY.EnQue(dyLocal);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        LocalTensor<DTYPE_X> tmp = tmpBuffer.Get<DTYPE_X>();

        // z=dy*res
        // res = div_up/div_down
        // div_up = e^(-1.702|x|) + 1.702xe^(-1.702|x|) + e^(1.702(x-|x|))
        // div_down = (e^(-1.702|x|)+1)^2

        DTYPE_X c1 = -1, c2 = 1.702, c3 = 1.0;

        Muls(xLocal, xLocal, c2, length);     // xLocal = 1.702x
        if constexpr (std::is_same_v<DTYPE_X, float>) { // tmp = 1.702|x|
            Or(tmp.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), signbit.ReinterpretCast<uint16_t>(), length * 2);
        }
        else {
            Or(tmp.ReinterpretCast<uint16_t>(), xLocal.ReinterpretCast<uint16_t>(), signbit.ReinterpretCast<uint16_t>(), length);  
        }
        Add(zLocal, xLocal, tmp, length);     // 1.702(x-|x|)
        Exp(zLocal, zLocal, length);          // e^(1.702(x-|x|))
        Exp(tmp, tmp, length);                // e^(-1.702|x|)
        Mul(xLocal, xLocal, tmp, length);     // 1.702xe^(-1.702|x|)
        Add(xLocal, xLocal, tmp, length);     // e^(-1.702|x|) + 1.702xe^(-1.702|x|)
        Add(zLocal, xLocal, zLocal, length);  // e^(-1.702|x|) + 1.702xe^(-1.702|x|) + e^(1.702(x-|x|))
        Adds(tmp, tmp, c3, length);           // e^(-1.702|x|) + 1
        Mul(tmp, tmp, tmp, length);           // (e^(-1.702|x|) + 1)^2
        Div(zLocal, zLocal, tmp, length);
        Mul(zLocal, zLocal, dyLocal, length);

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        // free input tensors for reuse
        inQueueDY.FreeTensor(dyLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer, signbitBuffer;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDY, inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_DY> dyGm;
    GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    LocalTensor<DTYPE_X> signbit;
};

extern "C" __global__ __aicore__ void fast_gelu_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelFastGeluGrad op;
    op.Init(dy, x, z, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}