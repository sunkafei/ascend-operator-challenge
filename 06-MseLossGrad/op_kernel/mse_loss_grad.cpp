#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelMseLossGrad {
public:
    __aicore__ inline KernelMseLossGrad() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, float divnum, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->divnum = divnum;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)predict + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)label + startPointer, bufferlength);
        dGm.SetGlobalBuffer((__gm__ DTYPE_Y*)dout + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);
        this->d = dGm.GetValue(0);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        // pipe.InitBuffer(inQueueD, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
        LocalTensor<DTYPE_Y> xLocal = inQueueX.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        // LocalTensor<DTYPE_Y> dLocal = inQueueD.AllocTensor<DTYPE_Y>();
        
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        DataCopy(yLocal, yGm[progress * this->tileLength], length);
        // DataCopy(dLocal, dGm[progress * this->tileLength], length);
        
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        // inQueueD.EnQue(dLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<DTYPE_Y> xLocal = inQueueX.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        // LocalTensor<DTYPE_Y> dLocal = inQueueD.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.AllocTensor<DTYPE_Y>();

        Sub(yLocal, xLocal, yLocal, length);
        Muls(zLocal, yLocal, this->divnum, length);
        // Mul(zLocal, zLocal, dLocal, length);
        Muls(zLocal, zLocal, this->d, length);

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<DTYPE_Y>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        // inQueueD.FreeTensor(dLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.DeQue<DTYPE_Y>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;//, inQueueD;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_Y> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Y> dGm;
    GlobalTensor<DTYPE_Y> zGm;
    DTYPE_Y divnum;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    DTYPE_Y d;
};

class KernelMseLossGrad2 {
public:
    __aicore__ inline KernelMseLossGrad2() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, float divnum, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->divnum = divnum;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)predict + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)label + startPointer, bufferlength);
        dGm.SetGlobalBuffer((__gm__ DTYPE_Y*)dout + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueD, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
        LocalTensor<DTYPE_Y> xLocal = inQueueX.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> dLocal = inQueueD.AllocTensor<DTYPE_Y>();
        
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        DataCopy(yLocal, yGm[progress * this->tileLength], length);
        DataCopy(dLocal, dGm[progress * this->tileLength], length);
        
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        inQueueD.EnQue(dLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<DTYPE_Y> xLocal = inQueueX.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> dLocal = inQueueD.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.AllocTensor<DTYPE_Y>();

        Sub(yLocal, xLocal, yLocal, length);
        Muls(zLocal, yLocal, this->divnum, length);
        Mul(zLocal, zLocal, dLocal, length);

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<DTYPE_Y>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueD.FreeTensor(dLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.DeQue<DTYPE_Y>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueD;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_Y> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Y> dGm;
    GlobalTensor<DTYPE_Y> zGm;
    DTYPE_Y divnum;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};



extern "C" __global__ __aicore__ void mse_loss_grad(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(TILING_KEY_IS(1)){
        KernelMseLossGrad op;
        op.Init(predict, label, dout, y, tiling_data.divnum, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }else if(TILING_KEY_IS(2)){
        KernelMseLossGrad2 op;
        op.Init(predict, label, dout, y, tiling_data.divnum, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
}