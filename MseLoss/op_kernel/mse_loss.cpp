#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelMseLoss {
public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR y, float divnum, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockIdx() < core_remain);
        this->tileLength = block_size;
        this->lastpadding = block_size;
        this->divnum = divnum;
        this->ALIGN_NUM = ALIGN_NUM;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->lastpadding = this->blockLength - this->lastpadding;

        auto startPointer = this->blockLength * GetBlockIdx() + (GetBlockIdx() < core_remain ? GetBlockIdx() : core_remain);
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)predict + startPointer, bufferlength);
        xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)label + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, ALIGN_NUM);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, ALIGN_NUM * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuffer2, ALIGN_NUM * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<DTYPE_Y> tmp2 = tmpBuffer2.Get<DTYPE_Y>();
        DTYPE_Y zero = 0;
        Duplicate(tmp2, zero, this->ALIGN_NUM);
        DataCopy(zGm, tmp2, this->ALIGN_NUM);
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1);

        // DataCopy(tmp2, zGm, this->ALIGN_NUM);
        // Muls(tmp2, tmp2, this->divnum, this->ALIGN_NUM);
        // DataCopy(zGm, tmp2, this->ALIGN_NUM);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<DTYPE_Y> xLocal = inQueueX.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        DataCopy(yLocal, yGm[progress * this->tileLength], length);
        
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<DTYPE_Y> xLocal = inQueueX.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_Y> tmp = tmpBuffer.Get<DTYPE_Y>();

        Sub(yLocal, xLocal, yLocal, length);
        Mul(yLocal, yLocal, yLocal, length);
        ReduceSum(xLocal, yLocal, tmp, length);
        Muls(xLocal, xLocal, this->divnum, this->ALIGN_NUM);
        DataCopy(zLocal, xLocal, this->ALIGN_NUM);


        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<DTYPE_Y>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<DTYPE_Y> zLocal = outQueueZ.DeQue<DTYPE_Y>();
        // copy progress_th tile from local tensor to global tensor
        SetAtomicAdd<float>();
        DataCopy(zGm, zLocal, this->ALIGN_NUM);
        SetAtomicNone();
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer, tmpBuffer2;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_Y> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Y> zGm;
    DTYPE_Y divnum;
    uint32_t ALIGN_NUM;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint8_t lastpadding;
};

extern "C" __global__ __aicore__ void mse_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelMseLoss op;
    op.Init(predict, label, y, tiling_data.divnum, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}