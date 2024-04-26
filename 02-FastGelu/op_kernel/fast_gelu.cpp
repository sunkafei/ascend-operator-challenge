#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + startPointer, bufferlength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(DTYPE_X));
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
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_X> tmp = tmpBuffer.Get<DTYPE_X>();

        DTYPE_X c1 = 0.851, c2 = -1.702, c3 = 1.0;
        DTYPE_X zero = 0.0;

        // x*e^(0.851)*(x-|x|)/(1+e^(-1.702|x|))

        Abs(tmp, xLocal, length);
        Muls(yLocal, tmp, c2, length);
        Exp(yLocal, yLocal, length);
        Adds(yLocal, yLocal, c3, length);

        Mins(xLocal, xLocal, zero, length);
        Div(yLocal, tmp, yLocal, length);
        Add(yLocal, yLocal, xLocal, length);

        /*Sub(tmp, xLocal, tmp, length);
        Muls(tmp, tmp, c1, length);
        Exp(tmp, tmp, length);

        Div(yLocal, xLocal, yLocal, length);
        Mul(yLocal, tmp, yLocal, length);*/

        // enque the output tensor to VECOUT queue
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        // free output tensor for reuse
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelFastGelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}
