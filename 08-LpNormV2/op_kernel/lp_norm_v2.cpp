#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

template<typename T> class KernelLpNormV2 {
public:
    __aicore__ inline KernelLpNormV2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, uint32_t ptype, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->p = p;
        this->invp = 1.0f / p;
        this->ptype = ptype;
        this->ALIGN_NUM = ALIGN_NUM;
        this->lastpadding = (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ T*)y + startPointer, ALIGN_NUM);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(SumQueue, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> sum = SumQueue.AllocTensor<T>();
        T zero = this->ptype == 3 ? 3.40282346638528859811704183484516925e+38F : 0;
        Duplicate(sum, zero, this->tileLength);
        SumQueue.EnQue<T>(sum);
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length - this->lastpadding);

        {
            LocalTensor<T> sum = SumQueue.DeQue<T>();
            LocalTensor<T> tmp = tmpBuffer.Get<T>();
            if(this->ptype == 0){
                ReduceSum(sum, sum, tmp, this->tileLength);
                T invp = this->invp;
                Ln(sum, sum, 1);
                Muls(sum, sum, invp, 1);
                Exp(sum, sum, 1);
            }else if(this->ptype == 2){
                ReduceMax(sum, sum, tmp, this->tileLength);
            }else if(this->ptype == 3){
                ReduceMin(sum, sum, tmp, this->tileLength);
            }else{
                ReduceSum(sum, sum, tmp, this->tileLength);
            }
            SumQueue.EnQue<T>(sum);
            sum = SumQueue.DeQue<T>();
            DataCopy(zGm, sum, this->ALIGN_NUM);
            SumQueue.FreeTensor(sum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> sum = SumQueue.DeQue<T>();
        
        Abs(xLocal, xLocal, length);
        if(this->ptype == 0){
            Ln(xLocal, xLocal, length);
            Muls(xLocal, xLocal, this->p, length);
            Exp(xLocal, xLocal, length);
            Add(sum, xLocal, sum, length);
        }else if(this->ptype == 2){
            Max(sum, xLocal, sum, length);
        }else if(this->ptype == 3){
            Min(sum, xLocal, sum, length);
        }else if(this->ptype == 1){
            LocalTensor<T> tmp = tmpBuffer.Get<T>();
            LocalTensor<uint8_t> tmp2 = tmpBuffer2.Get<uint8_t>();
            Duplicate(tmp, (T)0.0, length);
            Compare(tmp2, xLocal, tmp, CMPMODE::EQ, length);
            Select(xLocal, tmp2, tmp, (T)1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Add(sum, xLocal, sum, length);
        }
        
        SumQueue.EnQue<T>(sum);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer, tmpBuffer2;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, 1> SumQueue;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    T p;
    T invp;
    uint32_t ptype;
    uint32_t ALIGN_NUM;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint8_t lastpadding;
};


template<typename T> class KernelLpNormV2P2 {
public:
    __aicore__ inline KernelLpNormV2P2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, uint32_t ptype, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->p = p;
        this->invp = 1.0f / p;
        this->ptype = ptype;
        this->ALIGN_NUM = ALIGN_NUM;
        this->lastpadding = (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ T*)y + startPointer, ALIGN_NUM);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(SumQueue, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> sum = SumQueue.AllocTensor<T>();
        T zero = 0;
        Duplicate(sum, zero, this->tileLength);
        SumQueue.EnQue<T>(sum);
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length - this->lastpadding);

        {
            LocalTensor<T> sum = SumQueue.DeQue<T>();
            LocalTensor<T> tmp = tmpBuffer.Get<T>();
            ReduceSum(sum, sum, tmp, this->tileLength);
            Sqrt(sum, sum, 1);
            SumQueue.EnQue<T>(sum);
            sum = SumQueue.DeQue<T>();
            DataCopy(zGm, sum, this->ALIGN_NUM);
            SumQueue.FreeTensor(sum);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // alloc tensor from queue memory
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        // enque input tensors to VECIN queue
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> sum = SumQueue.DeQue<T>();

        Mul(xLocal, xLocal, xLocal, length);
        Add(sum, xLocal, sum, length);
        
        SumQueue.EnQue<T>(sum);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, 1> SumQueue;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    float p;
    float invp;
    uint32_t ptype;
    uint32_t ALIGN_NUM;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint8_t lastpadding;
};

extern "C" __global__ __aicore__ void lp_norm_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(tiling_data.p == 2){
        KernelLpNormV2P2<DTYPE_X> op;
        op.Init(x, y, tiling_data.p, tiling_data.ptype, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }else{
        KernelLpNormV2<DTYPE_X> op;
        op.Init(x, y, tiling_data.p, tiling_data.ptype, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
    
}