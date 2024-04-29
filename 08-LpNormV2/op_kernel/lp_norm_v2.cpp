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
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, ALIGN_NUM * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>(); LocalTensor<T> zLocal2 = outQueueZ.AllocTensor<T>();
        T zero = this->ptype == 3 ? 3.40282346638528859811704183484516925e+38F : 0;
        Duplicate(zLocal, zero, this->ALIGN_NUM);
        outQueueZ.EnQue<T>(zLocal); outQueueZ.EnQue<T>(zLocal2);
        zLocal = outQueueZ.DeQue<T>(); zLocal2 = outQueueZ.DeQue<T>();
        DataCopy(zGm, zLocal, this->ALIGN_NUM);
        outQueueZ.FreeTensor(zLocal); outQueueZ.FreeTensor(zLocal2);
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            ComputeFull(i, this->tileLength);
            CopyOut(i);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length - this->lastpadding);
        CopyOut(loopCount - 1);

        
        zLocal = outQueueZ.AllocTensor<T>(); zLocal2 = outQueueZ.AllocTensor<T>();
        DataCopy(zLocal, zGm, this->ALIGN_NUM);
        if(this->ptype == 0){
            T invp = this->invp;
            Ln(zLocal, zLocal, 1);
            Muls(zLocal, zLocal, invp, 1);
            Exp(zLocal, zLocal, 1);
        }
        outQueueZ.EnQue<T>(zLocal); outQueueZ.EnQue<T>(zLocal2);
        zLocal = outQueueZ.DeQue<T>(); zLocal2 = outQueueZ.DeQue<T>();
        DataCopy(zGm, zLocal, this->ALIGN_NUM);
        outQueueZ.FreeTensor(zLocal); outQueueZ.FreeTensor(zLocal2);
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
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        LocalTensor<float> tmp = tmpBuffer.Get<float>();

        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xf = tmpBuffer2.Get<float>();
            Cast(xf, xLocal, RoundMode::CAST_NONE, length);
            Abs(xf, xf, length);
            if(this->ptype == 0){
                Ln(xf, xf, length);
                Muls(xf, xf, this->p, length);
                Exp(xf, xf, length);
                ReduceSum(xf, xf, tmp, length);
            }else if(this->ptype == 2){
                ReduceMax(xf, xf, tmp, length);
            }else if(this->ptype == 3){
                ReduceMin(xf, xf, tmp, length);
            }else if(this->ptype == 1){
                LocalTensor<uint8_t> tmp2 = tmpBuffer3.Get<uint8_t>();
                Duplicate(tmp, 0.0f, length);
                Compare(tmp2, xf, tmp, CMPMODE::EQ, length);
                Select(xf, tmp2, tmp, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
                ReduceSum(xf, xf, tmp, length);
                // Muls(xf, xf, -1.0f, length);
                // LocalTensor<uint32_t> xuint = xf.template ReinterpretCast<uint32_t>();
                // ShiftRight(xuint, xuint, 31u, length);
                // LocalTensor<int32_t> xint = xuint.template ReinterpretCast<int32_t>();
                // Cast(xf, xint, RoundMode::CAST_NONE, length);
                // ReduceSum(xf, xf, tmp, length);
            }
            Cast(zLocal, xf, RoundMode::CAST_NONE, this->ALIGN_NUM);
        }else{
            Abs(xLocal, xLocal, length);
            if(this->ptype == 0){
                Ln(xLocal, xLocal, length);
                Muls(xLocal, xLocal, this->p, length);
                Exp(xLocal, xLocal, length);
                ReduceSum(xLocal, xLocal, tmp, length);
            }else if(this->ptype == 2){
                ReduceMax(xLocal, xLocal, tmp, length);
            }else if(this->ptype == 3){
                ReduceMin(xLocal, xLocal, tmp, length);
            }else if(this->ptype == 1){
                LocalTensor<uint8_t> tmp2 = tmpBuffer3.Get<uint8_t>();
                Duplicate(tmp, 0.0f, length);
                Compare(tmp2, xLocal, tmp, CMPMODE::EQ, length);
                Select(xLocal, tmp2, tmp, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
                ReduceSum(xLocal, xLocal, tmp, length);
                // Muls(xLocal, xLocal, -1.0f, length);
                // LocalTensor<uint32_t> xuint = xLocal.template ReinterpretCast<uint32_t>();
                // ShiftRight(xuint, xuint, 31u, length);
                // LocalTensor<int32_t> xint = xuint.template ReinterpretCast<int32_t>();
                // Cast(xLocal, xint, RoundMode::CAST_NONE, length);
                // ReduceSum(xLocal, xLocal, tmp, length);
            }
            DataCopy(zLocal, xLocal, this->ALIGN_NUM);
        }
        

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<T>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void ComputeFull(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        LocalTensor<float> tmp = tmpBuffer.Get<float>();

        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xf = tmpBuffer2.Get<float>();
            Cast(xf, xLocal, RoundMode::CAST_NONE, length);
            Abs(xf, xf, length);
            if(this->ptype == 0){
                Ln(xf, xf, length);
                Muls(xf, xf, this->p, length);
                Exp(xf, xf, length);
                uint64_t mask = 32 / 4 * 8;
                uint64_t repeat = (length + mask - 1) / mask;
                WholeReduceSum<float>(xf, xf, mask, repeat, 1, 1, 8);
                WholeReduceSum<float>(xf, xf, repeat, 1, 1, 1, 8);
            }else if(this->ptype == 2){
                ReduceMax(xf, xf, tmp, length);
            }else if(this->ptype == 3){
                ReduceMin(xf, xf, tmp, length);
            }else if(this->ptype == 1){
                LocalTensor<uint8_t> tmp2 = tmpBuffer3.Get<uint8_t>();
                Duplicate(tmp, 0.0f, length);
                Compare(tmp2, xf, tmp, CMPMODE::EQ, length);
                Select(xf, tmp2, tmp, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
                ReduceSum(xf, xf, tmp, length);
                // Muls(xf, xf, -1.0f, length);
                // LocalTensor<uint32_t> xuint = xf.template ReinterpretCast<uint32_t>();
                // ShiftRight(xuint, xuint, 31u, length);
                // LocalTensor<int32_t> xint = xuint.template ReinterpretCast<int32_t>();
                // Cast(xf, xint, RoundMode::CAST_NONE, length);
                // ReduceSum(xf, xf, tmp, length);
            }
            Cast(zLocal, xf, RoundMode::CAST_NONE, this->ALIGN_NUM);
        }else{
            Abs(xLocal, xLocal, length);
            if(this->ptype == 0){
                Ln(xLocal, xLocal, length);
                Muls(xLocal, xLocal, this->p, length);
                Exp(xLocal, xLocal, length);
                uint64_t mask = this->ALIGN_NUM * 8;
                uint64_t repeat = (length + mask - 1) / mask;
                WholeReduceSum<T>(xLocal, xLocal, mask, repeat, 1, 1, 8);
                WholeReduceSum<T>(xLocal, xLocal, repeat, 1, 1, 1, 8);
            }else if(this->ptype == 2){
                ReduceMax(xLocal, xLocal, tmp, length);
            }else if(this->ptype == 3){
                ReduceMin(xLocal, xLocal, tmp, length);
            }else if(this->ptype == 1){
                LocalTensor<uint8_t> tmp2 = tmpBuffer3.Get<uint8_t>();
                Duplicate(tmp, 0.0f, length);
                Compare(tmp2, xLocal, tmp, CMPMODE::EQ, length);
                Select(xLocal, tmp2, tmp, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
                ReduceSum(xLocal, xLocal, tmp, length);
                // Muls(xLocal, xLocal, -1.0f, length);
                // LocalTensor<uint32_t> xuint = xLocal.template ReinterpretCast<uint32_t>();
                // ShiftRight(xuint, xuint, 31u, length);
                // LocalTensor<int32_t> xint = xuint.template ReinterpretCast<int32_t>();
                // Cast(xLocal, xint, RoundMode::CAST_NONE, length);
                // ReduceSum(xLocal, xLocal, tmp, length);
            }
            DataCopy(zLocal, xLocal, this->ALIGN_NUM);
        }
        

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<T>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        // copy progress_th tile from local tensor to global tensor
        if(this->ptype == 2){
            SetAtomicMax<T>();
            DataCopy(zGm, zLocal, this->ALIGN_NUM);
            SetAtomicNone();
        }else if(this->ptype == 3){
            SetAtomicMin<T>();
            DataCopy(zGm, zLocal, this->ALIGN_NUM);
            SetAtomicNone();
        }else{
            SetAtomicAdd<T>();
            DataCopy(zGm, zLocal, this->ALIGN_NUM);
            SetAtomicNone();
        }
        
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer, tmpBuffer2, tmpBuffer3;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
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
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, ALIGN_NUM * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>(); LocalTensor<T> zLocal2 = outQueueZ.AllocTensor<T>();
        T zero = 0;
        Duplicate(zLocal, zero, this->ALIGN_NUM);
        outQueueZ.EnQue<T>(zLocal); outQueueZ.EnQue<T>(zLocal2);
        zLocal = outQueueZ.DeQue<T>(); zLocal2 = outQueueZ.DeQue<T>();
        DataCopy(zGm, zLocal, this->ALIGN_NUM);
        outQueueZ.FreeTensor(zLocal); outQueueZ.FreeTensor(zLocal2);
        // loop count need to be doubled, due to double buffer
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            ComputeFull(i, this->tileLength);
            CopyOut(i);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length - this->lastpadding);
        CopyOut(loopCount - 1);

        
        zLocal = outQueueZ.AllocTensor<T>(); zLocal2 = outQueueZ.AllocTensor<T>();
        DataCopy(zLocal, zGm, this->ALIGN_NUM);
        Sqrt(zLocal, zLocal, 1);
        outQueueZ.EnQue<T>(zLocal); outQueueZ.EnQue<T>(zLocal2);
        zLocal = outQueueZ.DeQue<T>(); zLocal2 = outQueueZ.DeQue<T>();
        DataCopy(zGm, zLocal, this->ALIGN_NUM);
        outQueueZ.FreeTensor(zLocal); outQueueZ.FreeTensor(zLocal2);
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
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xf = tmpBuffer.Get<float>();
            Cast(xf, xLocal, RoundMode::CAST_NONE, length);
            Mul(xf, xf, xf, length);
            ReduceSum(xf, xf, xf, length);
            Cast(zLocal, xf, RoundMode::CAST_NONE, this->ALIGN_NUM);
        }else{
            Mul(xLocal, xLocal, xLocal, length);
            ReduceSum(xLocal, xLocal, xLocal, length);
            DataCopy(zLocal, xLocal, this->ALIGN_NUM);
        }
        
        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<T>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void ComputeFull(int32_t progress, uint32_t length)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<float> xf = tmpBuffer.Get<float>();
            Cast(xf, xLocal, RoundMode::CAST_NONE, length);
            Mul(xf, xf, xf, length);
            uint64_t mask = 32 / 4 * 8;
            uint64_t repeat = (length + mask - 1) / mask;
            WholeReduceSum<float>(xf, xf, mask, repeat, 1, 1, 8);
            WholeReduceSum<float>(xf, xf, repeat, 1, 1, 1, 8);
            Cast(zLocal, xf, RoundMode::CAST_NONE, this->ALIGN_NUM);
        }else{
            Mul(xLocal, xLocal, xLocal, length);
            uint64_t mask = this->ALIGN_NUM * 8;
            uint64_t repeat = (length + mask - 1) / mask;
            WholeReduceSum<T>(xLocal, xLocal, mask, repeat, 1, 1, 8);
            WholeReduceSum<T>(xLocal, xLocal, repeat, 1, 1, 1, 8);
            DataCopy(zLocal, xLocal, this->ALIGN_NUM);
        }

        // enque the output tensor to VECOUT queue
        outQueueZ.EnQue<T>(zLocal);
        // free input tensors for reuse
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        // copy progress_th tile from local tensor to global tensor
        SetAtomicAdd<T>();
        DataCopy(zGm, zLocal, this->ALIGN_NUM);
        SetAtomicNone();
        // free output tensor for reuse
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
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