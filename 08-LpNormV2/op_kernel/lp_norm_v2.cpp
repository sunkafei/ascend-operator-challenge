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

template<typename T> class KernelLpNormV2Axes {
public:
    __aicore__ inline KernelLpNormV2Axes() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, float p, uint32_t ptype, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t* reduce, uint32_t* shape, uint32_t dim)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->totalLength = totalLength;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->p = p;
        this->invp = 1.0f / p;
        this->ptype = ptype;
        this->reduce = reduce;
        this->shape = shape;
        this->dim = dim;
        this->ALIGN_NUM = ALIGN_NUM;
        this->ALIGN256 = ALIGN_NUM * 8;
        this->lastpadding = (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        uint32_t outTotalLength = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce[i] == 0){
                outTotalLength *= this->shape[i];
            }
        }
        outTotalLength =(outTotalLength + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;

        // get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T*)x + startPointer, bufferlength);
        zGm.SetGlobalBuffer((__gm__ T*)y + startPointer, outTotalLength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(SumQueue, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ2, 1, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        // tiling strategy, pipeline parallel
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(0, i, this->tileLength);
            PreCal(i, this->tileLength);
            CopyPreOut(i, this->tileLength);
        }
        auto length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(0, loopCount - 1, length);
        PreCal(loopCount - 1, length - this->lastpadding);
        CopyPreOut(loopCount - 1, length);

        uint32_t outTotalLength = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce[i] == 0){
                outTotalLength *= this->shape[i];
            }
        }

        {
            int32_t loopCount = (outTotalLength + this->tileLength - 1) / this->tileLength;
            for (int32_t i = 0; i < loopCount-1; i++) {
                InitzGm(i, this->tileLength);
            }
            auto length = outTotalLength - this->tileLength * (loopCount - 1);
            length = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
            InitzGm(loopCount - 1, length);
        }

        bool copyed = false;
        uint32_t sufDim = 1;
        uint32_t preDim = this->totalLength;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce[i] == 0){
                preDim /= this->shape[i];
                sufDim *= this->shape[i];
            }else{
                if(sufDim == 1){
                    preDim /= this->shape[i];
                    auto length = (this->shape[i] + this->ALIGN256 - 1) / this->ALIGN256 * this->ALIGN256;
                    int32_t loopCount = (length + this->tileLength - 1) / this->tileLength;
                    for(int j=0;j<preDim;j++){
                        InitTensor(length < this->tileLength ? length : this->tileLength);
                        for(int k=0;k<loopCount-1;k++){
                            CopyIn(j * this->shape[i], k, this->tileLength);
                            Compute(i, this->tileLength);
                        }
                        auto L = this->shape[i] - this->tileLength * (loopCount - 1);

                        CopyIn(j * this->shape[i], loopCount - 1, (L + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM);
                        Compute(i, L);
                        CopyOut(j, length < this->tileLength ? length : this->tileLength);
                    }
                }else{
                    copyed = true;
                    auto length = (sufDim + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                    uint32_t loopCount = (length + this->tileLength - 1) / this->tileLength;
                    uint32_t d[11] = {0};
                    uint32_t dn[11] = {0};
                    d[i + 1] = dn[i + 1] = 1;
                    for(int k=i;k>=0;k--){
                        d[k] = d[k + 1] * this->shape[k];
                        if(this->reduce[k] == 0){
                            dn[k] = dn[k + 1] * this->shape[k];
                        }else{
                            dn[k] = dn[k + 1];
                        }
                    }
                    for(int j=0;j<preDim;j++){
                        uint32_t newp = 0;
                        for(int k=i;k>=0;k--){
                            if(this->reduce[k] == 0){
                                newp += dn[k + 1] * (j / d[k + 1] % this->shape[k]);
                            }
                        }

                        for(int k=0;k<loopCount-1;k++){
                            CopyIn2(j * sufDim, k, this->tileLength);
                            CopyValue(k, this->tileLength, 0);
                            CopyOut2(newp * sufDim, k, this->tileLength, 0);
                        }
                        auto L = sufDim - this->tileLength * (loopCount - 1);
                        auto L2 = (L + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                        CopyIn2(j * sufDim, loopCount - 1, L2);
                        CopyValue(loopCount - 1, L2, L2 - L);
                        CopyOut2(newp * sufDim, loopCount - 1, L2, L2 - L);
                    }

                    break;
                }
            }
        }
        
        if(!copyed){
            LocalTensor<T> sum = SumQueue.AllocTensor<T>();

            int32_t loopCount = (outTotalLength + this->tileLength - 1) / this->tileLength;
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyAnsIn(xGm, i, this->tileLength);
                CalAns(i, this->tileLength);
                CopyAns(i, this->tileLength);
            }
            auto length = outTotalLength - this->tileLength * (loopCount - 1);
            length = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
            CopyAnsIn(xGm, loopCount - 1, length);
            CalAns(loopCount - 1, length);
            CopyAns(loopCount - 1, length);

            SumQueue.FreeTensor(sum);
        }else{
            LocalTensor<T> sum = SumQueue.AllocTensor<T>();

            int32_t loopCount = (outTotalLength + this->tileLength - 1) / this->tileLength;
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyAnsIn(zGm, i, this->tileLength);
                CalAns(i, this->tileLength);
                CopyAns(i, this->tileLength);
            }
            auto length = outTotalLength - this->tileLength * (loopCount - 1);
            length = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
            CopyAnsIn(zGm, loopCount - 1, length);
            CalAns(loopCount - 1, length);
            CopyAns(loopCount - 1, length);

            SumQueue.FreeTensor(sum);
        }
    }

private:
    __aicore__ inline void InitTensor(uint32_t length)
    {
        LocalTensor<T> sum = SumQueue.AllocTensor<T>();
        T zero = this->ptype == 3 ? 3.40282346638528859811704183484516925e+38F : 0;
        Duplicate(sum, zero, length);
        SumQueue.EnQue<T>(sum);
    }
    __aicore__ inline void InitzGm(int32_t progress, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        T zero = this->ptype == 3 ? 3.40282346638528859811704183484516925e+38F : 0;
        Duplicate(zLocal, zero, length);
        outQueueZ.EnQue<T>(zLocal);
        zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyIn(int32_t start, int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[start + progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> sum = SumQueue.DeQue<T>();
        
        if(this->ptype == 0){
            Add(sum, xLocal, sum, length);
        }else if(this->ptype == 2){
            Max(sum, xLocal, sum, length);
        }else if(this->ptype == 3){
            Min(sum, xLocal, sum, length);
        }else if(this->ptype == 1){
            Add(sum, xLocal, sum, length);
        }
        
        SumQueue.EnQue<T>(sum);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t j, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ2.AllocTensor<T>();
        DataCopy(zLocal, xGm[j], this->ALIGN_NUM);
        LocalTensor<T> sum = SumQueue.DeQue<T>();
        LocalTensor<T> tmp = tmpBuffer.Get<T>();
        if(this->ptype == 0 || this->ptype == 1){
            ReduceSum(sum, sum, tmp, length);
        }else if(this->ptype == 2){
            ReduceMax(sum, sum, tmp, length);
        }else if(this->ptype == 3){
            ReduceMin(sum, sum, tmp, length);
        }
        outQueueZ2.EnQue(zLocal);
        zLocal = outQueueZ2.DeQue<T>();
        
        zLocal.SetValue(0, sum.GetValue(0));
        
        outQueueZ2.EnQue(zLocal);
        zLocal = outQueueZ2.DeQue<T>();
        DataCopy(xGm[j], zLocal, this->ALIGN_NUM);
        //xGm.SetValue(j, sum.GetValue(0));
        SumQueue.FreeTensor(sum);
        outQueueZ2.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyIn2(int32_t start, int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[start + progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CopyValue(int32_t progress, uint32_t length, uint32_t padding)
    {
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        DataCopy(zLocal, xLocal, length);
        if(padding){
            T zero = this->ptype == 3 ? 3.40282346638528859811704183484516925e+38F : 0;
            // Duplicate(xLocal, zero, padding);
            for(int i=length-padding;i<length;i++){
                zLocal.SetValue(i, zero);
            }
        }
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut2(int32_t start, int32_t progress, uint32_t length, uint32_t padding)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        if(this->ptype == 0 || this->ptype == 1){
            SetAtomicAdd<T>();
            DataCopy(zGm[start + progress * this->tileLength], zLocal, length);
            SetAtomicNone();
        }else if(this->ptype == 2){
            SetAtomicMax<T>();
            DataCopy(zGm[start + progress * this->tileLength], zLocal, length);
            SetAtomicNone();
        }else if(this->ptype == 3){
            SetAtomicMin<T>();
            DataCopy(zGm[start + progress * this->tileLength], zLocal, length);
            SetAtomicNone();
        }
        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyAnsIn(GlobalTensor<T> &Gm, int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, Gm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CalAns(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        if(this->ptype == 0){
            T invp = this->invp;
            Ln(xLocal, xLocal, length);
            Muls(xLocal, xLocal, invp, length);
            Exp(xLocal, xLocal, length);
        }
        DataCopy(zLocal, xLocal, length);
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyAns(int32_t progress, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void PreCal(int32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        
        Abs(zLocal, xLocal, length);
        if(this->ptype == 0){
            Ln(zLocal, zLocal, length);
            Muls(zLocal, zLocal, this->p, length);
            Exp(zLocal, zLocal, length);
        }else if(this->ptype == 1){
            LocalTensor<T> tmp = tmpBuffer.Get<T>();
            LocalTensor<uint8_t> tmp2 = tmpBuffer2.Get<uint8_t>();
            Duplicate(tmp, (T)0.0, length);
            Compare(tmp2, zLocal, tmp, CMPMODE::EQ, length);
            Select(zLocal, tmp2, tmp, (T)1.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        }
        
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyPreOut(int32_t progress, uint32_t length)
    {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(xGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer, tmpBuffer2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TQue<QuePosition::VECOUT, 1> SumQueue, outQueueZ2;
    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;
    T p;
    T invp;
    uint32_t totalLength;
    uint32_t ptype;
    uint32_t ALIGN_NUM, ALIGN256;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint8_t lastpadding;
    uint32_t* reduce;
    uint32_t* shape;
    uint32_t dim;
};

extern "C" __global__ __aicore__ void lp_norm_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(TILING_KEY_IS(1)){
        if(tiling_data.p == 2){
            KernelLpNormV2P2<DTYPE_X> op;
            op.Init(x, y, tiling_data.p, tiling_data.ptype, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
            op.Process();
        }else{
            KernelLpNormV2<DTYPE_X> op;
            op.Init(x, y, tiling_data.p, tiling_data.ptype, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
            op.Process();
        }
    }else if(TILING_KEY_IS(2)){
        KernelLpNormV2Axes<DTYPE_X> op;
        op.Init(x, y, tiling_data.p, tiling_data.ptype, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.reduce, tiling_data.shape, tiling_data.dim);
        op.Process();
    }
    
}