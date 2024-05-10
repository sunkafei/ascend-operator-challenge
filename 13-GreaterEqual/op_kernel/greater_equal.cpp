#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelGreaterEqual {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelGreaterEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_result, this->tileLength * sizeof(half));
        pipe.InitBuffer(B_zero, this->tileLength * sizeof(half));
        this->zero = B_zero.Get<half>();
        Duplicate(this->zero, half(0), this->tileLength);
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto x2 = B_x2.Get<float>();
            Duplicate(x2, float(0), this->tileLength);
        }
        else if constexpr (std::is_same_v<T, float>) {
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto x2 = B_x2.Get<float>();
            Duplicate(x2, float(0), this->tileLength);
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);
        Compute(loopCount - 1, length);
        CopyOut(loopCount - 1, (length + 31) / 32 * 32);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto inty = y.template ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, int8_t>) {
            auto float_x1 = B_x1.Get<half>();
            auto float_x2 = B_x2.Get<half>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Min(float_x2, float_x1, float_x2, length);
            Sub(float_x1, float_x1, float_x2, length);
            Compare(bits, float_x1, zero, CMPMODE::NE, length);
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Cast(inty, result, RoundMode::CAST_ROUND, length);
        }
        else {
            Min(x2, x1, x2, length);
            Sub(x1, x1, x2, length);
            if constexpr (std::is_same_v<T, int32_t>) {
                auto val = B_x1.Get<float>();
                auto float_zero = B_x2.Get<float>();
                Cast(val, x1, RoundMode::CAST_NONE, length);
                Compare(bits, val, float_zero, CMPMODE::NE, length);
            }
            else if constexpr (std::is_same_v<T, float>) {
                auto float_zero = B_x2.Get<float>();
                Compare(bits, x1, float_zero, CMPMODE::NE, length);
            }
            else { //half
                Compare(bits, x1, zero, CMPMODE::NE, length);
            }
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Cast(inty, result, RoundMode::CAST_ROUND, length);
        }
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_result, B_zero, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    LocalTensor<half> zero;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};


template<typename TYPE_X1, typename TYPE_X2, typename TYPE_Y> class KernelGreaterEqualBroadcast {
    using T = TYPE_X1;
public:
    __aicore__ inline KernelGreaterEqualBroadcast() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain, uint32_t* reduce1, uint32_t* reduce2, uint32_t* shape, uint32_t dim) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->reduce1 = reduce1;
        this->reduce2 = reduce2;
        this->totalLength = totalLength;
        this->ALIGN_NUM = ALIGN_NUM;
        this->shape = shape;
        this->dim = dim;

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        uint32_t inTotalLength1 = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce1[i] == 0){
                inTotalLength1 *= this->shape[i];
            }
        }
        inTotalLength1 =(inTotalLength1 + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;

        uint32_t inTotalLength2 = 1;
        for(int i=this->dim-1;i>=0;i--){
            if(this->reduce2[i] == 0){
                inTotalLength2 *= this->shape[i];
            }
        }
        inTotalLength2 =(inTotalLength2 + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;

        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, inTotalLength1);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, inTotalLength2);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x1, 1, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, 1, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, 1, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_bits, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_result, this->tileLength * sizeof(half));
        pipe.InitBuffer(B_zero, this->tileLength * sizeof(half));
        this->zero = B_zero.Get<half>();
        Duplicate(this->zero, half(0), this->tileLength);
        if constexpr (std::is_same_v<T, int32_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto x2 = B_x2.Get<float>();
            Duplicate(x2, float(0), this->tileLength);
        }
        else if constexpr (std::is_same_v<T, float>) {
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(float));
            auto x2 = B_x2.Get<float>();
            Duplicate(x2, float(0), this->tileLength);
        }
        else if constexpr (std::is_same_v<T, int8_t>) {
            pipe.InitBuffer(B_x1, this->tileLength * sizeof(half));
            pipe.InitBuffer(B_x2, this->tileLength * sizeof(half));
        }
    }
    __aicore__ inline void Process() {
        int32_t count = this->totalLength / this->shape[this->dim - 1];
        uint32_t totalLength = this->shape[this->dim - 1];
        this->tileNum = totalLength / this->tileLength + (totalLength % this->tileLength > 0);
        uint32_t d[21] = {0};
        uint32_t dn1[21] = {0};
        uint32_t dn2[21] = {0};
        auto dim = this->dim - 1;
        d[dim] = dn1[dim] = dn2[dim] = 1;
        for(int k=dim-1;k>=0;k--){
            d[k] = d[k + 1] * this->shape[k];
            if(this->reduce1[k] == 0){
                dn1[k] = dn1[k + 1] * this->shape[k];
            }else{
                dn1[k] = dn1[k + 1];
            }
            if(this->reduce2[k] == 0){
                dn2[k] = dn2[k + 1] * this->shape[k];
            }else{
                dn2[k] = dn2[k + 1];
            }
        }
        
        for(int j=0;j<count;j++){
            uint32_t start1 = 0, start2 = 0;
            for(int k=dim-1;k>=0;k--){
                if(this->reduce1[k] == 0){
                    start1 += dn1[k + 1] * (j / d[k + 1] % this->shape[k]);
                }
                if(this->reduce2[k] == 0){
                    start2 += dn2[k + 1] * (j / d[k + 1] % this->shape[k]);
                }
            }
            int32_t loopCount = this->tileNum;
            for (int32_t i = 0; i < loopCount-1; i++) {
                CopyIn(start1 * totalLength, start2 * totalLength, i, this->tileLength);
                Compute(i, this->tileLength);
                CopyOut(j * totalLength, i, this->tileLength);
            }
            uint32_t length = totalLength - this->tileLength * (loopCount - 1);
            auto length_align = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
            CopyIn(start1 * totalLength, start2 * totalLength, loopCount - 1, length_align);
            Compute(loopCount - 1, length);
            CopyOut(j * totalLength, loopCount - 1, (length + 31) / 32 * 32);
        }
        
        
    }
private:
    __aicore__ inline void CopyIn(uint32_t start1, uint32_t start2, int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(x1, Gm_x1[start1 + progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[start2 + progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto inty = y.template ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, int8_t>) {
            auto float_x1 = B_x1.Get<half>();
            auto float_x2 = B_x2.Get<half>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, length);
            Min(float_x2, float_x1, float_x2, length);
            Sub(float_x1, float_x1, float_x2, length);
            Compare(bits, float_x1, zero, CMPMODE::NE, length);
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Cast(inty, result, RoundMode::CAST_ROUND, length);
        }
        else {
            Min(x2, x1, x2, length);
            Sub(x1, x1, x2, length);
            if constexpr (std::is_same_v<T, int32_t>) {
                auto val = B_x1.Get<float>();
                auto float_zero = B_x2.Get<float>();
                Cast(val, x1, RoundMode::CAST_NONE, length);
                Compare(bits, val, float_zero, CMPMODE::NE, length);
            }
            else if constexpr (std::is_same_v<T, float>) {
                auto float_zero = B_x2.Get<float>();
                Compare(bits, x1, float_zero, CMPMODE::NE, length);
            }
            else { //half
                Compare(bits, x1, zero, CMPMODE::NE, length);
            }
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
            Cast(inty, result, RoundMode::CAST_ROUND, length);
        }
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(uint32_t start, int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[start + progress * this->tileLength], y, length);
        // for(int i=0;i<length;i++){
        //     Gm_y.SetValue(start + progress * this->tileLength + i, y.GetValue(i));
        // }
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, 1> Q_y;
    TBuf<QuePosition::VECCALC> B_result, B_zero, B_bits;
    TBuf<QuePosition::VECCALC> B_x1, B_x2;
    LocalTensor<half> zero;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t* reduce1;
    uint32_t* reduce2;
    uint32_t* shape;
    uint32_t dim;
    uint32_t ALIGN_NUM;
};


extern "C" __global__ __aicore__ void greater_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if(TILING_KEY_IS(1)){
        KernelGreaterEqual<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x2, x1, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }else if(TILING_KEY_IS(2)){
        KernelGreaterEqualBroadcast<DTYPE_X1, DTYPE_X2, DTYPE_Y> op;
        op.Init(x2, x1, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain, tiling_data.reduce2, tiling_data.reduce1, tiling_data.shape, tiling_data.dim);
        op.Process();
    }
}