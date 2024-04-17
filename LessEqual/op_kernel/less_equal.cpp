#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T> class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockIdx() < core_remain);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = this->blockLength * GetBlockIdx() + (GetBlockIdx() < core_remain ? GetBlockIdx() : core_remain);
        auto bufferlength = this->blockLength;

        Gm_x1.SetGlobalBuffer((__gm__ DTYPE_X1*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ DTYPE_X2*)x2 + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
            this->length = this->tileLength;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<DTYPE_X1> x1 = Q_x1.AllocTensor<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2 = Q_x2.AllocTensor<DTYPE_X2>();
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DTYPE_X1> x1 = Q_x1.DeQue<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2 = Q_x2.DeQue<DTYPE_X2>();
        LocalTensor<DTYPE_Y> y = Q_y.AllocTensor<DTYPE_Y>();
        auto bits = B_bits.Get<uint8_t>();
        auto result = B_result.Get<half>();
        auto inty = y.ReinterpretCast<uint8_t>();
        if constexpr (std::is_same_v<T, int8_t>) {
            auto float_x1 = B_x1.Get<half>();
            auto float_x2 = B_x2.Get<half>();
            Cast(float_x1, x1, RoundMode::CAST_NONE, this->length);
            Cast(float_x2, x2, RoundMode::CAST_NONE, this->length);
            Min(float_x2, float_x1, float_x2, this->length);
            Sub(float_x1, float_x1, float_x2, this->length);
            Compare(bits, float_x1, zero, CMPMODE::NE, this->length);
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->length);
            Cast(inty, result, RoundMode::CAST_ROUND, this->length);
        }
        else {
            Min(x2, x1, x2, this->length);
            Sub(x1, x1, x2, this->length);
            if constexpr (std::is_same_v<T, int32_t>) {
                auto val = B_x1.Get<float>();
                auto float_zero = B_x2.Get<float>();
                Cast(val, x1, RoundMode::CAST_NONE, this->length);
                Compare(bits, val, float_zero, CMPMODE::NE, this->length);
            }
            else if constexpr (std::is_same_v<T, float>) {
                auto float_zero = B_x2.Get<float>();
                Compare(bits, x1, float_zero, CMPMODE::NE, this->length);
            }
            else { //half
                Compare(bits, x1, zero, CMPMODE::NE, this->length);
            }
            Select(result, bits, zero, half(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->length);
            Cast(inty, result, RoundMode::CAST_ROUND, this->length);
        }
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<DTYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DTYPE_Y> y = Q_y.DeQue<DTYPE_Y>();
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
    GlobalTensor<DTYPE_X1> Gm_x1;
    GlobalTensor<DTYPE_X2> Gm_x2;
    GlobalTensor<DTYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t length;
};
extern "C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLessEqual<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}