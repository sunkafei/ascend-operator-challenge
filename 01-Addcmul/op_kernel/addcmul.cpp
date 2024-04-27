#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
template<typename T> struct Map {using type = T;};
template<> struct Map<int8_t> {using type = half;};
template<typename TYPE_INPUT_DATA, typename TYPE_X1, typename TYPE_X2, typename TYPE_VALUE, typename TYPE_Y> class KernelAddcmul {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_input_data.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)input_data + startPointer, bufferlength);
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + startPointer, bufferlength);
        Gm_value.SetGlobalBuffer((__gm__ TYPE_VALUE*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_input_data, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmp2, this->tileLength * sizeof(half));
        this->value = Gm_value.GetValue(0);
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
        CopyOut(loopCount - 1, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.AllocTensor<TYPE_INPUT_DATA>();
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(input_data, Gm_input_data[progress * this->tileLength], length);
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_input_data.EnQue(input_data);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.DeQue<TYPE_INPUT_DATA>();
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, signed char>) {
            auto p1 = tmp1.Get<half>();
            auto p2 = tmp2.Get<half>();
            Cast(p1, x1, RoundMode::CAST_NONE, length);
            Cast(p2, x2, RoundMode::CAST_NONE, length);
            Mul(p1, p1, p2, length);
            Muls(p1, p1, value, length);
            Cast(p2, input_data, RoundMode::CAST_NONE, length);
            Add(p1, p1, p2, length);
            Cast(y, p1, RoundMode::CAST_NONE, length);
        }
        else {
            Mul(x1, x1, x2, length);
            Muls(x1, x1, value, length);
            Add(y, x1, input_data, length);
        }
        Q_input_data.FreeTensor(input_data);
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
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_input_data, Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<TYPE_INPUT_DATA> Gm_input_data;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_VALUE> Gm_value;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    typename Map<TYPE_VALUE>::type value;
};
template<typename TYPE_INPUT_DATA, typename TYPE_X1, typename TYPE_X2, typename TYPE_VALUE, typename TYPE_Y> class KernelAddcmul_Broadcast {
    using T = TYPE_Y;
public:
    __aicore__ inline KernelAddcmul_Broadcast() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t input_data_length, uint32_t x1_length, uint32_t x2_length, uint32_t total_length, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->inputdataLength = input_data_length;
        this->x1Length = x1_length;
        this->x2Length = x2_length;
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->startPointer = core_size * GetBlockIdx();
        //auto bufferlength = this->blockLength;

        Gm_input_data.SetGlobalBuffer((__gm__ TYPE_INPUT_DATA*)input_data, total_length);
        Gm_x1.SetGlobalBuffer((__gm__ TYPE_X1*)x1, total_length);
        Gm_x2.SetGlobalBuffer((__gm__ TYPE_X2*)x2, total_length);
        Gm_value.SetGlobalBuffer((__gm__ TYPE_VALUE*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y, total_length);

        pipe.InitBuffer(Q_input_data, BUFFER_NUM, this->tileLength * sizeof(TYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(TYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(TYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmp2, this->tileLength * sizeof(half));
        this->value = Gm_value.GetValue(0);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        for (int32_t i = 0; i < loopCount-1; i++) {
            uint32_t position = startPointer + i * this->tileLength;
            CopyIn(position, this->tileLength);
            Compute(this->tileLength);
            CopyOut(position, this->tileLength);
        }
        uint32_t position = startPointer + (loopCount - 1) * this->tileLength;
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(position, length);
        Compute(length);
        CopyOut(position, length);
    }

private:
    __aicore__ inline void CopyIn(int32_t position, uint32_t length) {
        LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.AllocTensor<TYPE_INPUT_DATA>();
        LocalTensor<TYPE_X1> x1 = Q_x1.AllocTensor<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.AllocTensor<TYPE_X2>();
        DataCopy(input_data, Gm_input_data[position % inputdataLength], length);
        DataCopy(x1, Gm_x1[position % x1Length], length);
        DataCopy(x2, Gm_x2[position % x2Length], length);
        Q_input_data.EnQue(input_data);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<TYPE_INPUT_DATA> input_data = Q_input_data.DeQue<TYPE_INPUT_DATA>();
        LocalTensor<TYPE_X1> x1 = Q_x1.DeQue<TYPE_X1>();
        LocalTensor<TYPE_X2> x2 = Q_x2.DeQue<TYPE_X2>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, signed char>) {
            auto p1 = tmp1.Get<half>();
            auto p2 = tmp2.Get<half>();
            Cast(p1, x1, RoundMode::CAST_NONE, length);
            Cast(p2, x2, RoundMode::CAST_NONE, length);
            Mul(p1, p1, p2, length);
            Muls(p1, p1, value, length);
            Cast(p2, input_data, RoundMode::CAST_NONE, length);
            Add(p2, p1, p2, length);

            Cast(p1.ReinterpretCast<int16_t>(), p2, RoundMode::CAST_RINT, length);
            ShiftLeft(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), length);
            ShiftRight(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), length);
            Cast(p2, p1.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, length);
            Cast(y, p2, RoundMode::CAST_NONE, length);
        }
        else {
            Mul(x1, x1, x2, length);
            Muls(x1, x1, value, length);
            Add(y, x1, input_data, length);
        }
        Q_input_data.FreeTensor(input_data);
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t position, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[position], y, length);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_input_data, Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<TYPE_INPUT_DATA> Gm_input_data;
    GlobalTensor<TYPE_X1> Gm_x1;
    GlobalTensor<TYPE_X2> Gm_x2;
    GlobalTensor<TYPE_VALUE> Gm_value;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t startPointer;
    uint32_t inputdataLength;
    uint32_t x1Length;
    uint32_t x2Length;
    typename Map<TYPE_VALUE>::type value;
};
extern "C" __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.input_data_length == tiling_data.total_length && tiling_data.x1_length == tiling_data.total_length && tiling_data.x2_length == tiling_data.total_length) {
        KernelAddcmul<DTYPE_INPUT_DATA, DTYPE_X1, DTYPE_X2, DTYPE_VALUE, DTYPE_Y> op;
        op.Init(input_data, x1, x2, value, y, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
    else {
        KernelAddcmul_Broadcast<DTYPE_INPUT_DATA, DTYPE_X1, DTYPE_X2, DTYPE_VALUE, DTYPE_Y> op;
        op.Init(input_data, x1, x2, value, y, tiling_data.input_data_length, tiling_data.x1_length, tiling_data.x2_length, tiling_data.total_length, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
        op.Process();
    }
}