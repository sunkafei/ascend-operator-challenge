#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
template<typename T> struct Map {using type = T;};
template<> struct Map<int8_t> {using type = half;};
template<typename T> class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockIdx() < core_remain);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = this->blockLength * GetBlockIdx() + (GetBlockIdx() < core_remain ? GetBlockIdx() : core_remain);
        auto bufferlength = this->blockLength;

        Gm_input_data.SetGlobalBuffer((__gm__ DTYPE_INPUT_DATA*)input_data + startPointer, bufferlength);
        Gm_x1.SetGlobalBuffer((__gm__ DTYPE_X1*)x1 + startPointer, bufferlength);
        Gm_x2.SetGlobalBuffer((__gm__ DTYPE_X2*)x2 + startPointer, bufferlength);
        Gm_value.SetGlobalBuffer((__gm__ DTYPE_VALUE*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_input_data, BUFFER_NUM, this->tileLength * sizeof(DTYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X2));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmp1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmp2, this->tileLength * sizeof(half));
        this->value = Gm_value.GetValue(0);
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
        LocalTensor<DTYPE_INPUT_DATA> input_data = Q_input_data.AllocTensor<DTYPE_INPUT_DATA>();
        LocalTensor<DTYPE_X1> x1 = Q_x1.AllocTensor<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2 = Q_x2.AllocTensor<DTYPE_X2>();
        DataCopy(input_data, Gm_input_data[progress * this->tileLength], length);
        DataCopy(x1, Gm_x1[progress * this->tileLength], length);
        DataCopy(x2, Gm_x2[progress * this->tileLength], length);
        Q_input_data.EnQue(input_data);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DTYPE_INPUT_DATA> input_data = Q_input_data.DeQue<DTYPE_INPUT_DATA>();
        LocalTensor<DTYPE_X1> x1 = Q_x1.DeQue<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2 = Q_x2.DeQue<DTYPE_X2>();
        LocalTensor<DTYPE_Y> y = Q_y.AllocTensor<DTYPE_Y>();
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, signed char>) {
            auto p1 = tmp1.Get<half>();
            auto p2 = tmp2.Get<half>();
            Cast(p1, x1, RoundMode::CAST_NONE, length);
            Cast(p2, x2, RoundMode::CAST_NONE, length);
            Mul(p1, p1, p2, length);
            Muls(p1, p1, value, this->length);
            Cast(p2, input_data, RoundMode::CAST_NONE, length);
            Add(p1, p1, p2, length);
            Cast(y, p1, RoundMode::CAST_NONE, length);
        }
        else {
            Mul(x1, x1, x2, length);
            Muls(x1, x1, value, this->length);
            Add(y, x1, input_data, length);
        }
        Q_input_data.FreeTensor(input_data);
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
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_input_data, Q_x1, Q_x2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<DTYPE_INPUT_DATA> Gm_input_data;
    GlobalTensor<DTYPE_X1> Gm_x1;
    GlobalTensor<DTYPE_X2> Gm_x2;
    GlobalTensor<DTYPE_VALUE> Gm_value;
    GlobalTensor<DTYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t length;
    Map<DTYPE_VALUE>::type value;
};
extern "C" __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddcmul<DTYPE_Y> op;
    op.Init(input_data, x1, x2, value, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}