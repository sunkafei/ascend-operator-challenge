#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
class KernelAddcmul {
public:
    __aicore__ inline KernelAddcmul() {}
    __aicore__ inline void Init(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockIdx() < core_remain);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = this->blockLength * GetBlockIdx() + (GetBlockIdx() < core_remain ? GetBlockIdx() : core_remain);
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        Gm_input_data.SetGlobalBuffer((__gm__ DTYPE_INPUT_DATA*)input_data + startPointer, this->blockLength);
        Gm_x1.SetGlobalBuffer((__gm__ DTYPE_X1*)x1 + startPointer, this->blockLength);
        Gm_x2.SetGlobalBuffer((__gm__ DTYPE_X2*)x2 + startPointer, this->blockLength);
        Gm_value.SetGlobalBuffer((__gm__ DTYPE_VALUE*)value, 1);
        Gm_y.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, this->blockLength);
        pipe.InitBuffer(Q_input_data, BUFFER_NUM, this->tileLength * sizeof(DTYPE_INPUT_DATA));
        pipe.InitBuffer(Q_x1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X1));
        pipe.InitBuffer(Q_x2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X2));
        //pipe.InitBuffer(Q_value, BUFFER_NUM, this->tileLength * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        //pipe.InitBuffer(Q_tmp, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        this->value = Gm_value.GetValue(0);
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<DTYPE_INPUT_DATA> input_data = Q_input_data.AllocTensor<DTYPE_INPUT_DATA>();
        LocalTensor<DTYPE_X1> x1 = Q_x1.AllocTensor<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2 = Q_x2.AllocTensor<DTYPE_X2>();
        //LocalTensor<DTYPE_VALUE> value = Q_value.AllocTensor<DTYPE_VALUE>();
        DataCopy(input_data, Gm_input_data[progress * this->tileLength], this->tileLength);
        DataCopy(x1, Gm_x1[progress * this->tileLength], this->tileLength);
        DataCopy(x2, Gm_x2[progress * this->tileLength], this->tileLength);
        //DataCopy(value, Gm_value[progress * this->tileLength], this->tileLength);
        Q_input_data.EnQue(input_data);
        Q_x1.EnQue(x1);
        Q_x2.EnQue(x2);
        //Q_value.EnQue(value);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DTYPE_INPUT_DATA> input_data = Q_input_data.DeQue<DTYPE_INPUT_DATA>();
        LocalTensor<DTYPE_X1> x1 = Q_x1.DeQue<DTYPE_X1>();
        LocalTensor<DTYPE_X2> x2 = Q_x2.DeQue<DTYPE_X2>();
        //LocalTensor<DTYPE_VALUE> value = Q_value.DeQue<DTYPE_VALUE>();
        LocalTensor<DTYPE_Y> y = Q_y.AllocTensor<DTYPE_Y>();
        Mul(x1, x1, x2, this->tileLength);
        Muls(x1, x1, value, this->tileLength);
        Add(y, x1, input_data, this->tileLength);
        Q_input_data.FreeTensor(input_data);
        Q_x1.FreeTensor(x1);
        Q_x2.FreeTensor(x2);
        //Q_value.FreeTensor(value);
        Q_y.EnQue<DTYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DTYPE_Y> y = Q_y.DeQue<DTYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, this->tileLength);
        Q_y.FreeTensor(y);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_input_data, Q_x1, Q_x2/*, Q_value*/;
    //TQue<QuePosition::VECCALC, BUFFER_NUM> Q_tmp;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    GlobalTensor<DTYPE_INPUT_DATA> Gm_input_data;
    GlobalTensor<DTYPE_X1> Gm_x1;
    GlobalTensor<DTYPE_X2> Gm_x2;
    GlobalTensor<DTYPE_VALUE> Gm_value;
    GlobalTensor<DTYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    DTYPE_VALUE value;
};
extern "C" __global__ __aicore__ void addcmul(GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddcmul op;
    op.Init(input_data, x1, x2, value, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}