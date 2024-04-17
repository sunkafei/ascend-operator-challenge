#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename T> class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockIdx() < core_remain);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = this->blockLength * GetBlockIdx() + (GetBlockIdx() < core_remain ? GetBlockIdx() : core_remain);
        auto bufferlength = this->blockLength;

        Gm_x.SetGlobalBuffer((__gm__ DTYPE_X*)x + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ DTYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        Gm_clip_value_min.SetGlobalBuffer((__gm__ DTYPE_CLIP_VALUE_MIN*)clip_value_min, 1);
        Gm_clip_value_max.SetGlobalBuffer((__gm__ DTYPE_CLIP_VALUE_MAX*)clip_value_max, 1);
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        this->clip_value_min = Gm_clip_value_min.GetValue(0);
        this->clip_value_max = Gm_clip_value_max.GetValue(0);
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
        LocalTensor<DTYPE_X> x = Q_x.AllocTensor<DTYPE_X>();
        DataCopy(x, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(x);
    }
    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DTYPE_X> x = Q_x.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> y = Q_y.AllocTensor<DTYPE_Y>();
        Mins(x, x, this->clip_value_max, this->length);
        Maxs(y, x, this->clip_value_min, this->length);
        Q_x.FreeTensor(x);
        Q_y.EnQue<DTYPE_Y>(y);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DTYPE_Y> y = Q_y.DeQue<DTYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    GlobalTensor<DTYPE_X> Gm_x;
    GlobalTensor<DTYPE_CLIP_VALUE_MIN> Gm_clip_value_min;
    GlobalTensor<DTYPE_CLIP_VALUE_MAX> Gm_clip_value_max;
    GlobalTensor<DTYPE_Y> Gm_y;
    DTYPE_CLIP_VALUE_MIN clip_value_min;
    DTYPE_CLIP_VALUE_MAX clip_value_max;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t length;
};

extern "C" __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelClipByValue<DTYPE_X> op;
    op.Init(x, clip_value_min, clip_value_max, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}