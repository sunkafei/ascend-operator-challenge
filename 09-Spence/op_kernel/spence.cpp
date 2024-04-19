#include "kernel_operator.h"
#include <type_traits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
template<typename TYPE_X, typename TYPE_Y> class KernelSpence {
    static constexpr float A[] = {4.65128586073990045278E-5,7.31589045238094711071E-3,1.33847639578309018650E-1,8.79691311754530315341E-1,2.71149851196553469920E0,4.25697156008121755724E0,3.29771340985225106936E0,1.00000000000000000126E0};
    static constexpr float B[] = {6.90990488912553276999E-4,2.54043763932544379113E-2,2.82974860602568089943E-1,1.41172597751831069617E0,3.63800533345137075418E0,5.03278880143316990390E0,3.54771340985225096217E0,9.99999999999999998740E-1};
    static constexpr float PIFS = 1.64493406684822643647;
public:
    __aicore__ inline KernelSpence() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? 0 : core_remain);
        this->tileLength = block_size;
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);

        auto startPointer = core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        Gm_x.SetGlobalBuffer((__gm__ TYPE_X*)x + startPointer, bufferlength);
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        pipe.InitBuffer(Q_x, BUFFER_NUM, this->tileLength * sizeof(TYPE_X));
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(B_w, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_tmp1, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_tmp2, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_tmp3, this->tileLength * sizeof(float));
        pipe.InitBuffer(B_bits1, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_bits2, this->tileLength * sizeof(uint8_t));
        pipe.InitBuffer(B_bits3, this->tileLength * sizeof(uint8_t));
        if constexpr (!std::is_same_v<TYPE_X, float>) {
            pipe.InitBuffer(B_fx, this->tileLength * sizeof(float));
            pipe.InitBuffer(B_fy, this->tileLength * sizeof(float));
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
        CopyOut(loopCount - 1, length);
    }
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.AllocTensor<TYPE_X>();
        DataCopy(x, Gm_x[progress * this->tileLength], length);
        Q_x.EnQue(x);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_X> x = Q_x.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> y = Q_y.AllocTensor<TYPE_Y>();
        if constexpr (std::is_same_v<TYPE_X, float>) {
            Calculate(x, y, length);
        }
        else {
            auto fx = B_fx.Get<float>();
            auto fy = B_fy.Get<float>();
            Cast(fx, x, RoundMode::CAST_NONE, length);
            Calculate(fx, fy, length);
            Cast(y, fy, RoundMode::CAST_NONE, length);
        }
        Q_x.FreeTensor(x);
        Q_y.EnQue<TYPE_Y>(y);
    }
    __aicore__ inline void polevlf_A(LocalTensor<float> &dst, LocalTensor<float> &w, uint32_t length) {
        Duplicate(dst, float(0), length);
        for (int i = 0; i < 8; ++i) {
            Mul(dst, dst, w, length);
            Adds(dst, dst, A[i], length);
        }
    }
    __aicore__ inline void polevlf_B(LocalTensor<float> &dst, LocalTensor<float> &w, uint32_t length) {
        Duplicate(dst, float(0), length);
        for (int i = 0; i < 8; ++i) {
            Mul(dst, dst, w, length);
            Adds(dst, dst, B[i], length);
        }
    }
    __aicore__ inline void Calculate(LocalTensor<float> &x, LocalTensor<float> &y, uint32_t length) {
        auto w = B_w.Get<float>();
        auto tmp1 = B_tmp1.Get<float>(), tmp2 = B_tmp2.Get<float>(), tmp3 = B_tmp3.Get<float>();
        auto bits1 = B_bits1.Get<uint8_t>(), bits2 = B_bits2.Get<uint8_t>(), bits3 = B_bits3.Get<uint8_t>();

        Duplicate(tmp1, float(2), length);
        Compare(bits2, x, tmp1, CMPMODE::GT, length); //bits2: x > 2
        Duplicate(tmp2, float(1), length);
        Div(tmp3, tmp2, x, length); //tmp3: 1 / x
        Select(tmp1, bits2, tmp3, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Not(bits3.ReinterpretCast<uint16_t>(), bits2.ReinterpretCast<uint16_t>(), length / 2);
        Select(tmp2, bits3, x, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Or(x.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), length * 2);

        Duplicate(tmp1, float(1.5), length);
        Compare(bits3, x, tmp1, CMPMODE::GT, length); //bits1: x > 1.5
        Adds(tmp3, tmp3, float(-1.0), length); //tmp3: 1 / x - 1
        Select(w, bits3, tmp3, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Or(bits2.ReinterpretCast<uint16_t>(), bits2.ReinterpretCast<uint16_t>(), bits3.ReinterpretCast<uint16_t>(), length / 2); //bits2: flag2

        Duplicate(tmp1, float(0.5), length);
        Compare(bits1, x, tmp1, CMPMODE::LT, length); //bits1: x < 0.5
        Muls(tmp3, x, float(-1), length); //tmp3: -x
        Select(tmp1, bits1, tmp3, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Or(w.ReinterpretCast<uint16_t>(), w.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), length * 2);

        Or(bits3.ReinterpretCast<uint16_t>(), bits1.ReinterpretCast<uint16_t>(), bits3.ReinterpretCast<uint16_t>(), length / 2);
        Not(bits3.ReinterpretCast<uint16_t>(), bits3.ReinterpretCast<uint16_t>(), length / 2);
        Adds(tmp1, x, float(-1.0), length); //tmp1: x - 1
        Select(tmp2, bits3, tmp1, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Or(w.ReinterpretCast<uint16_t>(), w.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), length * 2);

        polevlf_A(tmp1, w, length);
        polevlf_B(tmp2, w, length);
        Muls(tmp1, tmp1, float(-1), length);
        Mul(tmp1, tmp1, w, length);
        Div(tmp2, tmp1, tmp2, length); //tmp2: y

        Ln(tmp1, x, length); //tmp1: log(x)
        Adds(tmp3, tmp3, float(1.0), length); //tmp3: 1 - x
        Ln(tmp3, tmp3, length); //tmp3: log(1 - x)
        Mul(tmp1, tmp1, tmp3, length); //tmp1: log(x) * log(1 - x)
        Muls(tmp1, tmp1, float(-1), length); //tmp1: -log(x) * log(1 - x)
        Adds(tmp1, tmp1, float(PIFS), length); //tmp1: PIFS - log(x) * log(1 - x)
        Sub(tmp1, tmp1, tmp2, length); //tmp1: PIFS - log(x) * log(1 - x) - y
        Select(tmp1, bits1, tmp1, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Not(bits1.ReinterpretCast<uint16_t>(), bits1.ReinterpretCast<uint16_t>(), length / 2);
        Select(tmp2, bits1, tmp2, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Or(tmp2.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), length * 2);

        Ln(tmp1, x, length); //tmp1: z
        Mul(tmp1, tmp1, tmp1, length); //tmp1: z * z
        Muls(tmp1, tmp1, float(-0.5), length); //tmp1: -0.5 * z * z
        Sub(tmp1, tmp1, tmp2, length); //tmp1: -0.5 * z * z - y
        Select(tmp1, bits2, tmp1, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Not(bits2.ReinterpretCast<uint16_t>(), bits2.ReinterpretCast<uint16_t>(), length / 2);
        Select(tmp2, bits2, tmp2, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
        Or(y.ReinterpretCast<uint16_t>(), tmp1.ReinterpretCast<uint16_t>(), tmp2.ReinterpretCast<uint16_t>(), length * 2);

        Duplicate(tmp1, float(0), length);
        Compare(bits1, x, tmp1, CMPMODE::NE, length);
        Select(y, bits1, y, float(PIFS), SELMODE::VSEL_TENSOR_SCALAR_MODE, length);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        DataCopy(Gm_y[progress * this->tileLength], y, length);
        Q_y.FreeTensor(y);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Q_x;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
    TBuf<QuePosition::VECCALC> B_w, B_fx, B_fy, B_tmp1, B_tmp2, B_tmp3;
    TBuf<QuePosition::VECCALC> B_bits1, B_bits2, B_bits3;
    GlobalTensor<TYPE_X> Gm_x;
    GlobalTensor<TYPE_Y> Gm_y;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};
extern "C" __global__ __aicore__ void spence(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSpence<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);
    op.Process();
}