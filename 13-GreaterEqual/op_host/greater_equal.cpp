
#include "greater_equal_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
const uint32_t BLOCK_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GreaterEqualTilingData tiling;
    int32_t NUM = 24;
    uint32_t sizeofdatatype;
    uint32_t totalLengthAligned;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();

    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    auto dt = context->GetInputTensor(0)->GetDataType();
    if(dt == ge::DT_INT8){
        sizeofdatatype = 1;
        NUM = 15;
    }else if(dt == ge::DT_FLOAT16 || dt == ge::DT_BF16){
        sizeofdatatype = 2;
        NUM = 9;
    }
    else if (dt == ge::DT_INT32) {
        sizeofdatatype = 4;
        NUM = 10;
    }
    else{ //DT_FLOAT
        sizeofdatatype = 4;
        NUM = 9;
    }

    auto inshape1 = context->GetInputShape(0)->GetOriginShape();
    auto inshape2 = context->GetInputShape(1)->GetOriginShape();
    auto outshape = context->GetOutputShape(0)->GetOriginShape();
    bool flag = false;
    for(int i=0;i<outshape.GetDimNum();i++){
        if(inshape1.GetDim(i) != inshape2.GetDim(i)) flag = true;
    }
    
    auto axes = context->GetAttrs()->GetListInt(1);
    if(flag){
        context->SetTilingKey(2);
        uint32_t reduce1[20] = {0};
        uint32_t reduce2[20] = {0};
        uint32_t shape[20] = {0};
        uint32_t d = 1;
        for(int i=0;i<outshape.GetDimNum();i++){
            shape[i] = outshape.GetDim(i);
            d *= outshape.GetDim(i);
            if(inshape1.GetDim(i) != outshape.GetDim(i)) reduce1[i] = 1;
            if(inshape2.GetDim(i) != outshape.GetDim(i)) reduce2[i] = 1;
        }
        uint32_t dim = outshape.GetDimNum();
        for(int i=dim-1;i>=1;i--){
            if(!reduce1[i - 1] && !reduce2[i - 1] && !reduce1[i] && !reduce2[i]){
                dim--;
                shape[i - 1] *= shape[i];
            }else{
                break;
            }
        }
        if(reduce1[dim - 1] || reduce2[dim - 1]){
            shape[dim] = 1;
            dim++;
        }

        tiling.set_shape(shape);
        tiling.set_reduce1(reduce1);
        tiling.set_reduce2(reduce2);
        tiling.set_dim(dim);
        aivNum = 1;
        totalLength = d;
    }else{
        context->SetTilingKey(1);
    }

    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= 8 ? tiling_size : tiling_size / 8 * 8;

    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < totalLength / block_size) ? aivNum : (totalLength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;

    uint32_t core_size = (totalLength / aivNum) / (ALIGN_NUM * 8) * (ALIGN_NUM * 8);
    uint32_t core_remain = totalLength - aivNum * core_size;

    tiling.set_totalLength(totalLength);
    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);

    context->SetBlockDim(aivNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}



namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class GreaterEqual : public OpDef {
public:
    explicit GreaterEqual(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(GreaterEqual);
}
