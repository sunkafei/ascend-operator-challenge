
#include "instance_norm_tiling.h"
#include "register/op_def_registry.h"
#include <vector>
#include <cstring>
#include <cstdlib>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    InstanceNormTilingData tiling;

    uint64_t totalSize[3] = {};
    uint64_t batchSize[3] = {};
    uint64_t stepSize[3] = {};
    uint64_t length = 0;
    for (int i = 0; i < 3; ++i)
        length = std::max<uint64_t>(length, context->GetInputShape(i)->GetStorageShape().GetDimNum());
    for (int i = 0; i < 3; ++i) {
        totalSize[i] = context->GetInputTensor(i)->GetShapeSize();
        const gert::StorageShape* shape = context->GetInputShape(i);
        const char *str = context->GetAttrs()->GetAttrPointer<char>(0);
        std::vector<uint64_t> dim(length, 1);
        int n = length;
        for (int j = shape->GetStorageShape().GetDimNum() - 1; j >= 0; --j) {
            dim[--n] = shape->GetStorageShape().GetDim(j);
        }
        if (strcmp(str, "NDHWC") == 0) {
            batchSize[i] = dim[0];
            stepSize[i] = dim[4];
        }
        else if(strcmp(str, "NCDHW") == 0) {
            batchSize[i] = dim[0] * dim[1];
            stepSize[i] = 1;
        }
        else if(strcmp(str, "NHWC") == 0) {
            batchSize[i] = dim[0];
            stepSize[i] = dim[3];
        }
        else if(strcmp(str, "NCHW") == 0) {
            batchSize[i] = dim[0] * dim[1];
            stepSize[i] = 1;
        }
        else { // ND
            batchSize[i] = dim[0] * dim[1];
            stepSize[i] = 1;
        }
    }
    auto ptr = context->GetAttrs()->GetFloat(1);
    tiling.set_epsilon(*ptr);
    tiling.set_totalSize(totalSize);
    tiling.set_batchSize(batchSize);
    tiling.set_stepSize(stepSize);

    for (int i = 0; i < 3; ++i) {
        std::cout << batchSize[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << stepSize[i] << " ";
    }
    std::cout << std::endl;

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
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
class InstanceNorm : public OpDef {
public:
    explicit InstanceNorm(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mean")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("variance")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
        this->Attr("epsilon").AttrType(OPTIONAL).Float(0.0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(InstanceNorm);
}
