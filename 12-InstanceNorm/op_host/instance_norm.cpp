
#include "instance_norm_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    InstanceNormTilingData tiling;

    const gert::StorageShape* shape = context->GetInputShape(0);
    uint32_t totalSize = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
        totalSize *= shape->GetStorageShape().GetDim(i);
    }
    tiling.set_totalSize(totalSize);
    const char *str = context->GetAttrs()->GetAttrPointer<char>(0);
    if (strcmp(str, "NDHWC") == 0) {
        tiling.set_batchSize(shape->GetStorageShape().GetDim(0));
        tiling.set_stepSize(shape->GetStorageShape().GetDim(4));
    }
    else if(strcmp(str, "NCDHW") == 0) {
        tiling.set_batchSize(shape->GetStorageShape().GetDim(0) * shape->GetStorageShape().GetDim(1));
        tiling.set_stepSize(1);
    }
    else if(strcmp(str, "NHWC") == 0) {
        tiling.set_batchSize(shape->GetStorageShape().GetDim(0));
        tiling.set_stepSize(shape->GetStorageShape().GetDim(3));
    }
    else if(strcmp(str, "NCHW") == 0) {
        tiling.set_batchSize(shape->GetStorageShape().GetDim(0) * shape->GetStorageShape().GetDim(1));
        tiling.set_stepSize(1);
    }
    else { // ND
        tiling.set_batchSize(shape->GetStorageShape().GetDim(0) * shape->GetStorageShape().GetDim(1));
        tiling.set_stepSize(1);
    }

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
