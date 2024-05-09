
#include "cross_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    CrossTilingData tiling;
    int64_t dim = *context->GetAttrs()->GetInt(0);
    uint64_t totalSize[3] = {1, 1, 1};
    uint64_t batchSize[3] = {1, 1, 1};
    uint64_t stepSize[3] = {1, 1, 1};
    for (int k = 0; k < 2; ++k) {
        const gert::StorageShape* shape = context->GetInputShape(k);
        for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
            totalSize[k] *= shape->GetStorageShape().GetDim(i);
            if (i < dim) {
                batchSize[k] *= shape->GetStorageShape().GetDim(i);
            }
            if (i > dim) {
                stepSize[k] *= shape->GetStorageShape().GetDim(i);
            }
        }
    }
    tiling.set_totalSize(totalSize);
    tiling.set_batchSize(batchSize);
    tiling.set_stepSize(stepSize);
    std::cout << "totalSize: " << totalSize[0] << ", " << totalSize[1] << std::endl;
    std::cout << "batchSize: " << batchSize[0] << ", " << batchSize[1] << std::endl;
    std::cout << "stepSize: " << stepSize[0] << ", " << stepSize[1] << std::endl;

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
class Cross : public OpDef {
public:
    explicit Cross(const char* name) : OpDef(name)
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
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-65530);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Cross);
}
