
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InstanceNormTilingData)
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 3, totalSize);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 3, batchSize);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 3, stepSize);
  TILING_DATA_FIELD_DEF(uint64_t, packNumber);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InstanceNorm, InstanceNormTilingData)
}
