
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InstanceNormTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalSize);
  TILING_DATA_FIELD_DEF(uint64_t, batchSize);
  TILING_DATA_FIELD_DEF(uint64_t, stepSize);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InstanceNorm, InstanceNormTilingData)
}
