
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InstanceNormTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalSize);
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, stepSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InstanceNorm, InstanceNormTilingData)
}
