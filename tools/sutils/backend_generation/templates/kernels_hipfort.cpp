#include <hip/hip_runtime.h>
#include "../hip_utils.h"  
#include "../amd_arrays.h"
#include "index_define.h"

% for k in kernels:
${k}
% endfor
