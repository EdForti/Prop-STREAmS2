% if device_exists:
${device_kernels}
% endif

% if reduction == True:

% for type,var,serial in reduction_kernels:

__global__ void ${launch_bounds} ${kernel_name}_${var}(${kernel_args}){
//Kernel for ${kernel_name}_${var}
${local_variables_macros}
% if debug_mode == False:
${thread_declaration}

if(${loop_conditions}){
${serial}
}
% endif
}

% endfor

% else:

__global__ void ${launch_bounds} ${kernel_name}(${kernel_args}){
//Kernel for ${kernel_name}
${local_variables_macros}
% if debug_mode == False:
${thread_declaration}

if(${loop_conditions}){
${translated_kernel}
}
% endif
}

% endif

extern "C"{
void ${wrapper_name}(hipStream_t stream,${kernel_args}){
dim3 block(${block_definition});
dim3 grid(${grid_definition});

% if reduction == True:

% for type,var,serial in reduction_kernels:
hipLaunchKernelGGL((${kernel_name}_${var}),grid,block,0,stream,${wrapper_args});
% if type == "+":
reduce<real, reduce_op_add>(redn_3d_gpu, nz*ny*nx, ${var});
% elif type == "max":
reduce<real, reduce_op_max>(redn_3d_gpu, nz*ny*nx, ${var});
% elif type == "min":
reduce<real, reduce_op_min>(redn_3d_gpu, nz*ny*nx, ${var});
% endif

% endfor

% else:
hipLaunchKernelGGL((${kernel_name}),grid,block,0,stream,${wrapper_args});
% endif
}
}
