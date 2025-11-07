__device__ ${return_type} ${kernel_name}(${kernel_args}){
//Device kernel for ${kernel_name}
${local_variables_macros}

${translated_kernel}

% if device_func == True:
return ${return_value};
% endif
}
