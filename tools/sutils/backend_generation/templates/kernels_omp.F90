% if kernel_type=="global":
% if local_arrays == True:
!$omp target data map(alloc:${",".join(larrays)})
% endif
<%
final_string=f"!$omp target teams distribute parallel do collapse({num_loop}) has_device_addr({','.join(gpu_arrays)}) {'private('+','.join(larrays)+')'  if local_arrays == True else ''} {'&' if is_reduction == True else ''}"
%>\
${final_string}
% if is_reduction == True:
% for redn_id,redn in enumerate(all_reductions):
% if len(all_reductions) > 1 and redn_id != len(all_reductions)-1:
!$omp& reduction(${redn[0]}:${redn[1]}) &
% else:
!$omp& reduction(${redn[0]}:${redn[1]}) 
% endif
% endfor
% endif
% elif kernel_type == "device":
!$omp declare target
% endif
% if kernel_type=="global":
% for idx in range(num_loop):
do ${index_list[idx]} = ${size[idx][0]},${size[idx][1]}
% endfor
% endif
${serial_part.strip()}
% if kernel_type=="global":
${'enddo\n'*num_loop}
% endif
% if kernel_type == "global" and local_arrays == True:
!$omp end target data
% endif