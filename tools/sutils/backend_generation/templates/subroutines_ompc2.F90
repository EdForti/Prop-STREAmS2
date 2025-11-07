% if kernel_type=="global":
% if local_arrays == True:

% endif
<%
final_string=f"!$omp parallel do collapse({num_loop}) {'private('+','.join(larrays)+')'  if local_arrays == True else ''} {'&' if is_reduction == True else ''}"
%>\
${final_string}
% if is_reduction == True:
% for redn_type,redn_scalars in all_reductions:
!$omp& reduction(${redn_type}:${redn_scalars})
% endfor
% endif
% elif kernel_type == "device":

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

% endif