% if kernel_type=="global":
% for idx in range(num_loop):
do ${index_list[idx]} = ${size[idx][0]},${size[idx][1]}
% endfor
% endif
${serial_part.strip()}
% if kernel_type=="global":
${'enddo\n'*num_loop}
% endif