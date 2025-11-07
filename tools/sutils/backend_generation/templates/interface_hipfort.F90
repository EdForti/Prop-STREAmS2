interface
subroutine ${kernel_name}_wrapper(stream,${all_variables})&
bind(c,name="${kernel_name}_wrapper")

import :: c_ptr, c_rkind, c_bool, c_int
implicit none

type(c_ptr), value :: stream
% if scalar_int != "":
integer(c_int), value :: ${scalar_int}
% endif
% if scalar_real != "":
real(c_rkind), value :: ${scalar_real}
% endif
% if scalar_bool != "":
logical(c_bool), value :: ${scalar_bool}
% endif
% if array_int != "":
type(c_ptr), value :: ${array_int}
% endif
% if array_real != "":
type(c_ptr), value :: ${array_real}
% endif
% if array_bool != "":
type(c_ptr), value :: ${array_bool}
% endif
% if scalar_redn != "":
real(c_rkind) :: ${scalar_redn}
% endif
% if array_redn != "":
type(c_ptr), value :: ${array_redn}
% endif

endsubroutine ${kernel_name}_wrapper
endinterface
