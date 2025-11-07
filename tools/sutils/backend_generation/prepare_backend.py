import logging
import os
from mako.lookup import TemplateLookup
import re
import pprint
from copy import deepcopy
import ast

from backend_generation.backend_tools import get_dict,get_tqdm_iterator,get_non_repeating_elements,split_by_comma
from tools import reverse_list,split_list
from backend_generation.regular_expressions.regex import CudaFRegularExpressions
from backend_generation.translator import FortranToCpp

BACKEND_PATH = os.path.realpath(os.path.dirname(__file__))

class PrepareBackend:
  def __init__(self,backend_dict,gpu_obj):
    self._backend_dict = backend_dict
    self._gpu_obj = gpu_obj
    self._cuf_dict = self._backend_dict["cuf_kernels"]
    self._global_dict = self._backend_dict["global_kernels"]
    self._device_dict = self._backend_dict["device_kernels"]
    self._prepare_dict = {}
    self._template_path = TemplateLookup(directories=BACKEND_PATH)
    self._cfregex = CudaFRegularExpressions()
    
  def generate_kernels(self):
    iterator = get_tqdm_iterator(self._backend_dict)
    for kernel_type,kernel_name,value in iterator:
      iterator.set_postfix(batch=kernel_name)
      if kernel_type == "cuf_kernels":
        self._prepare_cuf(kernel_name,value)
      elif kernel_type == "global_kernels":
        self._prepare_global(kernel_name,value)
      elif kernel_type == "device_kernels":
        self._prepare_device(kernel_name,value)
        
  @property
  def prepare_dict(self):
    return self._prepare_dict
      
  def _prepare_cuf(self):
    raise NotImplementedError
  
  def _prepare_global(self):
    raise NotImplementedError
  
  def _prepare_device(self):
    raise NotImplementedError
    
class PrepareAmd(PrepareBackend):
  def __init__(self, backend_dict,gpu_obj):
    super().__init__(backend_dict,gpu_obj)
    self._base_interface = []
    self._equation_interface = []
    self._kernel_interface = []
    
  @property
  def base_interface(self):
    return self._base_interface
  
  @property
  def equation_interface(self):
    return self._equation_interface
  
  @property
  def kernel_interface(self):
    return self._kernel_interface
  
  def _get_wrapper_args(self,integer,logical,real,int_arrays,logical_arrays,real_arrays,redn_scalar=None,redn_array=None,pointer_style=[]):
    def comprehend(data_list,pointer_style,original_symbol,type_string):
      final_list = []
      for i in data_list:
        if i in pointer_style:
          final_string = f"{type_string} &{i}"
        else:
          final_string = f"{type_string} {original_symbol}{i}"
        final_list.append(final_string)
        
      return final_list
    
    join_vars = integer+logical+real+int_arrays+logical_arrays+real_arrays
    join_c_vars = comprehend(integer,pointer_style,"","int")+comprehend(logical,pointer_style,"","bool")+comprehend(real,pointer_style,"","real")+\
                  comprehend(int_arrays,pointer_style,"*","int")+comprehend(logical_arrays,pointer_style,"*","bool")+comprehend(real_arrays,pointer_style,"*","real")
 
    if redn_scalar:
      join_vars += redn_scalar+redn_array
      join_c_vars += [f"real *{i}" for i in redn_scalar+redn_array]
      
    return join_vars,join_c_vars
    
  def _create_interface(self,template,kernel_name,all_variables,scalar_int,scalar_real,scalar_bool,
                        array_int,array_real,array_bool,
                        scalar_redn="",array_redn=""):
    logging.info(f"creating interface for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template(template)
    main_file = TEMPLATE_FILE.render(kernel_name=kernel_name,all_variables=all_variables,
                                     scalar_int=scalar_int,scalar_real=scalar_real,scalar_bool=scalar_bool,
                                     array_int=array_int,array_real=array_real,array_bool=array_bool,
                                     scalar_redn=scalar_redn,array_redn=array_redn).replace('\r\n','\n')
    return main_file
  
  def wrapper_f_call(self,stream,kernel_name,all_args,all_arrays):
    logging.info(f"Creating Fortran wrapper call for {kernel_name}")
    if stream:
      write_stream = stream
      f_wrapper = f"call {kernel_name}_wrapper({write_stream},{all_args})\n"
    else:
      write_stream = "c_null_ptr"
      f_wrapper = f"call {kernel_name}_wrapper({write_stream},{all_args})\ncall hipCheck(hipDeviceSynchronize())\n"
    
    f_wrapper = re.sub(r"("+all_arrays+r")",r"c_loc(\1)",f_wrapper)
    return f_wrapper
  
  def _get_kernel_logistics(self,idx,size,block_dim):
    # Get block
    block = ",".join(block_dim)
    # Get grid
    if len(idx) == 1:
      grid = f"divideAndRoundUp(({size[0][1]})-({size[0][0]})+1,block.x)"
    elif len(idx) == 2:
      grid = f"divideAndRoundUp(({size[0][1]})-({size[0][0]})+1,block.x),divideAndRoundUp(({size[1][1]})-({size[1][0]})+1,block.y)"
    elif len(idx) == 3:
      grid = f"divideAndRoundUp(({size[0][1]})-({size[0][0]})+1,block.x),divideAndRoundUp(({size[1][1]})-({size[1][0]})+1,block.y),divideAndRoundUp(({size[2][1]})-({size[2][0]})+1,block.z)"
    else:
      logging.error("Length of loop cannot be greater than 3!")
      raise Exception("Check logs!")
    
    # Get thread id and loop_cond
    thread_id = ""
    thread_dim = ["x","y","z"]
    loop_cond_list = []
    for i in range(len(idx)):
      id = idx[i]
      id_range_start = size[i][0]
      id_range_end = size[i][1]
      thread_id += f"{id} = __GIDX({thread_dim[i]},{id_range_start});\n"
      loop_cond_list.append(f"loop_cond({id},{id_range_end},1)")
      
    loop_cond = "&&".join(loop_cond_list)
    
    return block,grid,thread_id,loop_cond
    
  def _get_call_args(self,kernel_name,string):
    if not string.strip().startswith("call "):
      string += f"={string}"
    call_args = self._cfregex.SUBROUTINE_CALL_RE(kernel_name).findall(string)
    call_arguments = [i[3] for i in call_args]

    for i,ca in enumerate(call_arguments):
      ca = ca.replace("&","").replace("\n","").replace(" ","").lower()
      call_arguments[i] = split_by_comma(ca)

    return call_arguments[0]
  
  def _find_strings_in_lists(self, A, integer, linteger, logical, llogical, real, lreal, int_arrays, linteger_arrays, logical_arrays, llogical_arrays, real_arrays, lreal_arrays):
    lists_dict = {
        "real": real,
        "integer": integer,
        "logical": logical,
        "lreal": lreal,
        "linteger": linteger,
        "llogical": llogical,
        "lreal_arrays": lreal_arrays,
        "linteger_arrays": linteger_arrays,
        "llogical_arrays": llogical_arrays,
        "real_arrays": real_arrays,
        "int_arrays": int_arrays,
        "logical_arrays": logical_arrays,
    }

    final_dict = {}
    final_dict["real"]=[]
    final_dict["integer"]=[]
    final_dict["logical"]=[]
    final_dict["lreal"]=[]
    final_dict["linteger"]=[]
    final_dict["llogical"]=[]
    final_dict["lreal_arrays"]=[]
    final_dict["linteger_arrays"]=[]
    final_dict["llogical_arrays"]=[]
    final_dict["real_arrays"]=[]
    final_dict["int_arrays"]=[]
    final_dict["logical_arrays"]=[]

    final_idx_dict = {}
    final_idx_dict["real"]=[]
    final_idx_dict["integer"]=[]
    final_idx_dict["logical"]=[]
    final_idx_dict["lreal"]=[]
    final_idx_dict["linteger"]=[]
    final_idx_dict["llogical"]=[]
    final_idx_dict["lreal_arrays"]=[]
    final_idx_dict["linteger_arrays"]=[]
    final_idx_dict["llogical_arrays"]=[]
    final_idx_dict["real_arrays"]=[]
    final_idx_dict["int_arrays"]=[]
    final_idx_dict["logical_arrays"]=[]

    reg_match = self._cfregex.CAPTURE_ONLY_WORD_RE
    int_counter = 0
    real_counter = 0
    for idx, string in enumerate(A):
      idx_counter = False
      for list_name, lst in lists_dict.items():
        if string in lst and "(" not in string:
          final_dict[list_name].append(string)
          final_idx_dict[list_name].append(idx)
          idx_counter = True
        elif "(" in string:
          var_name = re.search(r"(\w+.*)\(",string).group(1)
          if var_name in lst:
            if list_name == "real_arrays" or list_name =="lreal_arrays":
              final_dict["real"].append(string)
              final_idx_dict["real"].append(idx)
            elif list_name == "int_arrays" or list_name =="linteger_arrays":
              final_dict["integer"].append(string)
              final_idx_dict["integer"].append(idx)
            elif list_name == "logical_arrays" or list_name =="llogical_arrays":
              final_dict["logical"].append(string)
              final_idx_dict["logical"].append(idx)
          idx_counter = True
        elif reg_match.search(string):
          all_matches = reg_match.findall(string)[0]
          if all_matches in lst:
            if list_name == "real_arrays" or list_name == "lreal_arrays" or list_name == "real" or list_name == "lreal":
              final_dict["real"].append(string)
              final_idx_dict["real"].append(idx)
            elif list_name == "int_arrays" or list_name == "linteger_arrays" or list_name == "integer" or list_name == "linteger":
              final_dict["integer"].append(string)
              final_idx_dict["integer"].append(idx)
            elif list_name == "logical_arrays" or list_name == "llogical_arrays" or list_name == "logical" or list_name == "llogical":
              final_dict["logical"].append(string)
              final_idx_dict["logical"].append(idx)
          idx_counter = True
          
      if not idx_counter:
        try:
          literal_val = ast.literal_eval(string)
        except:
          logging.error("Cannot treat argument type")
          raise Exception(f"Cannot treat argument:{string}")
        
        if type(literal_val) == int:
          final_dict["integer"].append(f"sutils_tmp_int_{int_counter}={literal_val}")
          final_idx_dict["integer"].append(idx)
          int_counter += 1
        elif type(literal_val) == float:
          final_dict["real"].append(f"sutils_tmp_int_{real_counter}={literal_val}")
          final_idx_dict["real"].append(idx)
          real_counter += 1
    return final_dict,final_idx_dict
    
  def _get_name_only(self,lst):
    return [i[0] for i in lst]
  
  def _generate_device_kernels(self,updated_kernel_name,device_kernel_calls,serial_code,kernel_var_info):

    if not device_kernel_calls:
      return {},serial_code,""
    
    global_lreal_arrays = kernel_var_info["lreal_arrays"]
    global_linteger_arrays = kernel_var_info["linteger_arrays"]
    global_llogical_arrays = kernel_var_info["llogical_arrays"]
    
    TEMPLATE_FILE = self._template_path.get_template("templates/device_kernels_hipfort.cpp")
    final_dict = {}
    final_device_kernel_str = ""
    for kernel,value in list(device_kernel_calls.items()):
      device_kernel_dict = self._device_dict[kernel]
      subroutine_info,kernel_info,var_info,reduction_info = get_dict(device_kernel_dict)
      subroutine_call = subroutine_info["subroutine_call"]
      sub_call_arguments = subroutine_info["sub_call_arguments"]
      args_map = subroutine_info["args_map"]
      kernel_args = subroutine_info["kernel_args"]
      array_index_map = subroutine_info["array_index_map"]
      device_serial_code = subroutine_info["subroutine_code"]
      final_dict[kernel] = {}
      for new_kernel_name,call in list(value.items()):
        old_call = call
        # Args of kernel call
        final_args = self._get_call_args(kernel,old_call)

        device_kernel_name = f"{kernel}_{updated_kernel_name}_{new_kernel_name.split('_')[-1]}"
        args_type_decompose,args_type_idx = self._find_strings_in_lists(final_args,kernel_var_info["integer"],kernel_var_info["linteger"],\
                kernel_var_info["logical"],(kernel_var_info["llogical"]),(kernel_var_info["real"]),\
                (kernel_var_info["lreal"]),(kernel_var_info["int_arrays"]),\
                self._get_name_only(kernel_var_info["linteger_arrays"]),(kernel_var_info["logical_arrays"]),\
                self._get_name_only(kernel_var_info["llogical_arrays"]),\
                (kernel_var_info["real_arrays"]),self._get_name_only(kernel_var_info["lreal_arrays"]))
        
        new_call = f''' {device_kernel_name}({','.join(
          args_type_decompose["integer"]+args_type_decompose["linteger"]+
          args_type_decompose["logical"]+args_type_decompose["llogical"]+
          args_type_decompose["real"]+args_type_decompose["lreal"]+
          args_type_decompose["int_arrays"]+args_type_decompose["linteger_arrays"]+
          args_type_decompose["logical_arrays"]+args_type_decompose["llogical_arrays"]+
          args_type_decompose["real_arrays"]+args_type_decompose["lreal_arrays"])})\n'''
 
        serial_code = serial_code.replace(old_call.strip(),new_call.strip())

        index_location = subroutine_call.index(old_call)
        mapping = array_index_map[index_location]
        #########################
        # Determine if there are local arrays from global kernels passed to device
        translation_var_info = deepcopy(var_info)
        
        for jk,ks in enumerate(var_info["lreal_arrays"]):
          local_array = ks[0]
          if local_array in kernel_args:
            local_array_index = kernel_args.index(local_array)
            global_kernel_map = final_args[local_array_index]
            real_dimension = "".join([i[1] for i in global_lreal_arrays if i[0] == global_kernel_map])
            translation_var_info["lreal_arrays"][jk][1] = ""
            translation_var_info["lreal_arrays"][jk][0] = (local_array,real_dimension)
            
        for jk,ks in enumerate(var_info["linteger_arrays"]):
          local_array = ks[0]
          if kernel in kernel_args:
            local_array_index = kernel_args.index(local_array)
            global_kernel_map = final_args[local_array_index]
            real_dimension = "".join([i[1] for i in global_linteger_arrays if i[0] == global_kernel_map])
            translation_var_info["linteger_arrays"][jk][1] = ""
            translation_var_info["linteger_arrays"][jk][1] = (local_array,real_dimension)
        for jk,ks in enumerate(var_info["llogical_arrays"]):
          local_array = ks[0]
          if local_array in kernel_args:
            local_array_index = kernel_args.index(local_array)
            global_kernel_map = final_args[local_array_index]
            real_dimension = "".join([i[1] for i in global_llogical_arrays if i[0] == global_kernel_map])
            translation_var_info["llogical_arrays"][jk][1] = ""
            translation_var_info["llogical_arrays"][jk][1] = (local_array,real_dimension)
        #########################
        
        translator = FortranToCpp(device_serial_code,var_info,self._gpu_obj.gpu_dict,mapping)
        translated_code = translator.translated_code
        final_dict[kernel][device_kernel_name] = {}
        final_dict[kernel][device_kernel_name]["translated_code"] = translated_code

        final_dict[kernel][device_kernel_name]["local_var_macros"] = self._generate_local_variables_macros(translation_var_info)
        _,final_dict[kernel][device_kernel_name]["kernel_args"] = self._get_wrapper_args(\
          [kernel_args[i] for i in args_type_idx["integer"]]+[kernel_args[i] for i in args_type_idx["linteger"]],\
          [kernel_args[i] for i in args_type_idx["logical"]]+[kernel_args[i] for i in args_type_idx["llogical"]],\
          [kernel_args[i] for i in args_type_idx["real"]]+[kernel_args[i] for i in args_type_idx["lreal"]],\
          [kernel_args[i] for i in args_type_idx["int_arrays"]]+[kernel_args[i] for i in args_type_idx["linteger_arrays"]],\
          [kernel_args[i] for i in args_type_idx["logical_arrays"]]+[kernel_args[i] for i in args_type_idx["llogical_arrays"]],\
          [kernel_args[i] for i in args_type_idx["real_arrays"]]+[kernel_args[i] for i in args_type_idx["lreal_arrays"]],pointer_style=var_info["pointer_variables"])

        final_dict[kernel][device_kernel_name]["final_device_kernels"] = TEMPLATE_FILE.render(kernel_name=device_kernel_name,return_type=kernel_info["function_type"],\
          kernel_args=",".join(final_dict[kernel][device_kernel_name]["kernel_args"]),local_variables_macros=final_dict[kernel][device_kernel_name]["local_var_macros"],\
          translated_kernel=final_dict[kernel][device_kernel_name]["translated_code"],return_value=kernel_info["return_value"],device_func=kernel_info["is_device_func"]).replace('\r\n','\n')
        final_device_kernel_str += final_dict[kernel][device_kernel_name]["final_device_kernels"]+"\n"
                                                            
        
    return final_dict,serial_code,final_device_kernel_str
  
  def _get_local_vars_as_string(self,var_list, var_type):
    if var_list:
      var_list = split_list(var_list,3)
      final_string = ""
      for vl in var_list:
        vl = [f"{var_type} {i}" for i in vl]
        join_var = ";".join(vl)+";\n"
        final_string += join_var
      return final_string  
    return ""

  def _get_local_arrays_as_string(self,array_list, array_type):
    def check_for_range(string):
      if ":" in string:
        range_val = string.split(":")
        return f"(({range_val[1]})-({range_val[0]}))+1",range_val[0]
      else:
        return string,"1"
    if array_list:
      final_string = ""
      for al in array_list:
        if al[1]:
          array_name = al[0]
          array_size = al[1].split(",")
          if len(array_size) == 1:
            total_size,first_val = check_for_range(array_size[0])
            array_def = f"{array_type} {array_name}[{total_size.strip()}];\n"
            array_id = f"#undef __LI_{array_name.upper()}\n#define __LI_{array_name.upper()}(i) (i-({first_val}))\n"
          elif len(array_size) == 2:
            total_size_1,first_val_1 = check_for_range(array_size[0])
            total_size_2,first_val_2 = check_for_range(array_size[1])
            total_size = f"{total_size_1}*{total_size_2}"
            array_def = f"{array_type} {array_name}[{total_size.strip()}];\n"
            array_id = f"#undef __LI_{array_name.upper()}\n#define __LI_{array_name.upper()}(i,j) ((i-({first_val_1}))+{total_size_1}*(j-({first_val_2})))\n"
          else:
            logging.error("Not implemented for size higher than 2D for local array")
            raise Exception("Check log files!")
        else:
          # For cases in device kernels where local kernels from global kernels are passed
          array_name = al[0][0]
          array_size = al[0][1].split(",")
          if len(array_size) == 1:
            total_size,first_val = check_for_range(array_size[0])
            array_def = f""
            array_id = f"#undef __LI_{array_name.upper()}\n#define __LI_{array_name.upper()}(i) (i-({first_val}))\n"
          elif len(array_size) == 2:
            total_size_1,first_val_1 = check_for_range(array_size[0])
            total_size_2,first_val_2 = check_for_range(array_size[1])
            total_size = f"{total_size_1}*{total_size_2}"
            array_def = f""
            array_id = f"#undef __LI_{array_name.upper()}\n#define __LI_{array_name.upper()}(i,j) ((i-({first_val_1}))+{total_size_1}*(j-({first_val_2})))\n"
          else:
            logging.error("Not implemented for size higher than 2D for local array")
            raise Exception("Check log files!")
        final_string += array_def+array_id
      return final_string  
    return ""
  
  def _generate_local_variables_macros(self,var_info):
    variable_string = ""
    
    lreal = var_info["lreal"]
    variable_string += self._get_local_vars_as_string(lreal,"real")
    
    lint = var_info["linteger"]
    variable_string += self._get_local_vars_as_string(lint,"int")
    
    llogical = var_info["llogical"]
    variable_string += self._get_local_vars_as_string(llogical,"bool")
     
    lreal_arrays = var_info["lreal_arrays"]
    variable_string += self._get_local_arrays_as_string(lreal_arrays,"real")
      
    lint_arrays = var_info["linteger_arrays"]
    variable_string += self._get_local_arrays_as_string(lint_arrays,"int")
      
    llogical_arrays = var_info["llogical_arrays"]
    variable_string += self._get_local_arrays_as_string(llogical_arrays,"bool")
    
    return variable_string
  
  def update_reduction_serial_part(self,serial_part,reduction_var,reduction_type,reduction_vars):
    if reduction_type == "+":
      serial_part = self._cfregex.REDN_RE(reduction_var).sub(r"redn_3d_gpu(i,j,k) = \2\3\n",serial_part)
      other_redn_vars = "|".join([i for i in reduction_vars if i!=reduction_var])
      if other_redn_vars:
        serial_part = self._cfregex.REDN_RE(other_redn_vars).sub(r"\n",serial_part)
    elif reduction_type == "max":
      for rv in reduction_vars:
        if rv == reduction_var:
          serial_part = self._cfregex.REDN_MAX_RE(rv).sub(r"redn_3d_gpu(i,j,k) = \2\n",serial_part)
        else:
          serial_part = self._cfregex.REDN_MAX_RE(rv).sub("\n",serial_part)
    elif reduction_type == "min":
      for rv in reduction_vars:
        if rv == reduction_var:
          serial_part = self._cfregex.REDN_MIN_RE(rv).sub(r"redn_3d_gpu(i,j,k) = \2\n",serial_part)
        else:
          serial_part = self._cfregex.REDN_MIN_RE(rv).sub("\n",serial_part)
    serial_part = self._cfregex.REDN_REMOVE_OTHER_RE.sub("",serial_part)
    return serial_part
  
  def _get_scalar_string(self,real_list,integer_list,logical_list,reduction_list=None):
    final_string = ""
    if integer_list:
      for ilst in integer_list:
        final_string += f"integer ::  {','.join(ilst)}\n"
    if real_list:
      for rlst in real_list:
        final_string += f"real(rkind) ::  {','.join(rlst)}\n"
    if logical_list:
      for llst in logical_list:
        final_string += f"logical(1) ::  {','.join(llst)}\n"
    if reduction_list:
      for rdst in reduction_list:
        final_string += f"real(rkind) ::  {','.join(rdst)}\n"
    return final_string

  def _get_arrays(self,array_list,array_type,gpuvar_dict,gpu_array_map):
    final_string = ""
    if array_list:
      for lst in array_list:
        # For some arrays, the definitions are not present in the gpu array dictionary.
        # Therefore, we map to its original array and extract its dimension definition
        final_string += f"{array_type}, {gpuvar_dict[gpu_array_map[lst]][1]}, target :: {lst}\n"
    
    return final_string
        
  def _prepare_cuf(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/global_kernels_hipfort.cpp")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    self._prepare_dict[kernel_name]["call_handling"] = {}
    self._prepare_dict[kernel_name]["subroutine_handling"] = {}
    self._prepare_dict[kernel_name]["full_kernel"] = ""
    integer = var_info["integer"]
    real = var_info["real"]
    logical = var_info["logical"]
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    device_kernel_calls = kernel_info["device_kernel_calls"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_name = subroutine_info["subroutine_name"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    kernel_args = subroutine_info["kernel_args"]
    non_kernel_var = kernel_info["non_kernel_def"]
    stream_info = subroutine_info["stream_in_call"]
    non_cuf = kernel_info["non_cuf"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    # We are only considering the first array index map from any of the calls, since it should represent the main kernel
    device_array_mapping = subroutine_info["array_index_map"][0]
    
    # Create cuf subroutine  
    if True in kernel_info["is_reduction"]:
      logging.warning("Explicit addition of redn_3d_gpu in this function")
      device_array_mapping["redn_3d_gpu"] = "redn_3d_gpu"
      reduction_array = ["redn_3d_gpu"]
      kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args+['redn_3d_gpu'])})\n"
    else:
      kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n"
      
    scalar_string = self._get_scalar_string(split_list(real,3),split_list(integer,3),split_list(logical,3))
    array_string = self._get_arrays(real_arrays,"real(rkind)",self._gpu_obj.gpu_dict,device_array_mapping)
    array_string += self._get_arrays(int_arrays,"integer",self._gpu_obj.gpu_dict,device_array_mapping)
    array_string += self._get_arrays(logical_arrays,"logical",self._gpu_obj.gpu_dict,device_array_mapping)
    
    if True in kernel_info["is_reduction"]:
      reduction_scalars = reduction_info["reduction_scalars"]
      reduction_array = ["redn_3d_gpu"]
      scalar_string += self._get_scalar_string([],[],[],split_list(sum(reduction_scalars,[]),3))
      array_string += self._get_arrays(reduction_array,"real(rkind)",self._gpu_obj.gpu_dict,device_array_mapping)

    kernel_string += scalar_string
    kernel_string += array_string
    kernel_string += non_kernel_var+"\n"
    
    if get_non_repeating_elements(stream_info):
      kernel_string += f"type(c_ptr), value :: {get_non_repeating_elements(stream_info)}\n"

    for id,names in enumerate(kernel_info["kernel_names"]):
      self._prepare_dict[kernel_name][names]={}
      final_dict = self._prepare_dict[kernel_name][names]
      self._prepare_dict[kernel_name]["source"] = subroutine_source
      is_reduction = kernel_info["is_reduction"][id]
      reduction_scalars = []
      reduction_array = []
      if is_reduction:
        reduction_scalars = reduction_info["reduction_scalars"][id]
        reduction_array = ["redn_3d_gpu"]
        serial[id] = f"redn_3d_gpu({','.join(reverse_list(idx[id]))})=0.0\n"+serial[id]
      
      # Adjust subroutine call arguments only for reduction
      for ijk,kcalls in enumerate(subroutine_calls):
        old_call = kcalls
        source = call_source[ijk]
        if is_reduction:
          join_kernel_args = ",".join(subroutine_call_arguments[ijk])+",self%redn_3d_gpu"
        else:
          join_kernel_args = ",".join(subroutine_call_arguments[ijk])
        new_kernel_call = f"call {subroutine_name}({join_kernel_args})\n"
        if not self._prepare_dict[kernel_name]["call_handling"].get(source):
          self._prepare_dict[kernel_name]["call_handling"][source] = []
          self._prepare_dict[kernel_name]["call_handling"][source].append((old_call,new_kernel_call))
        else:
          self._prepare_dict[kernel_name]["call_handling"][source].append((old_call,new_kernel_call))
          
    
      final_dict["wrapper_f_args"],final_dict["wrapper_c_args"] = self._get_wrapper_args(integer,logical,real,int_arrays,
                                  logical_arrays,real_arrays,redn_scalar=reduction_scalars,redn_array=reduction_array)
      
      final_dict["interface"] = self._create_interface("templates/interface_hipfort.F90",names,
                    ",".join(final_dict["wrapper_f_args"]),",".join(integer),",".join(real),",".join(logical),
                    ",".join(int_arrays),",".join(real_arrays),",".join(logical_arrays),
                    scalar_redn=",".join(reduction_scalars),array_redn=",".join(reduction_array))
      
      # Add interfaces to list
      if subroutine_source == "base_file":
        self._base_interface.append(final_dict["interface"])
      elif subroutine_source == "equation_file":
        self._equation_interface.append(final_dict["interface"])
      elif subroutine_source == "kernel_file":
        self._kernel_interface.append(final_dict["interface"])
      
      final_dict["wrapper_f_call"] = self.wrapper_f_call(stream[id],names,",".join(final_dict["wrapper_f_args"]),"|".join(int_arrays+real_arrays+logical_arrays+reduction_array))
    
      if launch_bounds:
        final_dict["launch_bounds"] = launch_bounds[id]
      else:
        final_dict["launch_bounds"] = ""

      final_dict["block"],final_dict["grid"],final_dict["thread_decl"],final_dict["loop_cond"] = self._get_kernel_logistics(reverse_list(idx[id]),reverse_list(size[id]),block_dim[id])
      
      # If device kernel exists
      logging.info(f"Adding device kernels for:{kernel_name}")
      final_dict["device_kernels"],final_dict["serial"],final_dict["final_device_kernel"] = self._generate_device_kernels(names,device_kernel_calls[id],serial[id],var_info)
      
      final_dict["local_var_macros"] = self._generate_local_variables_macros(var_info)
      
      if not is_reduction:
        # Non reduction case
        translator = FortranToCpp(final_dict["serial"],var_info,self._gpu_obj.gpu_dict,device_array_mapping)
        final_dict["serial"] = translator.translated_code
        final_dict["final_kernel"] = TEMPLATE_FILE.render(reduction=is_reduction,launch_bounds=final_dict["launch_bounds"],kernel_name=names,wrapper_name=f"{names}_wrapper",kernel_args=",".join(final_dict["wrapper_c_args"]),\
                                            reduction_kernels=[],local_variables_macros=final_dict["local_var_macros"],debug_mode=debug_mode,\
                                            thread_declaration=final_dict["thread_decl"],loop_conditions=final_dict["loop_cond"],translated_kernel=final_dict["serial"],block_definition=final_dict["block"],\
                                            grid_definition=final_dict["grid"],wrapper_args=",".join(final_dict["wrapper_f_args"]),device_exists=device_kernel_calls[id],device_kernels=final_dict["final_device_kernel"])
      else:
        # Reduction case
        redn_info = reduction_info["reduction_data"]
        translated_reduction = []
        rdata = redn_info[id]
        if rdata:
          for rtype,rvars in rdata.items():
            for v in rvars:
              updated_serial = self.update_reduction_serial_part(final_dict["serial"],v,rtype,rvars)
              
              translator = FortranToCpp(updated_serial,var_info,self._gpu_obj.gpu_dict,device_array_mapping,is_reduction=True)
              translated_reduction.append((rtype,v,translator.translated_code))
              
          final_dict["final_kernel"] = TEMPLATE_FILE.render(reduction=is_reduction,launch_bounds=final_dict["launch_bounds"],kernel_name=names,wrapper_name=f"{names}_wrapper",kernel_args=",".join(final_dict["wrapper_c_args"]),\
                                            reduction_kernels=translated_reduction,local_variables_macros=final_dict["local_var_macros"],debug_mode=debug_mode,\
                                            thread_declaration=final_dict["thread_decl"],loop_conditions=final_dict["loop_cond"],translated_kernel="",block_definition=final_dict["block"],\
                                            grid_definition=final_dict["grid"],wrapper_args=",".join(final_dict["wrapper_f_args"]),device_exists=device_kernel_calls[id],device_kernels=final_dict["final_device_kernel"])
      
      self._prepare_dict[kernel_name]["full_kernel"]+=final_dict["final_kernel"]
      
      # Continue creating cuf subroutine  
      wrapper_call = final_dict["wrapper_f_call"]
      top_part = non_cuf[id]
      bottom_part = non_cuf[id+1]
      if id > 0:
        top_part = ""
      
      kernel_string += f"{top_part.strip()}\n"
      kernel_string += f"{wrapper_call.strip()}\n"
      kernel_string += f"{bottom_part.strip()}"
    
    kernel_string += f"\n\nendsubroutine {subroutine_name}\n\n"
    self._prepare_dict[kernel_name]["subroutine_handling"][subroutine_source] = [full_subroutine,kernel_string]
      
  def _prepare_global(self,kernel_name,kernel_dict):
    logging.info(f"Preparing global kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/global_kernels_hipfort.cpp")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict=self._prepare_dict[kernel_name]
    final_dict["call_handling"] = {}
    integer = var_info["integer"]
    real = var_info["real"]
    logical = var_info["logical"]
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = reverse_list(kernel_info["idx"])
    size = reverse_list(kernel_info["size"])
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    device_kernel_calls = kernel_info["device_kernel_calls"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    device_array_mapping = subroutine_info["array_index_map"][0]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    args_map = subroutine_info["args_map"]
    subroutine_source = subroutine_info["source"]
    self._prepare_dict[kernel_name]["source"] = subroutine_source
    final_dict["wrapper_f_args"],final_dict["wrapper_c_args"] = self._get_wrapper_args(integer,logical,real,int_arrays,
                                logical_arrays,real_arrays)
    
    final_dict["interface"] = self._create_interface("templates/interface_hipfort.F90",kernel_name,
                                                     ",".join(final_dict["wrapper_f_args"]),",".join(integer),",".join(real),",".join(logical),
                                                     ",".join(int_arrays),",".join(real_arrays),",".join(logical_arrays))

    # Add interfaces to list
    if subroutine_source == "base_file":
      self._base_interface.append(final_dict["interface"])
    elif subroutine_source == "equation_file":
      self._equation_interface.append(final_dict["interface"])
    elif subroutine_source == "kernel_file":
      self._kernel_interface.append(final_dict["interface"])
    
    if launch_bounds:
      final_dict["launch_bounds"] = launch_bounds
    else:
      final_dict["launch_bounds"] = ""
    final_dict["block"],final_dict["grid"],final_dict["thread_decl"],final_dict["loop_cond"] = self._get_kernel_logistics(idx,size,block_dim)
    
    # If device kernel exists
    logging.info(f"Adding device kernels for:{kernel_name}")
    final_dict["device_kernels"],final_dict["serial"],final_dict["final_device_kernel"] = self._generate_device_kernels(kernel_name,device_kernel_calls,serial,var_info)
    
    # Translated serial code
    translator = FortranToCpp(final_dict["serial"],var_info,self._gpu_obj.gpu_dict,device_array_mapping)
    final_dict["serial"] = translator.translated_code
    final_dict["local_var_macros"] = self._generate_local_variables_macros(var_info)
    final_dict["full_kernel"] = TEMPLATE_FILE.render(reduction=False,launch_bounds=final_dict["launch_bounds"],kernel_name=kernel_name,wrapper_name=f"{kernel_name}_wrapper",kernel_args=",".join(final_dict["wrapper_c_args"]),\
                                            reduction_kernels=[],local_variables_macros=final_dict["local_var_macros"],debug_mode=debug_mode,\
                                            thread_declaration=final_dict["thread_decl"],loop_conditions=final_dict["loop_cond"],translated_kernel=final_dict["serial"],block_definition=final_dict["block"],\
                                            grid_definition=final_dict["grid"],wrapper_args=",".join(final_dict["wrapper_f_args"]),device_exists=device_kernel_calls,device_kernels=final_dict["final_device_kernel"])
    
    # Kernel calls creation
    for ijk,kcalls in enumerate(subroutine_calls):
      old_call = kcalls
      kernel_args = final_dict["wrapper_f_args"]
      source = call_source[ijk]
      inverted_args_map = dict(reversed(list(args_map[ijk].items())))
      new_call = ",".join([inverted_args_map[i] for i in kernel_args])
      new_kernel_call = self.wrapper_f_call(stream[ijk],kernel_name,new_call,"|".join(int_arrays+real_arrays+logical_arrays))
      if not final_dict["call_handling"].get(source):
        final_dict["call_handling"][source] = []
        final_dict["call_handling"][source].append((old_call,new_kernel_call))
      else:
        final_dict["call_handling"][source].append((old_call,new_kernel_call))

  def _prepare_device(self,valx,valy):
    pass

class PrepareCpu(PrepareBackend):
  def __init__(self, backend_dict,gpu_obj):
    super().__init__(backend_dict,gpu_obj)
    
  def _prepare_cuf(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/subroutines_cpu.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    final_dict["final_kernel"] = []
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    device_kernel_calls = kernel_info["device_kernel_calls"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_name = subroutine_info["subroutine_name"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    kernel_args = subroutine_info["kernel_args"]
    non_kernel_var = kernel_info["non_kernel_def"]
    kernel_var = subroutine_info["kernel_variables"]
    stream_info = subroutine_info["stream_in_call"]
    non_cuf = kernel_info["non_cuf"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    # We are only considering the first array index map from any of the calls, since it should represent the main kernel
    device_array_mapping = subroutine_info["array_index_map"][0]
    num_of_loops = kernel_info["num_loop"]
    redn_info = reduction_info["reduction_data"]
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    for id,names in enumerate(kernel_info["kernel_names"]):

      final_kernel = TEMPLATE_FILE.render(kernel_type="global",num_loop=num_of_loops[id],index_list=idx[id],size=size[id],serial_part=serial[id])
      final_dict["final_kernel"].append(final_kernel)
      # Continue creating cuf subroutine  
      wrapper_call = final_kernel
      top_part = non_cuf[id]
      bottom_part = non_cuf[id+1]
      if id > 0:
        top_part = ""
      
      kernel_string += f"{top_part.strip()}\n"
      kernel_string += f"{wrapper_call.strip()}\n"
      kernel_string += f"{bottom_part.strip()}"
    
    kernel_string += f"\n\nendsubroutine {subroutine_name}\n\n"
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
    
  def _prepare_global(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/subroutines_cpu.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    integer = var_info["integer"]
    real = var_info["real"]
    logical = var_info["logical"]
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_name = subroutine_info["subroutine_name"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    kernel_args = subroutine_info["kernel_args"]
    kernel_var = subroutine_info["kernel_variables"]
    stream_info = subroutine_info["stream_in_call"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    # We are only considering the first array index map from any of the calls, since it should represent the main kernel
    device_array_mapping = subroutine_info["array_index_map"][0]
  
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    num_of_loops = len(idx)
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    final_dict["final_kernel"] = TEMPLATE_FILE.render(kernel_type="global",num_loop=num_of_loops,index_list=idx,size=size,serial_part=serial)
    
    kernel_string += final_dict["final_kernel"]
    
    kernel_string += f"endsubroutine {kernel_name}\n"
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
    
  def _prepare_device(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/subroutines_cpu.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    serial = subroutine_info["subroutine_code"]
    subroutine_name = subroutine_info["subroutine_name"]
    kernel_args = subroutine_info["kernel_args"]
    kernel_var = subroutine_info["kernel_variables"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    return_val = kernel_info["return_value"]
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n" if not return_val else f"function {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    final_dict["final_kernel"] = TEMPLATE_FILE.render(kernel_type="device",serial_part=serial)
    
    kernel_string += final_dict["final_kernel"]
    
    kernel_string += f"endsubroutine {kernel_name}\n" if not return_val else f"endfunction {kernel_name}\n"
    
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
    
class PrepareOmp(PrepareBackend):
  def __init__(self, backend_dict,gpu_obj):
    super().__init__(backend_dict,gpu_obj)
    
  def _prepare_cuf(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/kernels_omp.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    final_dict["final_kernel"] = []
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    device_kernel_calls = kernel_info["device_kernel_calls"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_name = subroutine_info["subroutine_name"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    kernel_args = subroutine_info["kernel_args"]
    non_kernel_var = kernel_info["non_kernel_def"]
    kernel_var = subroutine_info["kernel_variables"]
    stream_info = subroutine_info["stream_in_call"]
    non_cuf = kernel_info["non_cuf"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    # We are only considering the first array index map from any of the calls, since it should represent the main kernel
    device_array_mapping = subroutine_info["array_index_map"][0]
    num_of_loops = kernel_info["num_loop"]
    redn_info = reduction_info["reduction_data"]
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    for id,names in enumerate(kernel_info["kernel_names"]):
      
      is_reduction = kernel_info["is_reduction"][id]
      rlist = []
      if is_reduction:
        rdata = redn_info[id]
        for rtype,rvars in rdata.items():
          rlist.append([rtype,",".join(rvars)])

      all_local_arrays = []
      local_arrays = False
      if var_info["lreal_arrays"] or var_info["linteger_arrays"] or var_info["llogical_arrays"]: local_arrays = True
      all_local_arrays = [i[0]for i in var_info["lreal_arrays"] + var_info["linteger_arrays"] + var_info["llogical_arrays"]]
      all_gpu_arrays = real_arrays+int_arrays+logical_arrays

      final_kernel = TEMPLATE_FILE.render(all_reductions=rlist,is_reduction=is_reduction,kernel_type="global",local_arrays=local_arrays,larrays=all_local_arrays,\
            num_loop=num_of_loops[id],gpu_arrays=all_gpu_arrays,index_list=idx[id],size=size[id],serial_part=serial[id])
      final_dict["final_kernel"].append(final_kernel)
      # Continue creating cuf subroutine  
      wrapper_call = final_kernel
      top_part = non_cuf[id]
      bottom_part = non_cuf[id+1]
      if id > 0:
        top_part = ""
      
      kernel_string += f"{top_part.strip()}\n"
      kernel_string += f"{wrapper_call.strip()}\n"
      kernel_string += f"{bottom_part.strip()}"
    
    kernel_string += f"\n\nendsubroutine {subroutine_name}\n\n"
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
  
  def _prepare_global(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/kernels_omp.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    integer = var_info["integer"]
    real = var_info["real"]
    logical = var_info["logical"]
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_name = subroutine_info["subroutine_name"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    kernel_args = subroutine_info["kernel_args"]
    kernel_var = subroutine_info["kernel_variables"]
    stream_info = subroutine_info["stream_in_call"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    # We are only considering the first array index map from any of the calls, since it should represent the main kernel
    device_array_mapping = subroutine_info["array_index_map"][0]
  
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    num_of_loops = len(idx)
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    all_local_arrays = []
    local_arrays = False
    if var_info["lreal_arrays"] or var_info["linteger_arrays"] or var_info["llogical_arrays"]: local_arrays = True
    all_local_arrays = [i[0]for i in var_info["lreal_arrays"] + var_info["linteger_arrays"] + var_info["llogical_arrays"]]
    all_gpu_arrays = real_arrays+int_arrays+logical_arrays
    final_dict["final_kernel"] = TEMPLATE_FILE.render(is_reduction=False,kernel_type="global",local_arrays=local_arrays,larrays=all_local_arrays,\
            num_loop=num_of_loops,gpu_arrays=all_gpu_arrays,index_list=idx,size=size,serial_part=serial)
    
    kernel_string += final_dict["final_kernel"]
    
    kernel_string += f"endsubroutine {kernel_name}\n"
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
  
  
  def _prepare_device(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/kernels_omp.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    serial = subroutine_info["subroutine_code"]
    subroutine_name = subroutine_info["subroutine_name"]
    kernel_args = subroutine_info["kernel_args"]
    kernel_var = subroutine_info["kernel_variables"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    return_val = kernel_info["return_value"]
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n" if not return_val else f"function {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    all_local_arrays = []
    local_arrays = False
    if var_info["lreal_arrays"] or var_info["linteger_arrays"] or var_info["llogical_arrays"]: local_arrays = True
    all_local_arrays = [i[0]for i in var_info["lreal_arrays"] + var_info["linteger_arrays"] + var_info["llogical_arrays"]]
    all_gpu_arrays = real_arrays+int_arrays+logical_arrays
    final_dict["final_kernel"] = TEMPLATE_FILE.render(is_reduction=False,kernel_type="device",local_arrays=local_arrays,larrays=all_local_arrays,\
            num_loop=1,gpu_arrays=all_gpu_arrays,index_list=1,size=1,serial_part=serial)
    
    kernel_string += final_dict["final_kernel"]
    
    kernel_string += f"endsubroutine {kernel_name}\n" if not return_val else f"endfunction {kernel_name}\n"
    
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
    
class PrepareOmpc(PrepareBackend):
  def __init__(self, backend_dict,gpu_obj):
    super().__init__(backend_dict,gpu_obj)
    
  def _prepare_cuf(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/subroutines_ompc.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    final_dict["final_kernel"] = []
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    device_kernel_calls = kernel_info["device_kernel_calls"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_name = subroutine_info["subroutine_name"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    kernel_args = subroutine_info["kernel_args"]
    non_kernel_var = kernel_info["non_kernel_def"]
    kernel_var = subroutine_info["kernel_variables"]
    stream_info = subroutine_info["stream_in_call"]
    non_cuf = kernel_info["non_cuf"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    # We are only considering the first array index map from any of the calls, since it should represent the main kernel
    device_array_mapping = subroutine_info["array_index_map"][0]
    num_of_loops = kernel_info["num_loop"]
    redn_info = reduction_info["reduction_data"]
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    for id,names in enumerate(kernel_info["kernel_names"]):
      
      is_reduction = kernel_info["is_reduction"][id]
      rlist = []
      if is_reduction:
        rdata = redn_info[id]
        for rtype,rvars in rdata.items():
          rlist.append([rtype,",".join(rvars)])

      all_local_arrays = []
      local_arrays = False
      if var_info["lreal_arrays"] or var_info["linteger_arrays"] or var_info["llogical_arrays"]: local_arrays = True
      all_local_arrays = [i[0]for i in var_info["lreal_arrays"] + var_info["linteger_arrays"] + var_info["llogical_arrays"]]
      all_gpu_arrays = real_arrays+int_arrays+logical_arrays

      final_kernel = TEMPLATE_FILE.render(all_reductions=rlist,is_reduction=is_reduction,kernel_type="global",local_arrays=local_arrays,larrays=all_local_arrays,\
            num_loop=num_of_loops[id],gpu_arrays=all_gpu_arrays,index_list=idx[id],size=size[id],serial_part=serial[id])
      final_dict["final_kernel"].append(final_kernel)
      # Continue creating cuf subroutine  
      wrapper_call = final_kernel
      top_part = non_cuf[id]
      bottom_part = non_cuf[id+1]
      if id > 0:
        top_part = ""
      
      kernel_string += f"{top_part.strip()}\n"
      kernel_string += f"{wrapper_call.strip()}\n"
      kernel_string += f"{bottom_part.strip()}"
    
    kernel_string += f"\n\nendsubroutine {subroutine_name}\n\n"
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
  
  def _prepare_global(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/subroutines_ompc.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    integer = var_info["integer"]
    real = var_info["real"]
    logical = var_info["logical"]
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    stream = subroutine_info["stream_in_call"]
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    launch_bounds = kernel_info["launch_bounds"]
    block_dim = kernel_info["blockdim"]
    serial = kernel_info["serial"]
    debug_mode=kernel_info["debug_mode"]
    subroutine_calls = subroutine_info["subroutine_call"]
    subroutine_name = subroutine_info["subroutine_name"]
    subroutine_call_arguments = subroutine_info["sub_call_arguments"]
    call_source = subroutine_info["call_sources"]
    kernel_args = subroutine_info["kernel_args"]
    kernel_var = subroutine_info["kernel_variables"]
    stream_info = subroutine_info["stream_in_call"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    # We are only considering the first array index map from any of the calls, since it should represent the main kernel
    device_array_mapping = subroutine_info["array_index_map"][0]
  
    idx = kernel_info["idx"]
    size = kernel_info["size"]
    num_of_loops = len(idx)
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    all_local_arrays = []
    local_arrays = False
    if var_info["lreal_arrays"] or var_info["linteger_arrays"] or var_info["llogical_arrays"]: local_arrays = True
    all_local_arrays = [i[0]for i in var_info["lreal_arrays"] + var_info["linteger_arrays"] + var_info["llogical_arrays"]]
    all_gpu_arrays = real_arrays+int_arrays+logical_arrays
    final_dict["final_kernel"] = TEMPLATE_FILE.render(is_reduction=False,kernel_type="global",local_arrays=local_arrays,larrays=all_local_arrays,\
            num_loop=num_of_loops,gpu_arrays=all_gpu_arrays,index_list=idx,size=size,serial_part=serial)
    
    kernel_string += final_dict["final_kernel"]
    
    kernel_string += f"endsubroutine {kernel_name}\n"
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]
  
  
  def _prepare_device(self,kernel_name,kernel_dict):
    logging.info(f"Preparing cuf kernels conversion for:{kernel_name}")
    TEMPLATE_FILE = self._template_path.get_template("templates/subroutines_ompc.F90")
    subroutine_info,kernel_info,var_info,reduction_info = get_dict(kernel_dict)
    self._prepare_dict[kernel_name]={}
    final_dict = self._prepare_dict[kernel_name]
    final_dict["subroutine_handling"] = {}
    real_arrays = var_info["real_arrays"]
    int_arrays = var_info["int_arrays"]
    logical_arrays = var_info["logical_arrays"]
    serial = subroutine_info["subroutine_code"]
    subroutine_name = subroutine_info["subroutine_name"]
    kernel_args = subroutine_info["kernel_args"]
    kernel_var = subroutine_info["kernel_variables"]
    subroutine_source = subroutine_info["source"]
    full_subroutine = subroutine_info["full_subroutine"]
    return_val = kernel_info["return_value"]
    
    # Create cuf subroutine  
    kernel_string = f"subroutine {subroutine_name}({','.join(kernel_args)})\n" if not return_val else f"function {subroutine_name}({','.join(kernel_args)})\n"
    
    # Add all variables
    kernel_string += kernel_var
    
    final_dict["source"] = subroutine_source

    all_local_arrays = []
    local_arrays = False
    if var_info["lreal_arrays"] or var_info["linteger_arrays"] or var_info["llogical_arrays"]: local_arrays = True
    all_local_arrays = [i[0]for i in var_info["lreal_arrays"] + var_info["linteger_arrays"] + var_info["llogical_arrays"]]
    all_gpu_arrays = real_arrays+int_arrays+logical_arrays
    final_dict["final_kernel"] = TEMPLATE_FILE.render(is_reduction=False,kernel_type="device",local_arrays=local_arrays,larrays=all_local_arrays,\
            num_loop=1,gpu_arrays=all_gpu_arrays,index_list=1,size=1,serial_part=serial)
    
    kernel_string += final_dict["final_kernel"]
    
    kernel_string += f"endsubroutine {kernel_name}\n" if not return_val else f"endfunction {kernel_name}\n"
    
    final_dict["subroutine_handling"][subroutine_source] = [full_subroutine.strip(),kernel_string]