import logging
from abc import ABC, abstractmethod
import pprint
from backend_generation.backend_tools import get_dict,update_list_order
from tools import reverse_list
from backend_generation.regular_expressions.regex import CudaFRegularExpressions

class KernelProbe(ABC):
  def __init__(self,kernel_name,gpu_array_obj,kernel_dict,kernel_config,device_kernels=None):
    self._kernel_name = kernel_name
    self._kernel_dict = kernel_dict
    self._kernel_config = kernel_config
    self._device_kernels = device_kernels
    self._subroutine_info,self._kernel_info,self._var_info,self._reduction_info = get_dict(self._kernel_dict)
    self._cfregex = CudaFRegularExpressions()
    self._gpu_array_obj = gpu_array_obj

  def _use_config_dict(self):
    if self._kernel_config:
      for key in self._kernel_config:
        self._kernel_info[key] = self._kernel_config[key]

  def _subroutine_declaration(self,kernel_name,full_subroutine):
    try:
      match_declaration = self._cfregex.CAPTURE_SUBROUTINE_ARGS_RE.search(full_subroutine)
      declaration = match_declaration.group(1)
      dummy_args = match_declaration.group(3)
    except:
      logging.error(f"Error in capturing subroutine declaration for: {kernel_name}")
      raise Exception("Error in getting subroutine, Check again!")
    dummy_args = dummy_args.replace("&","").replace("\n","").replace(" ","")

    return declaration,dummy_args.split(",")

  def _extract_arrays(self,variables_str):
    # Look for real device arrays
    real_arrays = ",".join(self._cfregex.REAL_EXP_RE.findall(variables_str))
    final_real_arrays = self._cfregex.GPU_ARRAY_NAME_RE.findall(real_arrays)
    # Look for int device arrays
    int_arrays = self._cfregex.INT_EXP_RE.findall(variables_str)
    final_int_arrays = []
    if int_arrays != [] : final_int_arrays = self._cfregex.GPU_ARRAY_NAME_RE.findall(",".join(int_arrays))
    # Look for logical device arrays
    logical_arrays = self._cfregex.LOG_EXP_RE.findall(variables_str)
    final_logical_arrays = []
    if logical_arrays != [] : final_logical_arrays = self._cfregex.GPU_ARRAY_NAME_RE.findall(",".join(logical_arrays))

    return final_real_arrays,final_int_arrays,final_logical_arrays

  def _get_array_index_map(self,gpu_arrays,args_map):
    final_dict = {}
    for arrays in gpu_arrays:
      final_dict[arrays] = arrays
      call_arg_map = args_map[arrays]
      call_arg_map = call_arg_map.split("%")[-1]
      if arrays != call_arg_map:
        if call_arg_map.split("(")[0] in self._gpu_array_obj.gpu_dict.keys():
          final_dict[arrays] = call_arg_map.split("(")[0]
        else:
          final_dict[arrays] = arrays

    return final_dict

  def _extract_scalars(self,all_variables):
    list_vars = all_variables.split(",")
    all_scalars = [x for x in list_vars if x not in self._cfregex.GPU_ARRAY_NAME_RE.findall(all_variables)]

    return all_scalars

  def _get_all_variables(self,vars_in_kernels_lines,all_scalars,attribute_status="cuf"):
    for idx,line in enumerate(vars_in_kernels_lines):
      if "::" in line and not self._cfregex.PARAMETER_CHECK.search(line):
        var_type = self._cfregex.VARS_AND_TYPE_RE.search(line).group(1)
        vars = self._cfregex.VARS_AND_TYPE_RE.search(line).group(2)
        main_scalars = vars.strip().replace(" ","").split(",")
        if "dimension" not in var_type:
          if "real" in var_type:
            for scalar in main_scalars:
              if scalar in all_scalars:
                self._var_info["real"].append(scalar)
              else:
                self._var_info["lreal"].append(scalar)
          if "integer" in var_type:
            for scalar in main_scalars:
              if scalar in all_scalars:
                self._var_info["integer"].append(scalar)
              else:
                self._var_info["linteger"].append(scalar)
          if "logical" in var_type:
            for scalar in main_scalars:
              if scalar in all_scalars:
                self._var_info["logical"].append(scalar)
              else:
                self._var_info["llogical"].append(scalar)
        if "dimension" in line and attribute_status != "cuf" and "_gpu" not in line:
          array_size = self._cfregex.CHECK_DIMENSION_ARRAY_RE.search(line).group(1)
          if "real" in var_type:
            for scalar in main_scalars:
              self._var_info["lreal_arrays"].append([scalar,array_size])
          if "integer" in var_type:
            for scalar in main_scalars:
              self._var_info["linteger_arrays"].append([scalar,array_size])
          if "logical" in var_type:
            for scalar in main_scalars:
              self._var_info["llogical_arrays"].append([scalar,array_size])

  def _common_routine(self,attribute):
    logging.info(f"Using the config file values into kernel_dict")
    self._use_config_dict()

    logging.info(f"Adding new subroutine name:{self._kernel_name}")
    self._subroutine_info["subroutine_name"] = self._kernel_name #if not self._kernel_name.endswith("_cuf") else self._kernel_name.replace("cuf","kernel")

    logging.info(f"Capture subroutine declaration and its dummy arguments for: {self._kernel_name}")
    self._subroutine_info["kernel_decl"],self._subroutine_info["kernel_args"] = self._subroutine_declaration(self._kernel_name,self._subroutine_info["full_subroutine"])

    logging.info(f"Extracting variables for: {self._kernel_name}")
    self._subroutine_info["kernel_variables"] = "".join(self._cfregex.CAPTURE_VARS_RE.findall(self._subroutine_info["full_subroutine"]))

    logging.info(f"Generate subroutine code for: {self._kernel_name}")
    try:
      self._subroutine_info["subroutine_code"] = self._cfregex.CAPTURE_INNER_SUB_RE.search(self._subroutine_info["full_subroutine"]).group(3)
    except:
      logging.error(f"Error in capturing inner part of the kernel subroutine for: {self._kernel_name}")
      raise Exception("Error!, check log file")

    logging.info(f"Extracting GPU arrays for: {self._kernel_name}")
    self._var_info["real_arrays"], self._var_info["int_arrays"], self._var_info["logical_arrays"] = self._extract_arrays(self._subroutine_info["kernel_variables"])

    logging.info(f"Extracting all variable types and attributes for: {self._kernel_name}")
    self._var_info["real"]=[]
    self._var_info["integer"]=[]
    self._var_info["logical"]=[]
    self._var_info["lreal"] = []
    self._var_info["linteger"] = []
    self._var_info["llogical"] = []

    self._var_info["lreal_arrays"] = []
    self._var_info["linteger_arrays"] = []
    self._var_info["llogical_arrays"] = []

    all_scalars = self._extract_scalars(",".join(self._subroutine_info["kernel_args"]))
    vars_in_kernels = self._subroutine_info["kernel_variables"]
    vars_in_kernels_lines = vars_in_kernels.split("\n")

    self._get_all_variables(vars_in_kernels_lines,all_scalars,attribute_status=attribute)

    # Map the subroutine arguments and subroutine call arguments
    logging.info(f"Mapping subroutine args and subroutine call args: {self._kernel_name}")
    logging.info(f"Mapping call and subroutine args for GPU arrays for: {self._kernel_name}")
    self._subroutine_info["args_map"] = []
    self._subroutine_info["array_index_map"] = []
    args_in_call = self._subroutine_info["sub_call_arguments"]

    for idx,args in enumerate(args_in_call):
      # Determine if kernel call and subroutine call array names differ and put them in main arrays
      for index,sub_args in enumerate(args):
        if sub_args.endswith("_gpu") and not self._subroutine_info["kernel_args"][index].endswith("_gpu"):
          local_psuedo_array = self._subroutine_info["kernel_args"][index]
          if local_psuedo_array in [sublist[0] for sublist in self._var_info["lreal_arrays"]]  and local_psuedo_array not in self._var_info["real_arrays"]:
            self._var_info["real_arrays"].append(local_psuedo_array)
            remove_index = [sublist[0] for sublist in self._var_info["lreal_arrays"]].index(local_psuedo_array)
            self._var_info["lreal_arrays"].pop(remove_index)
          elif local_psuedo_array in [sublist[0] for sublist in self._var_info["linteger_arrays"]] and local_psuedo_array not in self._var_info["int_arrays"]:
            self._var_info["int_arrays"].append(local_psuedo_array)
            remove_index = [sublist[0] for sublist in self._var_info["int_arrays"]].index(local_psuedo_array)
            self._var_info["linteger_arrays"].remove(local_psuedo_array)
          elif local_psuedo_array in [sublist[0] for sublist in self._var_info["llogical_arrays"]]  and local_psuedo_array not in self._var_info["logical_arrays"]:
            self._var_info["logical_arrays"].append(local_psuedo_array)
            remove_index = [sublist[0] for sublist in self._var_info["logical_arrays"]].index(local_psuedo_array)
            self._var_info["llogical_arrays"].remove(local_psuedo_array)

      mapping_of_vars = dict(zip(self._subroutine_info["kernel_args"],args))
      self._subroutine_info["args_map"].append(mapping_of_vars)

    for mapping in self._subroutine_info["args_map"]:
      self._subroutine_info["array_index_map"].append(self._get_array_index_map(self._var_info["real_arrays"]+
        self._var_info["int_arrays"]+self._var_info["logical_arrays"],mapping))

  def _look_for_device_calls(self,search_str):
    inner_code = search_str
    device_blueprint = {}
    for idx,d_kernel in enumerate(self._device_kernels):
      call = self._cfregex.SUBROUTINE_CALL_RE(d_kernel).findall(inner_code)
      if call:
        for c in call:
          if d_kernel in device_blueprint:
            kernel_count += 1
            device_blueprint[d_kernel][f"{d_kernel}_{self._kernel_name}_{kernel_count}"] = c[0] if not c[0][0]=="=" else c[0][1:]
          else:
            device_blueprint[d_kernel] = {}
            kernel_count = 0
            device_blueprint[d_kernel][f"{d_kernel}_{self._kernel_name}_{kernel_count}"] = c[0] if not c[0][0]=="=" else c[0][1:]
    return device_blueprint

  def _update_serial_code(self, serial, existing_serial_idx, existing_serial_size, to_change_serial_idx, to_change_serial_size):
    for id,val in enumerate(existing_serial_idx):
      old_idx = val
      old_size = existing_serial_size[id]
      new_idx = to_change_serial_idx[id]
      new_size = to_change_serial_size[id]
      # Capture this loop
      serial = self._cfregex.DO_2_RE(old_idx,old_size[0],old_size[1]).sub(f"do {new_idx} = {','.join(new_size)}", serial)

    return serial

class ProbeCuf(KernelProbe):
  def __init__(self,kernel_name,gpu_array_obj,kernel_dict,kernel_config,device_kernels):
    super().__init__(kernel_name,gpu_array_obj,kernel_dict,kernel_config,device_kernels=device_kernels)
    self._kernel_info["kernel"] = []
    self._kernel_info["non_cuf"] = []
    self._kernel_info["serial"] = []
    self._kernel_info["num_loop"] = []
    self._kernel_info["idx"] = []
    self._kernel_info["size"] = []
    self._kernel_info["blockdim"] = []
    self._kernel_info["non_kernel_def"] = ""
    self._subroutine_info["stream_in_call"] = []
    self._kernel_info["cuf_directive"] = []
    self._kernel_info["is_reduction"] = []
    self._kernel_info["launch_bounds"] = []
    self._device_kernels = device_kernels
    self._kernel_info["device_kernel_calls"]=[]

    self._common_routine("cuf")

    # idx and num_loop checks
    idx_num_loop_input = False
    if (self._kernel_info["num_loop"] and not self._kernel_info["idx"]) or (self._kernel_info["idx"] and not self._kernel_info["num_loop"]):
      raise Exception (f"Both idx and num_loop should exist not individually. Error in: {kernel_name}")
    elif (self._kernel_info["num_loop"] and self._kernel_info["idx"]):
      # Very important case to implement
      idx_num_loop_input = True
      input_num_loop = self._kernel_info["num_loop"].copy()
      input_idx = self._kernel_info["idx"].copy()
      for idx,nk in enumerate(input_num_loop):
        self._kernel_info["num_loop"][idx] = len(self._kernel_info["idx"][idx])
    else:
      logging.info(f"Extracting number of cuf loops for: {self._kernel_name}")
      num_kernels = self._cfregex.CUF_DO_RE.findall(self._subroutine_info["subroutine_code"])
      for nk in num_kernels:
        self._kernel_info["num_loop"].append(int(nk))

    logging.info(f"Extracting the non-parallel/serial part in the kernel for: {self._kernel_name}")
    subroutine_code = self._subroutine_info["subroutine_code"]
    subroutine_lines = subroutine_code.split("\n")
    num_loop = self._kernel_info["num_loop"]
    if (idx_num_loop_input) :
      _, _, capture_idx_full, capture_size_full, _, _ = \
      self._probe_cuf_loop(subroutine_lines,num_loop)
      self._kernel_info["num_loop"] = input_num_loop
      num_loop = self._kernel_info["num_loop"]

    kernel_count, self._kernel_info["cuf_directive"], capture_idx, capture_size, self._kernel_info["is_reduction"], self._kernel_info["serial"] = \
      self._probe_cuf_loop(subroutine_lines,num_loop)

    # Store the number of kernels in the subroutine
    self._kernel_info["kernel_count"] = kernel_count
    if self._kernel_info["kernel_count"] != len(num_loop):
      logging.error(f"Input count and the captured kernels do not match, Check again!")
      raise Exception("Check log file!")

    if idx_num_loop_input:
      # Case where both num_loop and idx are given in config file
      # Step 1: Update the captured full list based on the input ordering for parallel loops
      for kc in range(kernel_count):
        self._kernel_info["size"].append(update_list_order(capture_idx_full[kc],self._kernel_info["idx"][kc],capture_size_full[kc])[:self._kernel_info["num_loop"][kc]])
        self._kernel_info["idx"][kc] = self._kernel_info["idx"][kc][:self._kernel_info["num_loop"][kc]]
        if self._kernel_info["num_loop"][kc] <= len(input_idx[kc]):
          # Step 2: Serial part update
          # existing loop in serial if any
          num_loop_kc = self._kernel_info["num_loop"][kc]
          serial_code = self._kernel_info["serial"][kc]
          existing_serial_idx = capture_idx_full[kc][num_loop_kc:]
          existing_serial_size = capture_size_full[kc][num_loop_kc:]
          # Actual index and size we want in the serial part
          to_change_serial_idx = input_idx[kc][num_loop_kc:]
          indices_of_idx = [i for i,l in enumerate(capture_idx_full[kc]) if l in to_change_serial_idx]
          to_change_serial_size = [capture_size_full[kc][l] for i,l in enumerate(indices_of_idx)]
          self._kernel_info["serial"][kc] = self._update_serial_code(self._kernel_info["serial"][kc],\
            existing_serial_idx, existing_serial_size, to_change_serial_idx, to_change_serial_size)
        else:
          raise Exception(f"length of input idx is less than input num_loop for: {kernel_name}")
    else:
      # Case where no input is given
      for kc in range(kernel_count):
        self._kernel_info["idx"].append(capture_idx[kc])
        self._kernel_info["size"].append(capture_size[kc])

    # Look for device calls
    logging.info(f"Looking for device kernel calls inside cuf kernels for: {self._kernel_name}")
    for full_kernel in self._kernel_info["kernel"]:
      self._kernel_info["device_kernel_calls"].append(self._look_for_device_calls(full_kernel))

    logging.info(f"Extracting the non-kernel part for: {self._kernel_name}")
    self._kernel_info["non_cuf"] = self._get_non_cuf_part(self._subroutine_info["subroutine_code"],self._kernel_info["kernel"],len(num_loop))
    if self._kernel_info["non_cuf"] == [] or len(self._kernel_info["non_cuf"]) != kernel_count+1:
      logging.error(f"Look into non cuf extraction, could be a problem!")
      raise Exception("check log files!")

    logging.info(f"Extract stream information for: {self._kernel_name}")
    self._subroutine_info["stream_in_call"] = self._extract_stream()

    # replacing _cuf with replace_name
    logging.info(f"Renaming kernel name for: {self._kernel_name}")
    self._kernel_info["kernel_names"] = self._generate_kernel_names(len(num_loop),self._kernel_name)

    logging.info(f"Extracting variables in non-cuf part vars and adding to dict for: {self._kernel_name}")
    # Here, we are looking for variables in non cuf part. If found, add them to var_info["real"]/var_info["integer"]/var_info["logical"]
    # Because they need to be passed to the kernel
    for i in range(self._kernel_info["kernel_count"]):
      non_cuf_search_str = self._kernel_info["non_cuf"][i]
      all_vars = self._cfregex.NON_CUF_EXTRACT_RE.findall(non_cuf_search_str)
      all_vars = list(filter(None, all_vars))
      for av in all_vars:
        av = av.strip()
        if av not in self._subroutine_info["kernel_args"]:
          if av in self._var_info["lreal"] and av not in self._var_info["real"]:
            self._var_info["lreal"].remove(av)
            self._var_info["real"].append(av)
          if av in self._var_info["linteger"] and av not in self._var_info["integer"]:
            self._var_info["linteger"].remove(av)
            self._var_info["integer"].append(av)
          if av in self._var_info["llogical"] and av not in self._var_info["logical"]:
            self._var_info["llogical"].remove(av)
            self._var_info["logical"].append(av)

    self._reduction_info["reduction_data"] = []
    self._reduction_info["reduction_scalars"] = []
    for idx,reduction_status in enumerate(self._kernel_info["is_reduction"]):
      if reduction_status == True:
        logging.info(f"Extracting reduction information for a kernel in: {self._kernel_name}")
        directive = self._kernel_info["cuf_directive"][idx]
        all_match = self._cfregex.REDUCE_RE.findall(directive)
        redn_blueprint = {}
        for (type,vars) in all_match:
          redn_blueprint[type] = vars.replace(" ","").split(",")
        self._reduction_info["reduction_data"].append(redn_blueprint)
        for scalars in self._concatenate_lists(redn_blueprint):
          if scalars in self._var_info["real"]:
            self._var_info["real"].remove(scalars)
        # Create reduction scalars
        self._reduction_info["reduction_scalars"].append(self._concatenate_lists(redn_blueprint))
      else:
        self._reduction_info["reduction_data"].append("")

    logging.info(f"Adding blockdim definitions for: {self._kernel_name}")
    if not self._kernel_info["blockdim"]:
      num_loop = self._kernel_info["num_loop"]
      num_kernels = self._kernel_info["kernel_count"]
      for idx in range(num_kernels):
        if num_loop[idx] == 1:
          final_string = ["ONE_X"]
        elif num_loop[idx] == 2:
          final_string = ["TWO_X","TWO_Y"]
        elif num_loop[idx] == 3:
          final_string = ["THREE_X","THREE_Y","THREE_Z"]
        else:
          logging.error(f"Wrong count of the number of kernel loops, cannot be greater than 3. for: {self._kernel_name}")
          raise Exception("check log file!")

        self._kernel_info["blockdim"].append(final_string)

    logging.info(f"Extracting non kernel variable definitions for: {self._kernel_name}")
    self._kernel_info["non_kernel_def"] = self._get_non_kernel_var_definition()

  def _replace_subroutine(self, subroutine_lines, capture_idx, capture_size):
    for idx,line in enumerate(subroutine_lines):
      continue
    return subroutine_lines

  def _get_cuf_loop_data(self,kernel_name,parallel_loop_count,subroutine_lines,idx):
    idx_list = []
    size_list = []
    for i in range(parallel_loop_count):
      line_loop = subroutine_lines[idx+i+1]
      try:
        get_all_info = self._cfregex.DO_RE.search(line_loop)
        index = get_all_info.group(1)
        first_size = get_all_info.group(2)
        second_size = get_all_info.group(3)
      except:
        logging.error(f"Something wrong in capturing loop indices for: {kernel_name}")
        raise Exception("Error, Check log file!")

      # Get index info
      idx_list.append(index)

      # Get index range
      id_range = [f"{first_size}",f"{second_size}"]
      size_list.append(id_range)

    return idx_list,size_list

  def _get_kernel_and_serial(self,serial_part,idx,subroutine_lines,parallel_loop_count):
    enddo_counter = 0
    store_id = idx
    kernel_part = f"{subroutine_lines[store_id]}\n"
    while True:
      store_id += 1
      current_line = subroutine_lines[store_id]
      if self._cfregex.DO_RE.search(current_line) and current_line.replace(" ","") != "enddo":
        enddo_counter += 1
      if current_line.replace(" ","") == "enddo":
        enddo_counter -= 1
      kernel_part += f"{subroutine_lines[store_id]}\n"
      if enddo_counter == 0:
        break
    self._kernel_info["kernel"].append(kernel_part)
    remove_cuf = self._cfregex.CUF_REMOVE_RE.sub("",kernel_part).split("\n")
    remove_cuf = list(filter(None, remove_cuf))
    serial_part.append("\n".join(remove_cuf[parallel_loop_count:-parallel_loop_count]))

    return serial_part

  def _probe_cuf_loop(self, subroutine_lines, num_loop):

    kernel_count = 0 # output
    cuf_directive = [] # output
    capture_idx=[] # output
    capture_size=[] # output
    is_reduction = [] # output
    serial_part = [] # output
    # Loop through the code part of the kernel subroutine
    for idx,line in enumerate(subroutine_lines):
      # When $cuf is encountered
      if "$cuf" in line:

        # Store the directive
        cuf_directive.append(line.strip())
        # Retrieve the number of parallel do loops in the kernel
        parallel_loop_count = int(num_loop[kernel_count])

        # Check for reduction
        if self._cfregex.REDUCE_RE.search(line):
          is_reduction.append(True)
        else:
          is_reduction.append(False)

        # Storing the loop bounds of the kernel
        idx_out,size_out = self._get_cuf_loop_data(self._kernel_name,parallel_loop_count,subroutine_lines,idx)
        capture_idx.append(idx_out)
        capture_size.append(size_out)

        serial_part = self._get_kernel_and_serial(serial_part, idx,subroutine_lines,parallel_loop_count)
        # Increament kernel count
        kernel_count += 1
    return kernel_count, cuf_directive, capture_idx, capture_size, is_reduction, serial_part

  def _get_non_cuf_part(self,subroutine_code,full_kernel,num_of_kernels):
    splits=[]
    if num_of_kernels == 1:
      splits = subroutine_code.split(full_kernel[0])
      splits = [self._cfregex.CUF_REMOVE_RE.sub("",i) for i in splits]
      splits = [i.strip() for i in splits]
    elif num_of_kernels> 1:
      code_to_split = subroutine_code
      splits = []
      for i in range(num_of_kernels):
        split = code_to_split.split(full_kernel[i])
        code_to_split = split[1]
        splits.append(split[0])
      splits.append(code_to_split)
      splits = [self._cfregex.CUF_REMOVE_RE.sub("",i) for i in splits]
    for idx,sp in enumerate(splits):
      sp = "\n".join([i.strip().replace("\n","") for i in sp.split("\n")])
      splits[idx] = sp

    return splits

  def _extract_stream(self):
    final_list = []
    for directives in self._kernel_info["cuf_directive"]:
      try:
        stream_data = self._cfregex.GET_CUF_STREAM_RE.search(directives).group(1)
      except:
        stream_data = ""
        pass
      final_list.append(stream_data)
      # Also remove stream from integer list
      if stream_data in self._var_info["integer"]:
        self._var_info["integer"].remove(stream_data)

    return final_list

  def _generate_kernel_names(self,num_of_kernels,kernel_name):
    kernel_names = []
    if num_of_kernels == 1:
      kernel_name = kernel_name
      kernel_names = [kernel_name]
    else:
      for i in range(num_of_kernels):
        new_kernel_name = kernel_name+str(i+1)
        kernel_names.append(new_kernel_name)
    return kernel_names

  def _get_non_kernel_var_definition(self):
    kernel_vars = self._var_info["real_arrays"]+self._var_info["int_arrays"]+self._var_info["logical_arrays"]+self._var_info["real"]+self._var_info["integer"]+self._var_info["logical"]
    kernel_vars += self._concatenate_nested_lists(self._reduction_info["reduction_scalars"])
    kernel_local_vars = self._var_info["lreal"]+self._var_info["linteger"]+self._var_info["llogical"]
    non_kernel_vars = ""
    for line in self._subroutine_info["kernel_variables"].split("\n"):
      if line.replace("!","").strip() == "":continue
      var_type = line.split("::")[0]
      vars = line.split("::")[1]
      for v in vars.split(","):
        v = v.strip()
        v_exists = sum([v in i for i in self._kernel_info["non_cuf"]])
        if v not in kernel_vars and v not in kernel_local_vars and v_exists > 0:
          non_kernel_vars += f"{var_type.strip()} :: {v}\n"

    return non_kernel_vars

  def _concatenate_lists(self,dictionary):
    concatenated_list = []
    for value in dictionary.values():
        concatenated_list.extend(value)
    return concatenated_list
  def _concatenate_nested_lists(self,nested_list):
    concatenated_list = []
    for sublist in nested_list:
        concatenated_list.extend(sublist)
    return concatenated_list

class ProbeGlobal(KernelProbe):
  def __init__(self,kernel_name,gpu_array_obj,kernel_dict,kernel_config,device_kernels):
    super().__init__(kernel_name,gpu_array_obj,kernel_dict,kernel_config,device_kernels=device_kernels)
    self._kernel_info["kernel"] = []
    self._kernel_info["serial"] = []
    self._kernel_info["blockdim"] = []
    self._kernel_info["num_loop"] = None
    self._kernel_info["idx"] = []
    self._kernel_info["size"] = []
    self._kernel_info["launch_bounds"] = ""
    self._subroutine_info["stream_in_call"] = []
    self._device_kernels = device_kernels
    self._kernel_info["device_kernel_calls"]=""

    self._common_routine("global")

    logging.info(f"Looking for device kernel calls inside global kernels for: {self._kernel_name}")
    self._kernel_info["device_kernel_calls"] = self._look_for_device_calls(self._subroutine_info["subroutine_code"])

    logging.info(f"logging stream for:{self._kernel_name}")
    self._subroutine_info["stream_in_call"] = self._get_explicit_stream(self._subroutine_info["subroutine_call"])

    logging.info(f"Capturing idx,size and serial part for:{self._kernel_name}")
    self._kernel_info["idx"], self._kernel_info["size"], self._kernel_info["serial"], self._kernel_info["num_loop"] = self._get_explicit_info(self._subroutine_info["subroutine_code"],self._kernel_info["num_loop"],self._kernel_info["idx"],self._kernel_info["size"])

    logging.info(f"Add default blockdim config if not defined in config file: {self._kernel_name}")
    if not self._kernel_info["blockdim"]:
      if len(self._kernel_info["size"])==2:
        self._kernel_info["blockdim"]=["TWO_X","TWO_Y"]
      else:
        self._kernel_info["blockdim"]=["THREE_X","THREE_Y","THREE_Z"]

  def _get_explicit_stream(self,call):
    final_list = []
    for c in call:
      stream_capture = self._cfregex.GET_EXP_STREAM_RE.search(c).group(1)
      stream_split = stream_capture.split(",")

      if len(stream_split) < 4:
        final_str=""
      elif len(stream_split) > 2 and stream_split[3].strip() == "0":
        final_str=""
      else:
        final_str=stream_capture.split(",")[3]
      final_list.append(final_str.strip())
    return final_list

  def _get_explicit_info(self,subroutine_code,input_num_loop,input_idx,input_size):
    def get_first_index(index,id_name,string):
      grid_order = ["x","y","z"]
      thread_formula = lambda x: f"blockdim%{x}*(blockidx%{x}-1)+threadidx%{x}"
      loop_order_id = grid_order[index]
      thread_capture = self._cfregex.THREAD_ID_RE(loop_order_id).search(string.replace(" ",""))

      term_1 = f"blockdim%{loop_order_id}"
      term_2 = f"blockidx%{loop_order_id}"
      term_3 = f"threadidx%{loop_order_id}"

      # Evaluating: 3*(9-1)+7 = 31
      final_expected = eval(thread_formula(loop_order_id).replace(term_1,"3").replace(term_2,"9").replace(term_3,"7"))
      final_actual = eval(thread_capture.group(1).replace(term_1,"3").replace(term_2,"9").replace(term_3,"7"))

      try:
        assert final_expected == final_actual
      except:
        raise Exception(f"Thread formula used different from the standard form:blockDim%{loop_order_id}*(blockIdx%{loop_order_id}-1)+threadIdx%{loop_order_id}")

      first_index = "1"
      if thread_capture.group(2).strip() != "":
        first_index = f"{thread_capture.group(2)}+1"

      return first_index

    try:
      all_idx = self._cfregex.GLOBAL_KERNEL_IDX_RE.findall(subroutine_code)
      serial = self._cfregex.GLOBAL_KERNEL_IDX_RE.sub("",subroutine_code)
      size = []
      for index,id in enumerate(all_idx):
        first_index = get_first_index(index,id,subroutine_code)
        id_size = self._cfregex.GLOBAL_KERNEL_BOUND_RE(id).search(subroutine_code).group(1)
        size.append([first_index,id_size])
      serial = self._cfregex.GLOBAL_KERNEL_BOUND_RE(id).sub("",serial)
    except:
      raise Exception("Error! check log file")

    # Make changes here if input num_loop exists
    captured_num_loop = len(all_idx)
    if input_num_loop and input_idx:
      diff = (input_num_loop-captured_num_loop)
      final_serial = serial
      if diff > 0:
        additional_loop = diff
        do_counter = 0
        final_serial = ""
        for idx, line in enumerate(serial.strip().splitlines(True)):
          if idx == 0 and not self._cfregex.DO_RE.search(line.strip()):
            raise Exception("do loop does not exist in the top of the serial part to be extracted, check num_loop in config file!")
          elif self._cfregex.DO_RE.search(line.strip()) and do_counter != additional_loop:
            get_all_info = self._cfregex.DO_RE.search(line.strip())
            index = get_all_info.group(1)
            first_size = get_all_info.group(2)
            second_size = get_all_info.group(3)
            all_idx.append(index)
            size.append([first_size,second_size])
            do_counter += 1
          else:
            final_serial += line
        # Go in the reverse order and remove end do
        split_final_serial = final_serial.splitlines(True)
        for idx,line in enumerate(reversed(split_final_serial)):
          if line.replace(" ","") == "enddo" and do_counter != 0:
            split_final_serial[len(split_final_serial)-idx-1] = ""
            do_counter -= 1
        final_serial = "".join(split_final_serial)
      elif diff < 0:
        additional_loop = abs(diff)
        logging.error(f"Reduction of number of loops not possible!")
        raise Exception("decreasing parallel loop indices not supported yet!")

      if len(input_idx) > len(all_idx):
        # Follow same notation as cuf but invert it because parsing is inverted
        input_idx.reverse()
        diff_idx_len = len(input_idx) - len(all_idx)
        original_full_idx = all_idx.copy()
        original_full_size = size.copy()
        loop_idx = 0
        existing_serial_idx = []
        existing_serial_size = []

        for string in final_serial.strip().split("\n"):
          if self._cfregex.DO_RE.search(string):
            loop_idx += 1
            existing_serial_idx.append(self._cfregex.DO_RE.search(string).group(1))
            existing_serial_size.append([self._cfregex.DO_RE.search(string).group(2),self._cfregex.DO_RE.search(string).group(3)])
            original_full_idx.append(self._cfregex.DO_RE.search(string).group(1))
            original_full_size.append([self._cfregex.DO_RE.search(string).group(2),self._cfregex.DO_RE.search(string).group(3)])
            if loop_idx == diff_idx_len : break
        to_change_serial_idx = input_idx[:input_num_loop-1]

        indices_of_idx = [i for i,l in enumerate(original_full_idx) if l in to_change_serial_idx]
        to_change_serial_size = [original_full_size[l] for i,l in enumerate(indices_of_idx)]
        final_serial = self._update_serial_code(final_serial,existing_serial_idx, existing_serial_size, to_change_serial_idx, to_change_serial_size)

        all_idx = input_idx[-input_num_loop:]
        size = []
        for val in all_idx:
          if val in original_full_idx:
            chosen_idx = original_full_idx.index(val)
            size.append(original_full_size[chosen_idx])
        all_idx.reverse()
        size.reverse()

      elif len(input_idx) == len(all_idx):
        size = update_list_order(all_idx,input_idx,size)
        all_idx = input_idx
      else:
        logging.error("input_idx cannot be smaller than parsed idx, check config file again")
        raise Exception("input_idx cannot be smaller than parsed idx, check config file again")

      return all_idx,size,final_serial,len(all_idx)
    elif (not input_num_loop and input_idx) or (input_num_loop and not input_idx):
      logging.error(f"Both num_loop and idx must be present together")
      raise Exception("Both num_loop and idx must be present together in the config file")
    else:
      return reverse_list(all_idx),reverse_list(size),serial,len(all_idx)

class ProbeDevice(KernelProbe):
  def __init__(self,kernel_name,gpu_array_obj,kernel_dict,kernel_config,is_device_func):
    super().__init__(kernel_name,gpu_array_obj,kernel_dict,kernel_config)
    self._is_device_func = is_device_func
    self._common_routine("device")
    self._var_info["pointer_variables"] = []

    # Find pointers
    subroutine_code = self._subroutine_info["subroutine_code"]
    kernel_args = self._subroutine_info["kernel_args"]
    for vars in kernel_args:
      if self._cfregex.POINTER_CHECK_RE(vars).search(subroutine_code):
        self._var_info["pointer_variables"].append(vars)

    logging.info(f"Determining function/subroutine characteristics:{self._kernel_name}")
    self._kernel_info["is_device_func"]=False
    self._kernel_info["function_type"]="void"
    self._kernel_info["return_value"] = ""
    if self._is_device_func:
      self._kernel_info["is_device_func"] = True
      self._kernel_info["return_value"] = kernel_name
      if kernel_name in self._var_info["lreal"]:
        self._kernel_info["function_type"]="real"
      elif kernel_name in self._var_info["linteger"]:
        self._kernel_info["function_type"]="int"
      elif kernel_name in self._var_info["llogical"]:
        self._kernel_info["function_type"]="bool"
      else:
        logging.error(f"Data type for device kernel not implemented for:{kernel_name}")
        raise Exception("Check logs!")