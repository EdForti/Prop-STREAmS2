from abc import ABC, abstractmethod
from backend_generation.prepare_backend import PrepareAmd,PrepareCpu,PrepareOmp,PrepareOmpc
from backend_generation.regular_expressions.regex import CudaFRegularExpressions
from backend_generation.backend_tools import extract_backend_config
import logging
from tools import write_json,write_output,handle_line_break
from tqdm import tqdm
import os
import re
from re import I, compile,escape,search,sub
from mako.lookup import TemplateLookup
from indent.indent import StreamsIndent
from indent.assoc_clean import StreamsAssoclean

BACKEND_PATH = os.path.realpath(os.path.dirname(__file__))

class GenerateBackend(ABC):
  def __init__(self,extract_object,args):
    self._extract_object = extract_object
    self._template_path = TemplateLookup(directories=BACKEND_PATH)
    self._cfregex = CudaFRegularExpressions()
    self._backend_config = self._extract_object.backend_config
    self._input_args = args
    self._indent = StreamsIndent()
    #self._assoc_clean = StreamsAssoclean()

  @abstractmethod
  def _base_file(self):
    raise NotImplementedError

  @abstractmethod
  def _equation_file(self):
    raise NotImplementedError

  @abstractmethod
  def _equation_gpu_file(self):
    raise NotImplementedError

  @abstractmethod
  def _kernel_file(self):
    raise NotImplementedError

  @abstractmethod
  def generatefiles(self):
    raise NotImplementedError

  @abstractmethod
  def _array_declaration(self):
    raise NotImplementedError

  @abstractmethod
  def _handle_streams_parameters(self):
    raise NotImplementedError

  def _replace_subroutines(self,string,subroutine_list):
    for subr in subroutine_list:
      name = self._cfregex.SUBROUTINE_NAME_RE.search(subr).group(2)
      string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(name).sub(subr,string)

    return string

  def _add_modules(self,string,module):
    return self._cfregex.CUDAFOR_RE.sub(module,string)

  def _gpuarrayfile(self):
    if self._extract_object.gpu_array_obj.gpu_array_file:
      print(f"\nWriting GPU array file:{self._extract_object.gpu_array_file}")
      write_output(self._extract_object.gpu_array_file,self._extract_object.gpu_array_obj.gpu_array_file)

  def _log_prepare_dict(self,print_dict):
    print(f"\nWriting prepared_kernels.json")
    write_json("log_prepared_kernels.json",print_dict)

  # TODO: Add tests
  def _cpu_gpu_transfers(self,string,gpu_dict,backend):
    def transfer(final_matches,string,backend,flag):
      def adjust_index(side,bounds):
        if len(side.split("("))>1:
          orig_bounds = [[i.split(":")[0],i.split(":")[1]] for i in bounds.split(",")]
          get_brackets = search(r"\((.*?)\)",side,I).group(1)
          final_side = []
          for idx,val in enumerate(get_brackets.split(",")):
            ob = orig_bounds[idx]
            if val.strip()==":":
              final_side.append(":")
              continue
            else:
              if len(val.split(":"))>1:
                left = f"({val.split(':')[0]})-({ob[0]})+1"
                right = f"({val.split(':')[1]})-({ob[0]})+1"
                final_side.append(f"{left}:{right}")
              else:
                final_side.append(f"({val})-({ob[0]})+1")
          return f"{side.split('(')[0]}({','.join(final_side)})"
        else:
          return side
      for m in final_matches:
        m = m[0].strip()
        lhs = m.split("=")[0]
        rhs = m.split("=")[1]

        if m:
          if lhs.strip()=="" or rhs.strip()=="":
            raise Exception(f"Something wrong in capturing CPU GPU transfer for:{m}")
        final_str = ""
        if flag == 0:
          if rhs.split("(")[0].strip().endswith("_gpu"):
            if backend == "amd" : final_str = f"call hipCheck(hipMemcpy({lhs},{rhs},hipMemcpyDeviceToDevice))"
            if backend == "omp" : final_str = f"self%ierr = omp_target_memcpy_f({lhs},{rhs},0,0,self%mydev,self%mydev)"
            if backend == "omp-eqn" : final_str = f"self%ierr = omp_target_memcpy_f({lhs},{rhs},0,0,self%base_omp%mydev,self%base_omp%mydev)"
          else:
            # Adjust lhs
            adjusted_lhs = adjust_index(lhs,value[3])
            if backend == "amd" : final_str = f"call hipCheck(hipMemcpy({adjusted_lhs},{rhs},hipMemcpyHostToDevice))"
            if backend == "omp" : final_str = f"self%ierr = omp_target_memcpy_f({adjusted_lhs},{rhs},0,0,self%mydev,self%myhost)"
            if backend == "omp-eqn" : final_str = f"self%ierr = omp_target_memcpy_f({adjusted_lhs},{rhs},0,0,self%base_omp%mydev,self%base_omp%myhost)"
        elif flag == 1:
          if lhs.split("(")[0].strip().endswith("_gpu"):
            if backend == "amd" : final_str = f"call hipCheck(hipMemcpy({lhs},{rhs},hipMemcpyDeviceToDevice))"
            if backend == "omp" : final_str = f"self%ierr = omp_target_memcpy_f({lhs},{rhs},0,0,self%mydev,self%mydev)"
            if backend == "omp-eqn" : final_str = f"self%ierr = omp_target_memcpy_f({lhs},{rhs},0,0,self%base_omp%mydev,self%base_omp%mydev)"
          else:
            # Adjust rhs
            adjusted_rhs = adjust_index(rhs,value[3])
            if backend == "amd" : final_str = f"call hipCheck(hipMemcpy({lhs},{adjusted_rhs},hipMemcpyDeviceToHost))"
            if backend == "omp" : final_str = f"self%ierr = omp_target_memcpy_f({lhs},{adjusted_rhs},0,0,self%myhost,self%mydev)"
            if backend == "omp-eqn" : final_str = f"self%ierr = omp_target_memcpy_f({lhs},{adjusted_rhs},0,0,self%base_omp%myhost,self%base_omp%mydev)"
        if final_str == "":
          raise Exception(f"CPU GPU transfer issue for:{m},{flag},{backend}")
        string = string.replace(m,final_str)

      return string

    for key,value in gpu_dict.items():
      full_name = value[5]
      # CPU to GPU
      final_matches = self._cfregex.CPU_GPU_RE(full_name).findall(string)
      string = transfer(final_matches,string,backend,flag=0)
      final_matches = self._cfregex.CPU_GPU_RE(key).findall(string)
      string = transfer(final_matches,string,backend,flag=0)

      # GPU to CPU
      final_matches = self._cfregex.GPU_CPU_RE(full_name).findall(string)
      string = transfer(final_matches,string,backend,flag=1)
      final_matches = self._cfregex.GPU_CPU_RE(key).findall(string)
      string = transfer(final_matches,string,backend,flag=1)

    return string

  def _treat_allocation(self,string,gpu_dict,backend):
    def get_alloc(name,size):
      var_size_lst = []
      ubound = []
      lbound = []
      for vs in size.split(","):
        vs_split = vs.split(":")
        var_size_lst.append(f"({vs_split[1]})-({vs_split[0]})+1")
        lbound.append(vs_split[0])
        ubound.append(vs_split[1])
      var_size_hip = ",".join(var_size_lst)
      if backend=="amd":
        return f"call hipCheck(hipMalloc(self%{name},{var_size_hip}))"
      elif backend=="omp":
        return f"call omp_target_alloc_f(fptr_dev=self%{name},ubounds=[{','.join(ubound)}],lbounds=[{','.join(lbound)}],omp_dev=self%mydev,ierr=self%ierr)"
      elif backend=="omp-eqn":
        return f"call omp_target_alloc_f(fptr_dev=self%{name},ubounds=[{','.join(ubound)}],lbounds=[{','.join(lbound)}],omp_dev=self%base_omp%mydev,ierr=self%ierr)"

    alloc_list = self._cfregex.ALLOC_FOR_VAR_RE.findall(string)
    for alloc,arrays in alloc_list:
      old_alloc = alloc
      arrays = arrays.lower()
      array_names = []
      if arrays.count('_gpu') == 1:
        array_names = self._cfregex.ALLOC_VAR_ATTRIB_SINGLE_RE.findall(arrays)
      elif arrays.count('_gpu')>1:
        array_names = self._cfregex.ALLOC_VAR_ATTRIB_MULT_RE.findall(arrays)
      else:
        continue

      if len(array_names) == 1:
        alloc_final = get_alloc(array_names[0][0],gpu_dict[array_names[0][0]][4])
        string = string.replace(old_alloc,alloc_final)
      elif len(array_names) > 1:
        final_str = ""
        for an in array_names:
          final_str += get_alloc(an[0],gpu_dict[an[0]][4])+"\n"
        string = string.replace(old_alloc,final_str)

    return string

class GenerateAmd(GenerateBackend):
  def __init__(self,extract_object,args):
    super().__init__(extract_object,args)
    self._prepare_kernels = PrepareAmd(extract_object.backend_dict,extract_object.gpu_array_obj)

  def _handle_streams_parameters(self,string):
    match = compile(r".*use streams_parameters.*",I)
    string = match.sub("use streams_parameters",string)

    return string

  # TODO: Add tests
  def _array_declaration(self,string):
    string = string.split("\n")
    for idx,line in enumerate(string):
      split_by_colon = line.split("::")
      if len(split_by_colon) > 1:
        all_types = split_by_colon[0]
        if "allocatable" in all_types:
          string[idx] = self._cfregex.ALLOCATABLE_GPU_RE.sub(r"\1\2",string[idx])
          if "device" in all_types:
            string[idx] = self._cfregex.DEVICE_FLAG_RE.sub(r"\1pointer",string[idx])
        if "device" in all_types and "dimension" in all_types:
          dim_str_old = self._cfregex.CHECK_DIMENSION_ARRAY_RE.search(all_types).group(1)
          dim_str_new = ':' + ',:'.join([''] * (len(dim_str_old.split(","))))
          new_all_types = all_types.replace(dim_str_old,dim_str_new)
          new_all_types = new_all_types.replace("device","target")
          string[idx] = string[idx].replace(all_types,new_all_types)

    return "\n".join(string)

  def _equation_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation file:{self._extract_object.equation_amd}")
    if not final_string:
      final_string = self._extract_object.equation_file

    write_output(self._extract_object.equation_amd,handle_line_break(final_string))

  def _base_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Base file:{self._extract_object.base_amd}")
    if not final_string:
      final_string = self._extract_object.base_file

    # CPU - GPU transfers
    final_string = self._cpu_gpu_transfers(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="amd")

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_amd_object")
    final_string = final_string.replace("base_gpu","base_amd")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_amd")

    # Add modules
    final_string = self._add_modules(final_string,"use hipfort\nuse hipfort_check")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="amd")

    # Add interfaces
    final_string = re.sub(re.compile(r"(\s*implicit none\s*\n)"),r"\1"+"\n".join(self._prepare_kernels.base_interface),final_string)

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace cuda stream
    final_string = final_string.replace("0_cuda_stream_kind","c_null_ptr")

    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")

    write_output(self._extract_object.base_amd,handle_line_break(final_string))

  def _equation_gpu_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation GPU file:{self._extract_object.equation_gpu_amd}")
    if not final_string:
      final_string = self._extract_object.equation_gpu_file

    # CPU - GPU transfers
    final_string = self._cpu_gpu_transfers(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="amd")

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_amd_object")
    final_string = final_string.replace("base_gpu","base_amd")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_amd")

    # Add modules
    final_string = self._add_modules(final_string,"use hipfort\nuse hipfort_check")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add reduction variable
    final_string = re.sub(re.compile(r"((.*real.*pointer.*)wmean_gpu(.*\n))",I),r"\1\2redn_3d_gpu",final_string)
    final_string = re.sub(re.compile(r"((.*allocate.*)wmean_gpu.*\(.*?\)(.*\n))",I),r"\1\2redn_3d_gpu(1:nx,1:ny,1:nz)\3",final_string)

    # Add stream and Device Sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"type(c_ptr)\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"call hipCheck(hipStreamCreate(\1))",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("call hipCheck(hipDeviceSynchronize())",final_string)

    # Add allocations
    final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="amd")

    # Add interfaces
    final_string = re.sub(re.compile(r"(\s*implicit none\s*\n)"),r"\1"+"\n".join(self._prepare_kernels.equation_interface),final_string)

    # Replace cuda stream
    final_string = final_string.replace("0_cuda_stream_kind","c_null_ptr")

    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")

    write_output(self._extract_object.equation_gpu_amd,handle_line_break(final_string))

  def _kernel_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Kernel file:{self._extract_object.kernel_amd}")
    if not final_string:
      final_string = self._extract_object.kernel_file

    # Renaming
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_amd")

    # Remove global and device kernels
    final_string = self._cfregex.ALL_GLOBAL_DEVICE_RE.sub("",final_string)

    # Remove uncalled kernels
    for kernels in self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)

    # Add modules
    final_string = self._add_modules(final_string,"use hipfort\nuse hipfort_check")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add interfaces
    final_string = re.sub(re.compile(r"(\s*implicit none\s*\n)"),r"\1"+"\n".join(self._prepare_kernels.kernel_interface),final_string)

    # Replace cuda stream
    final_string = final_string.replace("0_cuda_stream_kind","c_null_ptr")

    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")

    write_output(self._extract_object.kernel_amd,handle_line_break(final_string))

  def _basecppfile(self,final_list,backend_config=None):
    print(f"\nWriting Base Cpp file:{self._extract_object.base_amd_cpp}")
    TEMPLATE_FILE = self._template_path.get_template("templates/base_kernels_hipfort.cpp")
    final_string = TEMPLATE_FILE.render(kernels=final_list)
    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")
    write_output(self._extract_object.base_amd_cpp,final_string)

  def _kernelcppfile(self,final_list,backend_config=None):
    print(f"\nWriting Kernel Cpp file:{self._extract_object.kernel_amd_cpp}")
    TEMPLATE_FILE = self._template_path.get_template("templates/kernels_hipfort.cpp")
    final_string = TEMPLATE_FILE.render(kernels=final_list)
    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")
    write_output(self._extract_object.kernel_amd_cpp,final_string)

  def generatefiles(self):
    logging.info("Generating files for AMD backend")

    print("\nPreparing kernels for AMD backend")
    self._prepare_kernels.generate_kernels()
    self._log_prepare_dict(self._prepare_kernels.prepare_dict)

    call_subroutine_dict = {"base_file":self._extract_object.base_file,"equation_file":self._extract_object.equation_gpu_file,"kernel_file":self._extract_object.kernel_file}
    print(f"\nReplacing subroutines and calls")
    iterator = tqdm(self._prepare_kernels.prepare_dict.items())
    base_cpp_list = []
    kernel_cpp_list = []
    for kernel,value in iterator:
      iterator.set_postfix(batch=kernel)
      logging.info(f"Replacing calls and subroutines for:{kernel}")
      all_calls = value["call_handling"]
      for file,calls in all_calls.items():
        for c in calls:
          call_subroutine_dict[file] = call_subroutine_dict[file].replace(c[0],c[1])

      try:
        subroutine = value["subroutine_handling"]
        for file,calls in subroutine.items():
          call_subroutine_dict[file] = call_subroutine_dict[file].replace(calls[0],calls[1])
      except:
        logging.warning(f"Kernel subroutine does not exist for:{kernel}")

      subroutine_source = value["source"]
      if subroutine_source == "base_file":
        base_cpp_list.append(value["full_kernel"])
      elif subroutine_source == "kernel_file":
        kernel_cpp_list.append(value["full_kernel"])

    self._gpuarrayfile()
    self._base_file(backend_config=extract_backend_config(self._backend_config,"base_file","amd"),final_string=call_subroutine_dict["base_file"])
    self._equation_file()
    self._equation_gpu_file(final_string=call_subroutine_dict["equation_file"])
    self._kernel_file(final_string=call_subroutine_dict["kernel_file"])
    self._basecppfile(base_cpp_list)
    self._kernelcppfile(kernel_cpp_list)
    if self._input_args.indent:
      print("\nIndenting output code")
      self._indent.indent(self._extract_object.output_code)
      #self._assoc_clean.assoclean(self._extract_object.output_code)

class GenerateCpu(GenerateBackend):
  def __init__(self,extract_object,args):
    super().__init__(extract_object,args)
    self._prepare_kernels = PrepareCpu(extract_object.backend_dict,extract_object.gpu_array_obj)

  def _array_declaration(self,string):
    string = string.split("\n")
    for idx,line in enumerate(string):
      split_by_colon = line.split("::")
      if len(split_by_colon) > 1:
        all_types = split_by_colon[0]
        if "allocatable" in all_types:
          string[idx] = self._cfregex.ALLOCATABLE_GPU_RE.sub(r"\1, allocatable\2",string[idx])
          if "device" in all_types:
            string[idx] = self._cfregex.DEVICE_FLAG_RE.sub("",string[idx])
        if "device" in all_types and "dimension" in all_types:
          string[idx] = self._cfregex.DEVICE_FLAG_RE.sub(r"",string[idx])
          # dim_str_old = self._cfregex.CHECK_DIMENSION_ARRAY_RE.search(all_types).group(1)
          # dim_str_new = ':' + ',:'.join([''] * (len(dim_str_old.split(","))))
          # new_all_types = all_types.replace(dim_str_old,dim_str_new)
          # new_all_types = self._cfregex.DEVICE_FLAG_RE.sub("",new_all_types)
          # string[idx] = string[idx].replace(all_types,new_all_types)

    return "\n".join(string)

  def _handle_streams_parameters(self,string):
    match = compile(r".*use streams_parameters.*",I)
    string = match.sub("use streams_parameters",string)

    return string

  def _base_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Base file:{self._extract_object.base_cpu}")
    if not final_string:
      final_string = self._extract_object.base_file

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Remove uncalled kernels
    uncalled_kernels = self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]+["check_gpu_mem"]
    for kernels in uncalled_kernels:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)
      final_string = self._cfregex.SUB_CALL_RE(kernels).sub("",final_string)
      final_string = self._cfregex.CAPTURE_PROC_DEF_RE(kernels).sub("",final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_cpu_object")
    final_string = final_string.replace("base_gpu","base_cpu")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_cpu")
    final_string = sub(compile(r"(copy_)gpu(_cpu)",I),r"copy_to_field",final_string)
    final_string = sub(compile(r"(copy_cpu_)gpu",I),r"copy_from_field",final_string)
    final_string = final_string.replace("_gpu","_cpu")

    # Add modules
    final_string = self._add_modules(final_string,"")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    #final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _subroutine
    final_string = final_string.replace("_cuf","_subroutine")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.base_cpu,handle_line_break(final_string))

  def _equation_gpu_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation GPU file:{self._extract_object.equation_gpu_cpu}")
    if not final_string:
      final_string = self._extract_object.equation_gpu_file

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Remove uncalled kernels
    uncalled_kernels = self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]+["check_gpu_mem"]
    for kernels in uncalled_kernels:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)
      final_string = self._cfregex.SUB_CALL_RE(kernels).sub("",final_string)
      final_string = self._cfregex.CAPTURE_PROC_DEF_RE(kernels).sub("",final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_cpu_object")
    final_string = final_string.replace("base_gpu","base_cpu")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_cpu")
    final_string = sub(compile(r"(copy_)gpu(_cpu)",I),r"copy_to_field",final_string)
    final_string = sub(compile(r"(copy_cpu_)gpu",I),r"copy_from_field",final_string)
    final_string = final_string.replace("_gpu","_cpu")

    # Add modules
    final_string = self._add_modules(final_string,"")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    #final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp-eqn")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _subroutine
    final_string = final_string.replace("_cuf","_subroutine")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.equation_gpu_cpu,handle_line_break(final_string))

  def _kernel_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Kernel file:{self._extract_object.kernel_cpu}")
    if not final_string:
      final_string = self._extract_object.kernel_file

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Remove uncalled kernels
    uncalled_kernels = self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]+["check_gpu_mem"]
    for kernels in uncalled_kernels:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)
      final_string = self._cfregex.SUB_CALL_RE(kernels).sub("",final_string)
      final_string = self._cfregex.CAPTURE_PROC_DEF_RE(kernels).sub("",final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_cpu_object")
    final_string = final_string.replace("base_gpu","base_cpu")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_cpu")
    final_string = sub(compile(r"(copy_)gpu(_cpu)",I),r"copy_to_field",final_string)
    final_string = sub(compile(r"(copy_cpu_)gpu",I),r"copy_from_field",final_string)
    final_string = final_string.replace("_gpu","_cpu")

    # Add modules
    final_string = self._add_modules(final_string,"")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    #final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _subroutine
    final_string = final_string.replace("_cuf","_subroutine")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.kernel_cpu,handle_line_break(final_string))

  def _equation_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation file:{self._extract_object.equation_cpu}")
    if not final_string:
      final_string = self._extract_object.equation_file

    write_output(self._extract_object.equation_cpu,handle_line_break(final_string))

  def generatefiles(self):
    logging.info("Generating files for CPU backend")

    print("\nPreparing kernels for CPU backend")
    self._prepare_kernels.generate_kernels()
    self._log_prepare_dict(self._prepare_kernels.prepare_dict)

    call_subroutine_dict = {"base_file":self._extract_object.base_file,"equation_file":self._extract_object.equation_gpu_file,"kernel_file":self._extract_object.kernel_file}
    print(f"\nReplacing subroutines and calls")
    iterator = tqdm(self._prepare_kernels.prepare_dict.items())
    for kernel,value in iterator:
      iterator.set_postfix(batch=kernel)
      logging.info(f"Replacing subroutines for:{kernel}")

      try:
        subroutine = value["subroutine_handling"]
        for file,calls in subroutine.items():
          call_subroutine_dict[file] = call_subroutine_dict[file].replace(calls[0],calls[1])
      except:
        logging.warning(f"Kernel subroutine does not exist for:{kernel}")

    self._base_file(backend_config=extract_backend_config(self._backend_config,"base_file","cpu"),final_string=call_subroutine_dict["base_file"])
    self._equation_file()
    self._equation_gpu_file(final_string=call_subroutine_dict["equation_file"],backend_config=extract_backend_config(self._backend_config,"equation_gpu_file","cpu"))
    self._kernel_file(final_string=call_subroutine_dict["kernel_file"],backend_config=extract_backend_config(self._backend_config,"kernels_file","cpu"))
    if self._input_args.indent:
      print("\nIndenting output code")
      self._indent.indent(self._extract_object.output_code)
      #self._assoc_clean.assoclean(self._extract_object.output_code)

class GenerateOmp(GenerateBackend):
  def __init__(self,extract_object,args):
    super().__init__(extract_object,args)
    self._prepare_kernels = PrepareOmp(extract_object.backend_dict,extract_object.gpu_array_obj)

  def _base_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Base file:{self._extract_object.base_omp}")
    if not final_string:
      final_string = self._extract_object.base_file

    # Temporary rename
    final_string = self._add_subroutines(final_string,"base_file",flag=0)

    # CPU - GPU transfers
    final_string = self._cpu_gpu_transfers(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Rename back
    final_string = self._add_subroutines(final_string,"base_file",flag=1)

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_omp_object")
    final_string = final_string.replace("base_gpu","base_omp")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_omp")

    # Remove uncalled kernels
    for kernels in self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)

    # Add modules
    final_string = self._add_modules(final_string,"use utils_omp\nuse omp_lib")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.base_omp,handle_line_break(final_string))

  def _equation_gpu_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation GPU file:{self._extract_object.equation_gpu_omp}")
    if not final_string:
      final_string = self._extract_object.equation_gpu_file

    # Temporary rename
    final_string = self._add_subroutines(final_string,"equation_file",flag=0)

    # CPU - GPU transfers
    final_string = self._cpu_gpu_transfers(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp-eqn")

    # Rename back
    final_string = self._add_subroutines(final_string,"equation_file",flag=1)

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_omp_object")
    final_string = final_string.replace("base_gpu","base_omp")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_omp")

    # Remove uncalled kernels
    for kernels in self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)

    # Add modules
    final_string = self._add_modules(final_string,"use utils_omp\nuse omp_lib")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp-eqn")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.equation_gpu_omp,handle_line_break(final_string))

  def _kernel_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Kernel file:{self._extract_object.kernel_omp}")
    if not final_string:
      final_string = self._extract_object.kernel_file

    # Temporary rename
    # final_string = self._add_subroutines(final_string,"kernel_file",flag=0)

    # CPU - GPU transfers
    #final_string = self._cpu_gpu_transfers(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Rename back
    # final_string = self._add_subroutines(final_string,"kernel_file",flag=1)
    
    final_string = self._add_subroutines(final_string,"kernel_file")

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_omp_object")
    final_string = final_string.replace("base_gpu","base_omp")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_omp")

    # Remove uncalled kernels
    for kernels in self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)

    # Add modules
    final_string = self._add_modules(final_string,"use utils_omp\nuse omp_lib")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _kernel
    final_string = final_string.replace("_cuf","_kernel")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.kernel_omp,handle_line_break(final_string))

  def _equation_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation file:{self._extract_object.equation_omp}")
    if not final_string:
      final_string = self._extract_object.equation_file

    write_output(self._extract_object.equation_omp,handle_line_break(final_string))

  def _array_declaration(self,string):
    string = string.split("\n")
    for idx,line in enumerate(string):
      split_by_colon = line.split("::")
      if len(split_by_colon) > 1:
        all_types = split_by_colon[0]
        if "allocatable" in all_types:
          string[idx] = self._cfregex.ALLOCATABLE_GPU_RE.sub(r"\1\2",string[idx])
          if "device" in all_types:
            string[idx] = self._cfregex.DEVICE_FLAG_RE.sub(r"\1pointer",string[idx])
        if "device" in all_types and "dimension" in all_types:
          string[idx] = self._cfregex.DEVICE_FLAG_RE.sub(r"",string[idx])
          # dim_str_old = self._cfregex.CHECK_DIMENSION_ARRAY_RE.search(all_types).group(1)
          # dim_str_new = ':' + ',:'.join([''] * (len(dim_str_old.split(","))))
          # new_all_types = all_types.replace(dim_str_old,dim_str_new)
          # new_all_types = self._cfregex.DEVICE_FLAG_RE.sub("",new_all_types)
          # string[idx] = string[idx].replace(all_types,new_all_types)

    return "\n".join(string)

  def _handle_streams_parameters(self,string):
    match = compile(r".*use streams_parameters.*",I)
    string = match.sub("use streams_parameters",string)

    return string

  def _add_subroutines(self,final_string,file_type,flag=3):
    for kernel,value in self._prepare_kernels.prepare_dict.items():
      try:
        subroutine = value["subroutine_handling"]
        for ff,calls in subroutine.items():
          if ff==file_type:
            if flag==1:
              final_string = final_string.replace(f"REPLACE_TEMPORARY:{kernel}",calls[1])
            elif flag==0:
              final_string = final_string.replace(calls[0],f"REPLACE_TEMPORARY:{kernel}")
            elif flag==3:
              final_string = final_string.replace(calls[0],calls[1])
      except:
        logging.warning(f"Kernel subroutine does not exist for:{kernel}")
    return final_string

  def generatefiles(self):
    logging.info("Generating files for OMP backend")

    print("\nPreparing kernels for OMP backend")
    self._prepare_kernels.generate_kernels()
    self._log_prepare_dict(self._prepare_kernels.prepare_dict)

    call_subroutine_dict = {"base_file":self._extract_object.base_file,"equation_file":self._extract_object.equation_gpu_file,"kernel_file":self._extract_object.kernel_file}

    self._base_file(backend_config=extract_backend_config(self._backend_config,"base_file","omp"),final_string=call_subroutine_dict["base_file"])
    self._equation_file()
    self._equation_gpu_file(backend_config=extract_backend_config(self._backend_config,"equation_gpu_file","omp"),final_string=call_subroutine_dict["equation_file"])
    self._kernel_file(backend_config=extract_backend_config(self._backend_config,"kernels_file","omp"),final_string=call_subroutine_dict["kernel_file"])

    if self._input_args.indent:
      print("\nIndenting output code")
      self._indent.indent(self._extract_object.output_code)
      #self._assoc_clean.assoclean(self._extract_object.output_code)

class GenerateOmpc(GenerateBackend):
  def __init__(self,extract_object,args):
    super().__init__(extract_object,args)
    self._prepare_kernels = PrepareOmpc(extract_object.backend_dict,extract_object.gpu_array_obj)

  def _array_declaration(self,string):
    string = string.split("\n")
    for idx,line in enumerate(string):
      split_by_colon = line.split("::")
      if len(split_by_colon) > 1:
        all_types = split_by_colon[0]
        if "allocatable" in all_types:
          string[idx] = self._cfregex.ALLOCATABLE_GPU_RE.sub(r"\1, allocatable\2",string[idx])
          if "device" in all_types:
            string[idx] = self._cfregex.DEVICE_FLAG_RE.sub("",string[idx])
        if "device" in all_types and "dimension" in all_types:
          string[idx] = self._cfregex.DEVICE_FLAG_RE.sub(r"",string[idx])
          # dim_str_old = self._cfregex.CHECK_DIMENSION_ARRAY_RE.search(all_types).group(1)
          # dim_str_new = ':' + ',:'.join([''] * (len(dim_str_old.split(","))))
          # new_all_types = all_types.replace(dim_str_old,dim_str_new)
          # new_all_types = self._cfregex.DEVICE_FLAG_RE.sub("",new_all_types)
          # string[idx] = string[idx].replace(all_types,new_all_types)

    return "\n".join(string)

  def _handle_streams_parameters(self,string):
    match = compile(r".*use streams_parameters.*",I)
    string = match.sub("use streams_parameters",string)

    return string

  def _base_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Base file:{self._extract_object.base_ompc}")
    if not final_string:
      final_string = self._extract_object.base_file

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Remove uncalled kernels
    uncalled_kernels = self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]+["check_gpu_mem"]
    for kernels in uncalled_kernels:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)
      final_string = self._cfregex.SUB_CALL_RE(kernels).sub("",final_string)
      final_string = self._cfregex.CAPTURE_PROC_DEF_RE(kernels).sub("",final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_ompc_object")
    final_string = final_string.replace("base_gpu","base_ompc")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_ompc")
    final_string = sub(compile(r"(copy_)gpu(_cpu)",I),r"copy_to_field",final_string)
    final_string = sub(compile(r"(copy_cpu_)gpu",I),r"copy_from_field",final_string)
    final_string = final_string.replace("_gpu","_ompc")

    # Add modules
    final_string = self._add_modules(final_string,"")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    #final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _subroutine
    final_string = final_string.replace("_cuf","_subroutine")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.base_ompc,handle_line_break(final_string))

  def _equation_gpu_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation GPU file:{self._extract_object.equation_gpu_ompc}")
    if not final_string:
      final_string = self._extract_object.equation_gpu_file

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Remove uncalled kernels
    uncalled_kernels = self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]+["check_gpu_mem"]
    for kernels in uncalled_kernels:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)
      final_string = self._cfregex.SUB_CALL_RE(kernels).sub("",final_string)
      final_string = self._cfregex.CAPTURE_PROC_DEF_RE(kernels).sub("",final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_ompc_object")
    final_string = final_string.replace("base_gpu","base_ompc")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_ompc")
    final_string = sub(compile(r"(copy_)gpu(_cpu)",I),r"copy_to_field",final_string)
    final_string = sub(compile(r"(copy_cpu_)gpu",I),r"copy_from_field",final_string)
    final_string = final_string.replace("_gpu","_ompc")

    # Add modules
    final_string = self._add_modules(final_string,"")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    #final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp-eqn")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _subroutine
    final_string = final_string.replace("_cuf","_subroutine")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.equation_gpu_ompc,handle_line_break(final_string))

  def _kernel_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Kernel file:{self._extract_object.kernel_ompc}")
    if not final_string:
      final_string = self._extract_object.kernel_file

    # Array declarations
    final_string = self._array_declaration(final_string)

    # Remove uncalled kernels
    uncalled_kernels = self._extract_object.backend_dict["additional_info"]["uncalled_kernels"]+["check_gpu_mem"]
    for kernels in uncalled_kernels:
      final_string = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",final_string)
      final_string = self._cfregex.SUB_CALL_RE(kernels).sub("",final_string)
      final_string = self._cfregex.CAPTURE_PROC_DEF_RE(kernels).sub("",final_string)

    # Renaming
    final_string = final_string.replace("_gpu_object","_ompc_object")
    final_string = final_string.replace("base_gpu","base_ompc")
    final_string = final_string.replace("streams_kernels_gpu","streams_kernels_ompc")
    final_string = sub(compile(r"(copy_)gpu(_cpu)",I),r"copy_to_field",final_string)
    final_string = sub(compile(r"(copy_cpu_)gpu",I),r"copy_from_field",final_string)
    final_string = final_string.replace("_gpu","_ompc")

    # Add modules
    final_string = self._add_modules(final_string,"")

    # Correct streams parameters
    final_string = self._handle_streams_parameters(final_string)

    # Add allocations
    #final_string = self._treat_allocation(final_string,self._extract_object.gpu_array_obj.gpu_dict,backend="omp")

    # Replace subroutines from input file
    final_string = self._replace_subroutines(final_string,backend_config["replace_subroutines"])

    # Replace any remaining _cuf to _subroutine
    final_string = final_string.replace("_cuf","_subroutine")

    # Remove chevron/stream create/dim3/device sync
    final_string = self._cfregex.STREAM_INT_RE.sub(r"integer\1",final_string)
    final_string = self._cfregex.STREAM_CREATE_RE.sub(r"",final_string)
    final_string = self._cfregex.DEVICE_SYNC_RE.sub("",final_string)
    final_string = self._cfregex.GET_EXP_STREAM_RE.sub("",final_string)
    final_string = self._cfregex.DIM_RE.sub("",final_string)
    final_string = final_string.replace("0_cuda_stream_kind","0")

    write_output(self._extract_object.kernel_ompc,handle_line_break(final_string))

  def _equation_file(self,final_string=None,backend_config=None):
    print(f"\nWriting Equation file:{self._extract_object.equation_ompc}")
    if not final_string:
      final_string = self._extract_object.equation_file

    write_output(self._extract_object.equation_ompc,handle_line_break(final_string))

  def generatefiles(self):
    logging.info("Generating files for OMP-CPU backend")

    print("\nPreparing kernels for OMP-CPU backend")
    self._prepare_kernels.generate_kernels()
    self._log_prepare_dict(self._prepare_kernels.prepare_dict)

    call_subroutine_dict = {"base_file":self._extract_object.base_file,"equation_file":self._extract_object.equation_gpu_file,"kernel_file":self._extract_object.kernel_file}
    print(f"\nReplacing subroutines and calls")
    iterator = tqdm(self._prepare_kernels.prepare_dict.items())
    for kernel,value in iterator:
      iterator.set_postfix(batch=kernel)
      logging.info(f"Replacing subroutines for:{kernel}")

      try:
        subroutine = value["subroutine_handling"]
        for file,calls in subroutine.items():
          call_subroutine_dict[file] = call_subroutine_dict[file].replace(calls[0],calls[1])
      except:
        logging.warning(f"Kernel subroutine does not exist for:{kernel}")

    self._base_file(backend_config=extract_backend_config(self._backend_config,"base_file","ompc"),final_string=call_subroutine_dict["base_file"])
    self._equation_file()
    self._equation_gpu_file(final_string=call_subroutine_dict["equation_file"],backend_config=extract_backend_config(self._backend_config,"equation_gpu_file","ompc"))
    self._kernel_file(final_string=call_subroutine_dict["kernel_file"],backend_config=extract_backend_config(self._backend_config,"kernels_file","ompc"))
    if self._input_args.indent:
      print("\nIndenting output code")
      self._indent.indent(self._extract_object.output_code)
      #self._assoc_clean.assoclean(self._extract_object.output_code)