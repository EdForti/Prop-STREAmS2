import shutil
import logging
import os
import sys
from abc import ABC, abstractmethod
from mako.lookup import TemplateLookup
import tomllib

from tools import check_dir_status,write_output,write_json

from backend_generation.backend_tools import read_code
from backend_generation.gpu_arrays import GpuArrays
from backend_generation.get_kernels import KernelExtraction
from backend_generation.regular_expressions.regex import CudaFRegularExpressions
from backend_generation.external.external_libraries import remove_external
from indent.indent import StreamsIndent

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
BACKEND_PATH = os.path.realpath(os.path.dirname(__file__))

class InitBackend(ABC):
  """Main abstract class to set all the required parameters to choose a specific backend.
  Contains four implementations:
    1. InitBackendAmd
    2. InitBackendCpu
    3. InitBackendOmp
    4. InitBackendOmpc
    4. InitBackendOriginal

  In the current configuration, four main files are read from the source code:
    1. base_file
    2. equation_file
    3. equation_gpu_file
    4. kernel_file
  This naming convention should be strictly followed throughout this code as it is
  also related to how we access the input TOML file
  """
  def __init__(self,args):
    self._args = args
    self._equation_type = self._args.equation
    logging.info(f"Equation type: {self._equation_type}")
    self._backend_config = self._backend_input_config()
    self._backend_type = ""
    self._input_code = self._args.input_dir
    self._input_code_path = f"{self._input_code}/src"
    self._input_equation_path = f"{self._input_code}/src/{self._equation_type}"
    self._output_code = ""
    self._output_code_path = ""
    self._output_equation_path = ""
    self._gpu_array_obj = GpuArrays()
    self._kernel_extract_obj = KernelExtraction()
    self._template_path = TemplateLookup(directories=BACKEND_PATH)
    self._base_file = ""
    self._equation_file= ""
    self._equation_gpu_file = ""
    self._kernel_file = ""
    self._backend_dict = {}
    self._indent = StreamsIndent()
    self._cfregex = CudaFRegularExpressions()
    
  @property
  def output_code(self):
    return self._output_code

  @property
  def output_code_path(self):
    return self._output_code_path

  @property
  def output_equation_path(self):
    return self._output_equation_path

  @property
  def equation_type(self):
    return self._equation_type

  @property
  def base_file(self):
    return self._base_file
  
  @base_file.setter
  def base_file(self,value):
    self._base_file=value
  
  @property
  def equation_file(self):
    return self._equation_file
  
  @equation_file.setter
  def equation_file(self,value):
    self._equation_file=value

  @property
  def equation_gpu_file(self):
    return self._equation_gpu_file
  
  @equation_gpu_file.setter
  def equation_gpu_file(self,value):
    self._equation_gpu_file=value

  @property
  def kernel_file(self):
    return self._kernel_file
  
  @kernel_file.setter
  def kernel_file(self,value):
    self._kernel_file=value
  
  @property
  def backend_dict(self):
    return self._backend_dict
  
  @property
  def gpu_array_obj(self):
    return self._gpu_array_obj
  
  @property
  def backend_config(self):
    return self._backend_config
  
  def _backend_input_config(self):
    print(f"Reading the backend config file: {self._args.backend_config}")
    logging.info(f"Reading the backend config file")
    
    with open(self._args.backend_config, "rb") as f:
      config = tomllib.load(f)
    return config
      
  def _input_management(self):
    self._input_code = os.path.abspath(self._args.input_dir)
    # Input directory, code directory and equation directory must all exist
    check_dir_status(self._input_code,1)
    check_dir_status(self._input_code_path,1)
    check_dir_status(self._input_equation_path,1)
    self.output_dir = os.path.abspath(self._args.output_dir)
    logging.info(f"Reading input code from: {self._input_code} and output will be written in: {self.output_dir}")
    
  def _remove_plugins(self):
    if self._args.remove_plugins:
      plugins_to_remove = self._args.remove_plugins
      logging.info(f"Removing the plugins: {plugins_to_remove}")
      removed_files = remove_external({"equation_file":self._equation_file,"equation_gpu_file":self._equation_gpu_file,"kernel_file":self._kernel_file},plugins_to_remove, self._backend_config, self._backend_type)
      
      self._equation_file= removed_files[0]
      self._equation_gpu_file = removed_files[1]
      self._kernel_file = removed_files[2]
      
  def _define_input_files(self,output_code_path,output_equation_path):
    self._base_file = read_code(f"{output_code_path}/base_gpu.F90")
    self._equation_gpu_file = read_code(f"{output_equation_path}/{self._equation_type}_gpu.F90")
    self._equation_file= read_code(f"{output_equation_path}/{self._equation_type}.F90",required=False)
    self._kernel_file = read_code(f"{output_equation_path}/kernels_gpu.F90")
    
  def _extract_gpu_arrays(self):
    logging.info("Backend using GPU array object")
    self._gpu_array_obj.files_str = self._base_file+self._equation_gpu_file
    self._gpu_array_obj.generate_dict()
    self._gpu_array_obj.generate_macros()
    
  def _extract_kernels(self):
    logging.info("Extracting kernel subroutines and calls")
    self._kernel_extract_obj.files_str = {"base_file":self._base_file,"equation_file":self._equation_gpu_file,"kernel_file":self._kernel_file}
    self._kernel_extract_obj.backend_config = self._backend_config
    self._kernel_extract_obj.backend_type = self._backend_type
    self._kernel_extract_obj.extract_subroutines()
    self._kernel_extract_obj.gpu_obj = self._gpu_array_obj
    self._kernel_extract_obj.extract_kernels()
    self._backend_dict = self._kernel_extract_obj.kernel_dict
    
  def log_backend_dict(self,print_dict):
    print(f"\nWriting log_kernel_dict.json after extracting kernels and calls")    
    write_json("log_kernel_dict.json",print_dict)
    if self._kernel_extract_obj.gpu_obj.gpu_dict:
      print(f"\nWriting log_gpu_dict.json after extracting kernels and calls")
      write_json("log_gpu_dict.json",self._kernel_extract_obj.gpu_obj.gpu_dict)
  
  @abstractmethod 
  def _output_management(self):
    raise NotImplementedError
   
  @abstractmethod 
  def _add_main_program(self):
    raise NotImplementedError
  
  @abstractmethod 
  def _output_files(self):
    raise NotImplementedError
   
  @abstractmethod 
  def _copy_input_files(self):
    raise NotImplementedError
    
class InitBackendAmd(InitBackend):
  def __init__(self,args):
    super().__init__(args)
    self._backend_type = "amd"
    self._input_management() 
    self._output_management()
    self._add_main_program()
    self._copy_input_files()
    logging.info("Setting up input/output files for AMD backend")
    self._define_input_files(self._output_code_path,self._output_equation_path)
    self._output_files()
    # Remove plugins
    self._remove_plugins()
    # Extract GPU arrays
    if self._args.generate_gpu_arrays:
      self._extract_gpu_arrays()
    # Extract subroutines
    self._extract_kernels()
    self.log_backend_dict(self._backend_dict)

  def index_define_lower_case(self):
    target_path = os.path.join(self._output_equation_path, "index_define.h")
    if os.path.exists(target_path):
      with open(target_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.writelines(lines[:-1])
        file.writelines("#define n_s N_S\n")
        file.writelines(line.lower() for line in lines[3:])
        file.truncate()
      print(f"in index_define.h added lowercase indices.")
    else:
      print("index_define.h not found in the expected directory.")

  def fix_include_statements(self):
    file_paths = [f"{self.output_dir}/code_amd/src/{self._equation_type}/{self._equation_type}.F90",
                  f"{self.output_dir}/code_amd/src/{self._equation_type}/{self._equation_type}_gpu.F90",
                  f"{self.output_dir}/code_amd/src/{self._equation_type}/kernels_gpu.F90"]
   
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                lines = file.readlines()
                if lines and "#include 'index_define.h'" in lines[0]:
                    lines[0] = "#include \"index_define.h\"\n"
                file.seek(0)
                file.writelines(lines)
                file.truncate()
            print(f"Fixed include statement in: {file_path}")
        else:
            print(f"File not found: {file_path}")

    
  def _output_management(self):
    self._output_code = f"{self.output_dir}/code_amd/"
    self._output_code_path = f"{self.output_dir}/code_amd/src/"
    self._output_equation_path = f"{self.output_dir}/code_amd/src/{self._equation_type}"
    check_dir_status(self._output_code,0)
    shutil.copytree(self._input_code,self._output_code,dirs_exist_ok=True)
    self.index_define_lower_case()
    self.fix_include_statements()
    
  def _add_main_program(self):
    TEMPLATE_FILE = self._template_path.get_template("templates/main.F90")
    main_program = f"main_amd.F90"
    main_file = TEMPLATE_FILE.render(equation_type=self._equation_type,backend="amd").replace('\r\n','\n')
    write_output(f"{self._output_equation_path}/{main_program}",main_file)
    
  def _copy_input_files(self):
    files_to_copy = [(f"{BACKEND_PATH}/input_files/amd/hip_utils.h",self._output_code_path)]
    
    for source,dest in files_to_copy:
      shutil.copy2(source,dest)
      
  def _output_files(self):
    self._base_amd = f"{self._output_code_path}/base_amd.F90"
    self._base_amd_cpp = f"{self._output_code_path}/base_amd_cpp.cpp"
    self._equation_gpu_amd = f"{self._output_equation_path}/{self._equation_type}_amd.F90"
    self._equation_amd = f"{self._output_equation_path}/{self._equation_type}.F90"
    self._kernel_amd = f"{self._output_equation_path}/kernels_amd.F90"
    self._kernel_amd_cpp = f"{self._output_equation_path}/kernels_amd_cpp.cpp"
    self._gpu_array_file = f"{self._output_code_path}/amd_arrays.h"

  @property
  def base_amd(self):
    return self._base_amd   

  @property
  def base_amd_cpp(self):
    return self._base_amd_cpp  
  
  @property
  def equation_amd(self):
    return self._equation_amd   

  @property
  def equation_gpu_amd(self):
    return self._equation_gpu_amd     

  @property
  def kernel_amd(self):
    return self._kernel_amd  

  @property
  def kernel_amd_cpp(self):
    return self._kernel_amd_cpp

  @property
  def gpu_array_file(self):
    return self._gpu_array_file
    
class InitBackendCpu(InitBackend):
  def __init__(self,args):
    super().__init__(args)
    self._backend_type = "cpu"
    self._input_management() 
    self._output_management()
    self._add_main_program()
    self._copy_input_files()
    logging.info("Setting up input/output files for CPU backend")
    self._define_input_files(self._output_code_path,self._output_equation_path)
    self._output_files()
    # Remove plugins
    self._remove_plugins()
    # Extract GPU arrays
    if self._args.generate_gpu_arrays:
      self._extract_gpu_arrays()
    # Extract subroutines
    self._extract_kernels()
    self.log_backend_dict(self._backend_dict)

  def index_define_lower_case(self):
    target_path = os.path.join(self._output_equation_path, "index_define.h")
    if os.path.exists(target_path):
      with open(target_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.writelines(lines[:-1])
        file.writelines("#define n_s N_S\n")
        file.writelines(line.lower() for line in lines[3:])
        file.truncate()
      print(f"in index_define.h added lowercase indices.")
    else:
      print("index_define.h not found in the expected directory.")

  def fix_include_statements(self):
    file_paths = [f"{self.output_dir}/code_cpu/src/{self._equation_type}/{self._equation_type}.F90",
                  f"{self.output_dir}/code_cpu/src/{self._equation_type}/{self._equation_type}_gpu.F90",
                  f"{self.output_dir}/code_cpu/src/{self._equation_type}/kernels_gpu.F90"]
   
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r+') as file:
                lines = file.readlines()
                if lines and "#include 'index_define.h'" in lines[0]:
                    lines[0] = "#include \"index_define.h\"\n"
                file.seek(0)
                file.writelines(lines)
                file.truncate()
            print(f"Fixed include statement in: {file_path}")
        else:
            print(f"File not found: {file_path}")
    
  def _output_management(self):
    self._output_code = f"{self.output_dir}/code_cpu/"
    self._output_code_path = f"{self.output_dir}/code_cpu/src/"
    self._output_equation_path = f"{self.output_dir}/code_cpu/src/{self._equation_type}"
    check_dir_status(self._output_code,0)
    shutil.copytree(self._input_code,self._output_code,dirs_exist_ok=True)
    self.index_define_lower_case()
    self.fix_include_statements()
    
  def _add_main_program(self):
    TEMPLATE_FILE = self._template_path.get_template("templates/main.F90")
    main_program = f"main_cpu.F90"
    main_file = TEMPLATE_FILE.render(equation_type=self._equation_type,backend="cpu").replace('\r\n','\n')
    write_output(f"{self._output_equation_path}/{main_program}",main_file)
    
  def _copy_input_files(self):
    pass
  
  def _output_files(self):
    self._base_cpu = f"{self._output_code_path}/base_cpu.F90"
    self._equation_gpu_cpu = f"{self._output_equation_path}/{self._equation_type}_cpu.F90"
    self._kernel_cpu = f"{self._output_equation_path}/kernels_cpu.F90"
    self._equation_cpu = f"{self._output_equation_path}/{self._equation_type}.F90"
   
  @property
  def base_cpu(self):
    return self._base_cpu    

  @property
  def equation_gpu_cpu(self):
    return self._equation_gpu_cpu    
  
  @property
  def equation_cpu(self):
    return self._equation_cpu   

  @property
  def kernel_cpu(self):
    return self._kernel_cpu  

  @property
  def gpu_array_file(self):
    return self._gpu_array_file
    
class InitBackendOmp(InitBackend):
  def __init__(self,args):
    super().__init__(args)
    self._backend_type = "omp"
    self._input_management() 
    self._output_management()
    self._add_main_program()
    self._copy_input_files()
    logging.info("Setting up input/output files for OMP backend")
    self._define_input_files(self._output_code_path,self._output_equation_path)
    self._output_files()
    # Remove plugins
    self._remove_plugins()
    # Extract GPU arrays
    if self._args.generate_gpu_arrays:
      self._extract_gpu_arrays()
    # Extract subroutines
    self._extract_kernels()
    self.log_backend_dict(self._backend_dict)
    
  def _output_management(self):
    self._output_code = f"{self.output_dir}/code_omp/"
    self._output_code_path = f"{self.output_dir}/code_omp/src/"
    self._output_equation_path = f"{self.output_dir}/code_omp/src/{self._equation_type}"
    check_dir_status(self._output_code,0)
    shutil.copytree(self._input_code,self._output_code,dirs_exist_ok=True)
    
  def _add_main_program(self):
    TEMPLATE_FILE = self._template_path.get_template("templates/main.F90")
    main_program = f"main_omp.F90"
    main_file = TEMPLATE_FILE.render(equation_type=self._equation_type,backend="omp").replace('\r\n','\n')
    write_output(f"{self._output_equation_path}/{main_program}",main_file)
    
  def _copy_input_files(self):
    files_to_copy = [(f"{BACKEND_PATH}/input_files/omp/utils_omp.F90",self._output_code_path)]
    
    for source,dest in files_to_copy:
      shutil.copy2(source,dest)
  
  def _output_files(self):
    self._base_omp = f"{self._output_code_path}/base_omp.F90"
    self._equation_gpu_omp = f"{self._output_equation_path}/{self._equation_type}_omp.F90"
    self._kernel_omp = f"{self._output_equation_path}/kernels_omp.F90"
    self._equation_omp = f"{self._output_equation_path}/{self._equation_type}.F90"

  @property
  def base_omp(self):
    return self._base_omp 

  @property
  def equation_gpu_omp(self):
    return self._equation_gpu_omp     
  
  @property
  def equation_omp(self):
    return self._equation_omp   

  @property
  def kernel_omp(self):
    return self._kernel_omp  

  @property
  def gpu_array_file(self):
    return self._gpu_array_file
  
class InitBackendOmpc(InitBackend):
  def __init__(self,args):
    super().__init__(args)
    self._backend_type = "ompc"
    self._input_management() 
    self._output_management()
    self._add_main_program()
    self._copy_input_files()
    logging.info("Setting up input/output files for OMP-CPU backend")
    self._define_input_files(self._output_code_path,self._output_equation_path)
    self._output_files()
    # Remove plugins
    self._remove_plugins()
    # Extract GPU arrays
    if self._args.generate_gpu_arrays:
      self._extract_gpu_arrays()
    # Extract subroutines
    self._extract_kernels()
    self.log_backend_dict(self._backend_dict)
    
  def _output_management(self):
    self._output_code = f"{self.output_dir}/code_ompc/"
    self._output_code_path = f"{self.output_dir}/code_ompc/src/"
    self._output_equation_path = f"{self.output_dir}/code_ompc/src/{self._equation_type}"
    check_dir_status(self._output_code,0)
    shutil.copytree(self._input_code,self._output_code,dirs_exist_ok=True)
    
  def _add_main_program(self):
    TEMPLATE_FILE = self._template_path.get_template("templates/main.F90")
    main_program = f"main_ompc.F90"
    main_file = TEMPLATE_FILE.render(equation_type=self._equation_type,backend="ompc").replace('\r\n','\n')
    write_output(f"{self._output_equation_path}/{main_program}",main_file)
    
  def _copy_input_files(self):
    files_to_copy = []
    
    for source,dest in files_to_copy:
      shutil.copy2(source,dest)
  
  def _output_files(self):
    self._base_ompc = f"{self._output_code_path}/base_ompc.F90"
    self._equation_gpu_ompc = f"{self._output_equation_path}/{self._equation_type}_ompc.F90"
    self._kernel_ompc = f"{self._output_equation_path}/kernels_ompc.F90"
    self._equation_ompc = f"{self._output_equation_path}/{self._equation_type}.F90"

  @property
  def base_ompc(self):
    return self._base_ompc 

  @property
  def equation_gpu_ompc(self):
    return self._equation_gpu_ompc   
  
  @property
  def equation_ompc(self):
    return self._equation_ompc  

  @property
  def kernel_ompc(self):
    return self._kernel_ompc  

  @property
  def gpu_array_file(self):
    return self._gpu_array_file

class InitBackendOriginal(InitBackend):
  def __init__(self,args):
    super().__init__(args)
    self._backend_type = "original"
    self._input_management() 
    self._output_management()
    self._define_input_files(self._output_code_path,self._output_equation_path)
    self._output_files()
    self._remove_plugins()
    # Extract subroutines
    self._extract_kernels()
    self.log_backend_dict(self._backend_dict)
    self._write_files()
    if self._args.indent:
      print("\nIndenting output code")
      self._indent.indent(self._output_code)
    
  def _write_files(self):
    print(f"\nWriting Equation file:{self._equation_ext}")
    # Remove uncalled kernels
    for kernels in self._backend_dict["additional_info"]["uncalled_kernels"]:
      self._equation_file= self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",self._equation_file)
    write_output(self._equation_ext,self._equation_file)
    
    print(f"\nWriting Equation GPU file:{self._equation_gpu_ext}")
    # Remove uncalled kernels
    for kernels in self._backend_dict["additional_info"]["uncalled_kernels"]:
      self._equation_gpu_file = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",self._equation_gpu_file)
    write_output(self._equation_gpu_ext,self._equation_gpu_file)
    
    print(f"\nWriting Kernel file:{self._kernel_ext}")
    # Remove uncalled kernels
    for kernels in self._backend_dict["additional_info"]["uncalled_kernels"]:
      self._kernel_file = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",self._kernel_file)
    write_output(self._kernel_ext,self._kernel_file)
    
    print(f"\nWriting Base GPU file:{self._base_gpu_ext}")
    # Remove uncalled kernels
    for kernels in self._backend_dict["additional_info"]["uncalled_kernels"]:
      self._base_file = self._cfregex.GET_SUBROUTINE_FUNCTION_RE(kernels).sub("",self._base_file)
    write_output(self._base_gpu_ext,self._base_file)
    
  def _output_management(self):
    self._output_code = f"{self.output_dir}/code_ext/"
    self._output_code_path = f"{self.output_dir}/code_ext/src/"
    self._output_equation_path = f"{self.output_dir}/code_ext/src/{self._equation_type}"
    check_dir_status(self._output_code,0)
    shutil.copytree(self._input_code,self._output_code,dirs_exist_ok=True)
    
  def _output_files(self):
    self._equation_gpu_ext = f"{self._output_equation_path}/{self._equation_type}_gpu.F90"
    self._equation_ext = f"{self._output_equation_path}/{self._equation_type}.F90"
    self._kernel_ext = f"{self._output_equation_path}/kernels_gpu.F90"
    self._base_gpu_ext = f"{self._output_code_path}/base_gpu.F90"
    
  def _add_main_program(self):
    pass
   
  def _copy_input_files(self):
    pass
