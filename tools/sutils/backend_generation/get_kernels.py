from backend_generation.backend_tools import get_dict,split_by_comma,get_tqdm_iterator,get_device_kernel_names,extract_backend_config
from backend_generation.regular_expressions.regex import CudaFRegularExpressions
from backend_generation.kernel_probe import ProbeCuf,ProbeGlobal,ProbeDevice
                                     
import logging
from tqdm import tqdm

class KernelExtraction():
  #Class attributes
  def __init__(self):
    logging.info("Merging input files")
    self._files_str = {}
    self._kernel_dict = {}
    self._backend_config = None
    self._backend_type = ""
    self._cfregex = CudaFRegularExpressions()
    self._gpu_obj = None
    
  @property
  def gpu_obj(self):
    return self._gpu_obj
  
  @gpu_obj.setter
  def gpu_obj(self,value):
    self._gpu_obj=value 
    
  @property
  def files_str(self):
    return self._files_str
  
  @files_str.setter
  def files_str(self,value):
    self._files_str=value
    
  @property
  def kernel_dict(self):
    return self._kernel_dict
  
  @property
  def backend_config(self):
    return self._backend_config
  
  @backend_config.setter
  def backend_config(self,value):
    self._backend_config = value
    
  @property
  def backend_type(self):
    return self._backend_type
  
  @backend_type.setter
  def backend_type(self,value):
    self._backend_type = value
    
  def extract_subroutines(self):
    # Extract all subroutines
    logging.info("Extracting all the subroutines")
    print("\nExtracting subroutines")
    # Create sub dictionaries for kernel types
    self._kernel_dict["cuf_kernels"] = {}
    self._kernel_dict["global_kernels"] = {}
    self._kernel_dict["device_kernels"] = {}
    self._kernel_dict["additional_info"] = {}
    self._kernel_dict["additional_info"]["uncalled_kernels"] = []
    iterator = tqdm(self._files_str.items())
    for file_src,file_str in iterator:
      all_subroutines = self._cfregex.ALL_SUBROUTINES_RE.findall(file_str)
      for st in all_subroutines:
        logging.info(f"Setting up subroutine:{st[2]}")
        subroutine = st[0]
        subroutine_name = st[2].strip()
        iterator.set_postfix(batch=subroutine_name)
        if self._cfregex.EXPLICIT_KERNEL_CAPTURE_RE("device").search(subroutine):
          kernel_type = self._kernel_dict["device_kernels"]
        elif self._cfregex.EXPLICIT_KERNEL_CAPTURE_RE("global").search(subroutine):
          kernel_type = self._kernel_dict["global_kernels"]
        elif "_cuf" in subroutine_name:
          kernel_type = self._kernel_dict["cuf_kernels"]
        else:
          continue
          
        kernel_type[subroutine_name] = {}
        kernel_type[subroutine_name]["subroutine_info"] = {}
        kernel_type[subroutine_name]["subroutine_info"]["full_subroutine"] = subroutine
        kernel_type[subroutine_name]["subroutine_info"]["source"] = file_src
        kernel_type[subroutine_name]["kernel_info"] = {}
        kernel_type[subroutine_name]["reduction_info"] = {}
        kernel_type[subroutine_name]["kernel_info"]["debug_mode"] = False
        kernel_type[subroutine_name]["var_info"] = {}
    iterator.close()
 
  def extract_kernels(self):
    logging.info("Extracting kernels from the subroutines")
    print("\nExtracting kernels from subroutines")
    iterator = get_tqdm_iterator(self._kernel_dict)
    device_kernels = get_device_kernel_names(self._kernel_dict)
    for kernel_type,kernel_name,value in iterator:
      if kernel_type == "additional_info": continue
      iterator.set_postfix(batch=kernel_name)
      subroutine_info,_,_,_ = get_dict(value)
      logging.info(f"Extracting kernel subroutine calls for: {kernel_name}")
      subroutine_info["subroutine_call"],subroutine_info["sub_call_arguments"],subroutine_info["call_sources"],kernel_status,is_device_func = self._extract_kernel_subroutine_call(kernel_name)
      if kernel_status:
        logging.info(f"Kernel call not found, removing kernel from dictionary for : {kernel_name}")
        self._kernel_dict["additional_info"]["uncalled_kernels"].append(kernel_name)
        del self._kernel_dict[kernel_type][kernel_name]
        continue
      
      if kernel_type == "cuf_kernels":
        ProbeCuf(kernel_name,self._gpu_obj,self._kernel_dict[kernel_type][kernel_name],extract_backend_config(self._backend_config,kernel_name,self._backend_type),device_kernels=device_kernels)
      elif kernel_type == "global_kernels":
        ProbeGlobal(kernel_name,self._gpu_obj,self._kernel_dict[kernel_type][kernel_name],extract_backend_config(self._backend_config,kernel_name,self._backend_type),device_kernels=device_kernels)
      elif kernel_type == "device_kernels":
        ProbeDevice(kernel_name,self._gpu_obj,self._kernel_dict[kernel_type][kernel_name],extract_backend_config(self._backend_config,kernel_name,self._backend_type),is_device_func)
    
    iterator.close() 
      
  def _extract_kernel_subroutine_call(self,kernel_name):

    kernel_call_sources = []
    final_call_arguments = []
    final_call = []
    device_function=False
    for file_src,file_str in self._files_str.items():
      call = self._cfregex.SUBROUTINE_CALL_RE(kernel_name).findall(file_str)
      if not call:
        continue
      call_arguments = [i[3] for i in call]
  
      call = [i[0] for i in call]
      # Check if its a device function
      for idx,cl in enumerate(call):
        if cl[0]=="=":
          device_function = True
          call[idx] = cl[1:]
        else:
          continue
      
      for i,ca in enumerate(call_arguments):
        ca = ca.replace("&","").replace("\n","").replace(" ","").lower()
        call_arguments[i] = split_by_comma(ca)
        kernel_call_sources.append(file_src)
      final_call += call
      final_call_arguments += call_arguments
      
    if not kernel_call_sources:        
      return [],[],[],True,False
    else:
      return final_call,final_call_arguments,kernel_call_sources,False,device_function
    
