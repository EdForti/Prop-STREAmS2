from tools import read_file,format_string
from backend_generation.regular_expressions.regex import CudaFRegularExpressions
from tqdm import tqdm
import re
from re import I, compile,escape,sub

def read_code(file: str,required=True) -> tuple:
  file_lst = read_file(file)
  reg = CudaFRegularExpressions().CAPTURE_COMMENT_RE
  # Handle comments
  for idx,fl in enumerate(file_lst):
    if CudaFRegularExpressions.FLOATING_CONST_RE.search(fl):
      file_lst[idx] = fl
    elif required:
      fl = fl.lower()
      file_lst[idx] = fl
    if fl.strip().startswith("!"):
      if "_var_start" in fl or "_var_end" in fl or "_proc_start" in fl or "_proc_end" in fl:
        continue
      else:
        file_lst[idx] = reg.sub("",fl)
        if file_lst[idx].strip() == "":
          file_lst[idx] = ""
    elif re.search(r"(\'|\").*?\!.*?(\'|\")",fl):
      continue
    elif "!" in fl.strip():
      file_lst[idx] = reg.sub("",fl)
    
  file_str = "".join(file_lst)
  
  file_str = format_string(file_str)
  
  return file_str

def get_dict(dict: dict) -> tuple: return dict["subroutine_info"],dict["kernel_info"],dict["var_info"],dict["reduction_info"]

def iterate_nested_dict(my_dict):
  for kernel_type,kernel_name in my_dict.items():
    for kernel,value in kernel_name.items():
      yield kernel_type,kernel,value
      
def get_tqdm_iterator(kernel_dict):
  iterator = list(iterate_nested_dict(kernel_dict))
  iterator_len = len(iterator)
  iterator = tqdm(iterator,total=iterator_len)
    
  return iterator
  
def get_device_kernel_names(kernel_dict):
  iterator = list(iterate_nested_dict(kernel_dict))
  device_kernels=[j for i,j,k in iterator if i=="device_kernels"]
  
  return device_kernels
      
def split_by_comma(txt):
  split_txt = txt.split(",")
  final_list = []
  counter = 0
  temp_str = ""
  for st in split_txt:
    st = st.strip()
    if counter == 0 and "(" not in st and ")" not in st:
      final_list.append(st)
      continue

    if counter == 0 and "(" in st and ")" in st:
      final_list.append(st)
      continue
    
    if "(" in st and ")" not in st and counter == 0:
      temp_str += st
      counter = 1
      continue

    if counter == 1:
      temp_str += ","+st
      if ")" in st:
        counter = 0
        final_list.append(temp_str)
        temp_str = ""

  return final_list
  
def extract_backend_config(backend_config,config_key,backend):

  if config_key in backend_config:
    if (backend == "cpu" or backend == "ompc") and backend in backend_config[config_key]:
      return backend_config[config_key][backend]
    elif (backend == "cpu" or backend == "ompc") and "omp" in backend_config[config_key]:
      return backend_config[config_key]["omp"]
    elif backend in backend_config[config_key]:
      return backend_config[config_key][backend]
    else:
      return None
  else:
    return None

def get_non_repeating_elements(input_list):
  final_list = []
  for il in input_list:
    if il not in final_list and input_list.count(il) > 1:
      final_list.append(il)
    elif input_list.count(il) == 1:
      final_list.append(il)

  return ",".join(final_list) if len(final_list)>1 else "".join(final_list)

def update_list_order(a, new_a, b):
    updated_b = [b[a.index(value)] for value in new_a]
    return updated_b