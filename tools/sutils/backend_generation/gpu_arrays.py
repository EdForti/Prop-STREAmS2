from backend_generation.regular_expressions.regex import FortranRegularExpressions
from tqdm import tqdm
import logging

class GpuArrays:
  def __init__(self):
    self._files_str = ""
    self._fregex = FortranRegularExpressions()
    self._gpu_dict = {}
    self._gpu_array_file = ""
    self._alloc_list = []
  
  @property
  def files_str(self):
    return self._files_str
  
  @files_str.setter
  def files_str(self,value):
    self._files_str=value
  
  @property  
  def gpu_array_file(self):
    return self._gpu_array_file
  
  @property  
  def gpu_dict(self):
    return self._gpu_dict
  
  def _get_array_names(self,files_str):
    return self._fregex.ALLOC_FOR_VAR_RE.findall(files_str)
  
  def generate_dict(self):
    logging.info(f"Using the merged source files to extract GPU arrays")
    print(f"\nExtracting GPU arrays")
    merged_files = self._files_str
    
    all_gpu_arrays = self._get_array_names(merged_files)
    
    array_names = []
    original_name = []
    
    for i in all_gpu_arrays:
      i = i[1]
      original_name += self._fregex.ALLOC_FULL_NAME_RE.findall(i)
      i = i.split("!")[0].replace(" ","")
      if i.lower().count('_gpu') == 1:
        array_names += self._fregex.ALLOC_VAR_ATTRIB_SINGLE_RE.findall(i)
      elif i.lower().count('_gpu')>1:
        array_names += self._fregex.ALLOC_VAR_ATTRIB_MULT_RE.findall(i)

    iterator = tqdm(array_names)
    for ijk,vn in enumerate(iterator):
      array_name = vn[0]
      array_size = vn[1]
      iterator.set_postfix(batch=array_name)
      logging.info(f"Working with array: {array_name}")
      """
      -> Most probably a bug here
      -> we are trying to eliminate self%... from array dimensions and retain the size
      """
      # Retain the original format to be used for allocate redefinition
      original_size = array_size
      original_size = original_size.split(",")
      
      for idx,ov in enumerate(original_size):
        if ":" not in ov:
          original_size[idx] = f"1:{ov}"
      original_size = ",".join(original_size)
      
      # Continue with other extractions  
      if "%" in array_size:
        array_size = self._fregex.DIM_FORMAT_RE.sub("",array_size)
      array_size = array_size.split(",")  
      for idx,ars in enumerate(array_size):
        ars_remove = ars.split("%")[-1]
        #  Add 1: if it doesnt exit
        if ":" not in ars_remove:
          ars_remove = f"1:{ars_remove}"
        array_size[idx] = ars_remove.split("%")[-1]
        
      try:
        array_type = self._fregex.ARRAY_TYPE_DIM_RE(array_name).search(merged_files).group(1)
        array_dim = self._fregex.ARRAY_TYPE_DIM_RE(array_name).search(merged_files).group(2).replace(" ","")
      except:
        logging.error(f"{array_name} not found")
        raise Exception("Error! Check log file")

      if array_type == "logical":
        array_type = "bool"
      elif array_type == "integer":
        array_type = "int"
      
      idx_convert_name = f"__I{len(array_size)}_{array_name[:-4].upper()}"
      self._gpu_dict[array_name] = [idx_convert_name,array_dim,array_type,",".join(array_size),original_size,original_name[ijk].strip()]
    iterator.close()
    # Add reduction variable here
    logging.warning(f"Adding reduction gpu array to the dictionary")
    self._gpu_dict["redn_3d_gpu"] = ["__I3_REDN_3D","dimension(:,:,:)","real","1:nx,1:ny,1:nz","1:nx,1:ny,1:nz","self%redn_3d_gpu"]

  def generate_macros(self):
    print("\nGenerating Macros")
    indices = ["i","j","k","m","l"] 
    prepoc_begin = "#define "
    self._gpu_array_file = "//Warning: Automatically generated from the converter. Could alter the code if changes are made\n\n" 
    logging.info(f"Sorting dict according to array size")
    sorted_dict = dict(sorted(self._gpu_dict.items(),key=lambda e: e[1][0]))
    logging.info(f"Looping through each GPU array and evaluating the conversion")
    iterator = tqdm(sorted_dict.items())
    for key,value in iterator:
      array_id_name = value[0]
      iterator.set_postfix(batch=key)
      array_size = value[3].split(",")
      array_len = len(array_size)
      first_value = []
      stride = []
      logging.info(f"Working with array: {array_id_name}")
      for idx,var in enumerate(array_size):
        # Subtract indices from first index
        index_value = indices[idx]
        array_index_first_value = array_size[idx].split((":"))[0]
        array_index_second_value = array_size[idx].split((":"))[1]
        first_value.append(f"({index_value})-({array_index_first_value})")
        stride.append(f"({array_index_second_value})-({array_index_first_value})+1")
        
      if array_len == 1:
        convert_index = f"({first_value[0]})"
      elif array_len == 2:
        convert_index = f"(({first_value[0]})+({stride[0]})*({first_value[1]}))"
      elif array_len == 3:
        convert_index = f"(({first_value[0]})+({stride[0]})*({first_value[1]})+({stride[0]})*({stride[1]})*({first_value[2]}))"
      elif array_len == 4:
        convert_index = f"(({first_value[0]})+({stride[0]})*({first_value[1]})+({stride[0]})*({stride[1]})*({first_value[2]})+({stride[0]})*({stride[1]})*({stride[2]})*({first_value[3]}))"
      elif array_len == 5:
        convert_index = f"(({first_value[0]})+({stride[0]})*({first_value[1]})+({stride[0]})*({stride[1]})*({first_value[2]})+({stride[0]})*({stride[1]})*({stride[2]})*({first_value[3]})+({stride[0]})*({stride[1]})*({stride[2]})*({stride[3]})*({first_value[4]}))"
      else:
        logging.error("Array size more than 4 not implemented!")
        raise Exception("Array size more than 4 not implemented! Check log")
      
      to_string = f"{prepoc_begin}{array_id_name}({','.join(indices[:array_len])}) {convert_index}\n"
      
      self._gpu_array_file += to_string
    iterator.close()   
