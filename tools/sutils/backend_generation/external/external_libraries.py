import re
import logging
from tqdm import tqdm
import os

# local imports
from backend_generation.backend_tools import extract_backend_config
from tools import merge_lines_with_ampersand

# Regex flags
RE_FLAGS = re.IGNORECASE

CURRENT_PATH = os.path.realpath(os.path.dirname(__file__))

def remove_external(inp_files : dict, mode: list, backend_dict: dict, backend: str) -> dict:
  """Removes all the plugins from the source code

  Args:
      inp_files (dict): input files that are required to be processed
      mode (list): plugins that should be removed

  Returns:
      dict: files with the plugins with the same order as the input
  """
  op_files = list(inp_files.values())
  file_names = list(inp_files.keys())
  inp_files = list(inp_files.values())
  # Loop over the modes
  for m in mode:
    logging.info(f"Working with mode: {m}")
    print(f"\nRemoving content for : {m}")

    # Loop over the files
    iterator = tqdm(enumerate(inp_files),total=len(inp_files))
    var_list = []
    for idx,cf in iterator:
      iterator.set_postfix(batch=m)
      op_files[idx],var_list = remove_plugins(m,op_files[idx],var_list)
      backend_file = extract_backend_config(backend_dict,file_names[idx],backend)
      if backend_file:
        for idx,subroutines in enumerate(backend_file["replace_subroutines"]):
          subroutines = "".join(merge_lines_with_ampersand(subroutines.splitlines(True)))
          backend_file["replace_subroutines"][idx],_ = remove_plugins(m,subroutines,var_list)
  
  return op_files
  
def remove_plugins(mode: str, output_str: str, var_list: list) -> tuple:
  """Removes a particular plugin

  Args:
      mode (str): plugin mode
      output_str (str): output string
      var_list (list): list of variables to be removed across files

  Returns:
      tuple: output string and variable list
  """
  input_list = output_str.splitlines(True)
  logging.info(f"Extracting vars and removing")
  all_vars = get_all_vars(mode,input_list)
  var_list += all_vars
  
  output_str = CAPTURE_VAR_RE("|".join(var_list)).sub("",output_str) if var_list else output_str
  output_str = LOOK_FOR_VAR_RE(mode).sub("",output_str)
  output_str = ASSOC_1("|".join(var_list)).sub("",output_str) if var_list else output_str
  output_str = ASSOC_2("|".join(var_list)).sub(")\n",output_str) if var_list else output_str 
  
  logging.info(f"Extracting procs and removing")
  all_procs = get_all_procs(mode,input_list)
  for p in all_procs:
    # Remove subroutines and functions
    output_str = CAPTURE_PROC_RE(p).sub("",output_str)
    # Remove procedure definitions
    output_str = CAPTURE_PROC_DEF_RE(p).sub("",output_str)
  output_str = LOOK_FOR_PROC_RE(mode).sub("",output_str)

  logging.info(f"Extracting ifs and removing")
  if_list = get_all_if(input_list,mode)
  for ift in if_list:
    new_ift = keep_else(ift)
    output_str = output_str.replace(ift,new_ift)

  #---------------------------------------------------#
  # From here specific changes
  #---------------------------------------------------#
  logging.warning(f"Specific changes that cannot be automated")
  output_str = CHECK_GPU_RE(mode).sub("",output_str)
  output_str = PRINT_RE(f"{mode}|{mode.upper()}").sub("",output_str)
  if mode == "ibm":
    output_str = re.sub(r".*use.*cgal.*\n","",output_str)
  elif mode == "insitu":
    output_str = re.sub(r".*use.*(catalyst|tcp).*\n","",output_str)
    # output_str = TIME_IS_RE.sub("",output_str)
  #---------------------------------------------------#
    
  return output_str,var_list

#----------------------------------------------------------------------------#
# Local Regex
#----------------------------------------------------------------------------#
# Get variables and procedures 
VAR_PROC_RE = re.compile(r"\:\:(.*)",RE_FLAGS)
# Capture any type of procedure
CAPTURE_PROC_RE = lambda string: re.compile(r".*?(subroutine|function).*"+string+r"(.|\n)*?end\s*(subroutine|function)\s*"+string+r".*\n",RE_FLAGS) 
# Capture procedure declaration
CAPTURE_PROC_DEF_RE = lambda string: re.compile(r"procedure.*\:\:.*("+string+r"[0-9a-z\_]*).*\n",RE_FLAGS)
# Capture a general procedure
CAPTURE_ALL_PROC_RE = re.compile(r"(.*(subroutine|function).*(.|\n)*?end.*?(subroutine|function).*)",RE_FLAGS)
# Capture variable related Regex
CAPTURE_VAR_RE = lambda string: re.compile(r".*(integer|real|logical|character|c\_ptr).*("+string+r").*\n",RE_FLAGS)
# Capture If without "then"
CAPTURE_IF_1_RE = lambda string: re.compile(r"if.*"+string+r".*((?!then).)*",RE_FLAGS)
# Replace associates
ASSOC_1 = lambda string: re.compile(r"("+string+")\s*\=\>.*?("+string+")\s*\,",RE_FLAGS)
ASSOC_2 = lambda string: re.compile(r"\,\s*("+string+")\s*\=\>.*?("+string+").*?\)",RE_FLAGS)
# Remove check gpu routines (not important)
CHECK_GPU_RE = lambda string: re.compile(r".*call.*check\_gpu.*"+string+r".*\n",RE_FLAGS)
PRINT_RE = lambda string: re.compile(r"!.*("+string+r").*",RE_FLAGS)
OTHER_VAR_RE = lambda string: re.compile(r".*("+string+r").*\=.*\n",RE_FLAGS)
# Look for time_is_freezed_fun
TIME_IS_RE = re.compile(r".*time_is_freezed_fun().*\n",RE_FLAGS)
# Look for variables
LOOK_FOR_VAR_RE = lambda string: re.compile(r".*"+string+r"_var_start.*((.|\n)*)"+string+r"_var_end.*\n",RE_FLAGS)
# Look for procedures
LOOK_FOR_PROC_RE = lambda string: re.compile(r".*"+string+r"_proc_start.*((.|\n)*)"+string+r"_proc_end.*\n",RE_FLAGS)
CAPTURE_VARS_RE = re.compile(r".*(integer|real|logical|character|c\_ptr).*\:\:.*\n",RE_FLAGS)

#----------------------------------------------------------------------------#
# Local functions
#----------------------------------------------------------------------------#
def get_all_vars(mode,input_lines):
  start_var = 0
  end_var = 0
  for idx,f in enumerate(input_lines):
    if f"{mode}_var_start" in f:
      start_var = idx + 1
    if f"{mode}_var_end" in f:
      end_var = idx

  get_vars = input_lines[start_var:end_var]
  if get_vars == []:
    return []
  all_vars = []

  for gv in get_vars:
    if gv.strip():
      match_var =  VAR_PROC_RE.search(gv).group(1).replace(" ","").split(",")
      for mv in match_var:
        mv = mv.split("=")[0].strip()
        all_vars.append(mv)
  return all_vars

def get_all_procs(mode,input_lines):
  start_proc = 0
  end_proc = 0
  for idx,f in enumerate(input_lines):
    if f"{mode}_proc_start" in f:
      start_proc = idx + 1
    if f"{mode}_proc_end" in f:
      end_proc = idx

  get_procs = input_lines[start_proc:end_proc]
  if get_procs == []:
    return []
  all_procs = []

  for gv in get_procs:
    if gv.strip():
      match_var =  VAR_PROC_RE.search(gv).group(1).replace(" ","").split(",")
      for mv in match_var:
        mv = mv.split("=")[0].strip()
        all_procs.append(mv)
  return all_procs

def get_all_if(filelines: list, mode: str) -> list:
  enable_mode = f"enable_{mode}"
  final_list = []
  for idx,line in enumerate(filelines):
    #if "if" in line and any(word in line for word in enable_mode):
    if "if" in line and enable_mode in line:
      # Case where there is no "then"
      if "then" not in line:
        final_list.append(line)  
        continue
      # Case for full if condition
      if "then" in line:
        if_string = line
        endif_counter = 1
        num_counter = idx
        while endif_counter != 0:
          num_counter += 1
          if "then" in filelines[num_counter]:
            endif_counter += 1
          elif "endif" in filelines[num_counter]:
            endif_counter -= 1

          if_string += filelines[num_counter]
        
        final_list.append((if_string)) 
  return final_list

def keep_else(if_input: str) -> str:
  if_level = 0
  all_lines = if_input.split("\n")
  all_lines = [x for x in all_lines if x.strip() != ""]
  else_start_line = len(all_lines)
  for idx,line in enumerate(all_lines):
    if "if" in line and "then" not in line:
      return ""
    if "if" in line and "then" in line:
      if_level += 1
    if "endif" in line:
      if_level -= 1
    if if_level == 1 and "else" in line:
      else_start_line = idx
      break
  return "\n".join(all_lines[else_start_line+1:-1])+"\n"
#----------------------------------------------------------------------------#
