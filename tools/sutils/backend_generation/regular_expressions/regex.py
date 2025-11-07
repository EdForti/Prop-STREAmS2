from dataclasses import dataclass
from re import I,M,compile,escape
from typing import Pattern

# Class for general Fortran match
@dataclass(frozen=True)
class FortranRegularExpressions:
  ALLOC_FOR_VAR_RE: Pattern = compile(r"(ALLOCATE\s*\((.*)\))",I)
  ALLOC_VAR_ATTRIB_SINGLE_RE: Pattern = compile(r"(\w+_GPU)\s*\((.*?)\)$",I)
  ALLOC_VAR_ATTRIB_MULT_RE: Pattern = compile(r"(\w+_GPU)\s*\((.*?)\)",I)
  DIM_FORMAT_RE: Pattern = compile(r"self.*?%([a-z_]*[^%][,:+$%-]*)",I)
  ARRAY_TYPE_DIM_RE: Pattern = lambda string: compile(r"[ ]*?(real|integer|logical).*(dimension\s*\(.*?\)).*?(?:(?!(real|integer|logical)).|\n)*?"+string,I)
  CAPTURE_COMMENT_RE: Pattern = compile(r"\!([^\n)].*|\n)",I)
  GET_SUBROUTINE_FUNCTION_RE: Pattern = lambda name: compile(r"(.*(subroutine|function)\s*"+name+r"(.*)\s*\((.|\n)*?end\s*(subroutine|function).*"+name+r"\n)",I)
  ALL_SUBROUTINES_RE = GET_SUBROUTINE_FUNCTION_RE("")
  SUBROUTINE_CALL_RE: Pattern = lambda kernel: compile(r"(.*call\s*"+kernel+r"\s*\(((.|\n)*?)\)\s*\n)",I)
  SUBROUTINE_NAME_RE: Pattern = compile(r"(subroutine|function)\s*(.*)\s*\(",I)
  CAPTURE_SUBROUTINE_ARGS_RE: Pattern = compile(r"(.*(subroutine|function).*\((.*)\).*\n)",I) # Tests to add!
  CAPTURE_VARS_RE: Pattern = compile(r".*\:\:.*\n",I) # Tests to add!
  CAPTURE_INNER_SUB_RE: Pattern = compile(r".*(subroutine|function).*(.|\n)*\:\:.*((.|\n)*)end\s*(subroutine|function).*",I) # Tests to add!
  REAL_EXP_RE: Pattern = compile(r"real.*?\:\:\s*(.*)") # Tests to add!
  INT_EXP_RE: Pattern = compile(r"integer.*?\:\:\s*(.*)",I) # Tests to add!
  LOG_EXP_RE: Pattern = compile(r"logical.*?\:\:\s*(.*)",I) # Tests to add!
  VARS_AND_TYPE_RE: Pattern = compile(r"(.*)\:\:(.*)",I)  # Tests to add!
  CHECK_DIMENSION_ARRAY_RE: Pattern = compile(r"dimension\((.*?)\)",I) # Tests to add!
  DO_RE: Pattern = compile(r".*?do.*?(\w+)\s*\=\s*([a-z0-9(+*/\-)_]*)\s*\,\s*([a-z0-9(+*/\-)_]*).*",I) # Tests to add! 
  DO_2_RE: Pattern = lambda idx,sz1,sz2: compile(r".*?do.*?"+idx+"\s*\=\s*"+sz1+"\s*\,\s*"+sz2+".*",I) # Tests to add!
  PARAMETER_CHECK: Pattern = compile(r"\,\s*parameter",I)
  SELECT_CASE_RE: Pattern = compile(r".*select.*?\((.*)\)",I) # Tests to add!
  CASE_RE: Pattern = compile(r"case\s*\((.*?)\)",I) # Tests to add!
  END_CASE_RE: Pattern = compile(r".*end\s*select.*",I) # Tests to add!
  ARRAY_MAP_RE: Pattern = lambda string,size: compile(r"("+string+r")\s*\(("+size+r")\)",I) # Tests to add!
  LOCAL_ARRAY_INIT_RE: Pattern = lambda string: compile(string+r"\s*\=\s*\[(.*?)\]",I) # Tests to add!
  IF_TAG_RE: Pattern = compile(r"([a-z0-9_]*)\s*\:\s*(if.*?then(.|\n)*?end\s*if)\s*\1",I) # Tests to add!
  DEVICE_KERNEL_CALL_RE: Pattern = lambda kernel: compile(kernel+r"(\s*\()",I)
  POINTER_CHECK_RE: Pattern = lambda var: compile(r"(?:^|\s)"+var+r"\s*\=\s*",I)
  ALLOC_FULL_NAME_RE: Pattern = compile(r"(self.*?_gpu)\s*\(",I)
  SEARCH_ALLOCATE_RE: Pattern = lambda string: compile(r"allocate.*self%"+string+r".*",I)
  FLOATING_CONST_RE: Pattern = compile(r"([0-9\.]*?)D([\+\-0-9])")
  CAPTURE_ONLY_WORD_RE: Pattern = compile(r"\b[^\d\W]+\b",I)
  CAPTURE_REAL_RE: Pattern = compile(r"real\((.*?)\,.*?\)",I)
  SUB_CALL_RE: Pattern = lambda string: compile(r"(.*call.*"+string+r".*\(((.|\n)*?)\)\s*\n)",I)
  CAPTURE_PROC_DEF_RE: Pattern = lambda string: compile(r"procedure.*\:\:.*("+string+r"[0-9a-z\_]*).*\n",I)
  
# Class for CUDA Fortran specific match
@dataclass(frozen=True)
class CudaFRegularExpressions(FortranRegularExpressions):
  CAPTURE_COMMENT_RE: Pattern = compile(r"\!([^($|@|\n)].*|\n)",I)
  EXPLICIT_KERNEL_CAPTURE_RE: Pattern = lambda type: compile(r"attributes\s*\("+type+r"\)\s*(?:launch_bounds\(.*?\))?\s*(subroutine|function)(.|\n)*?end\s*(subroutine|function).*",I) # Tests to add!
  SUBROUTINE_CALL_RE: Pattern = lambda kernel: compile(r"((.*call|\=)\s*"+kernel+r"\s*(?:\<(.*)\>)?\s*\(((.|\n)*?)\)\s*\n)",I) # Tests to add!
  GPU_ARRAY_NAME_RE: Pattern = compile(r"\w+\_gpu",I) # Tests to add!
  GET_EXP_STREAM_RE: Pattern = compile(r"\<\<\<(.*?)\>\>\>",I) # Tests to add!
  #EXPLICIT_THREAD_BOUNDS_RE: Pattern = compile(r"([a-z0-9_]*)\s*\=\s*blockDim.*\n.*?([a-z0-9_]*)\s*\=\s*blockDim.*(.|\n)*?if\s*\(\s*\1\s*[><=]*\s*([a-z0-9_%]*).*\2\s*[><=]*\s*([a-z0-9_%]*)\s*\).*\n",I) # Tests to add!
  GLOBAL_KERNEL_IDX_RE: Pattern = compile(r"([a-z0-9_]*)\s*\=\s*blockDim.*\n",I) # Tests to add!
  GLOBAL_KERNEL_BOUND_RE: Pattern = lambda idx:compile(r"if.*\(.*"+idx+r"\s*[><]\s*([a-z_0-9]*).*\).*return",I) 
  CUF_DO_RE: Pattern = compile(r"do\((\d)\)",) # Tests to add!
  REDUCE_RE: Pattern = compile(r"reduce\((.*?)\:(.*?)\)",I) # Tests to add!
  CUF_REMOVE_RE: Pattern = compile(r".*\!\$cuf.*\n|.*\!\@cuf.*\n",I) # Tests to add!
  GET_CUF_STREAM_RE: Pattern = compile(r"stream\s*\=\s*(.*?)\s*\>",I) # Tests to add!
  NON_CUF_EXTRACT_RE: Pattern = compile(r"([0-9a-z_]*?)\s*\=",I) # Tests to add!
  REDN_RE: Pattern = lambda rvar: compile(r"("+rvar+r")\s*\=\s*\1\s*([+\-*])\s*(.*)",I) # Tests to add!
  REDN_MAX_RE: Pattern = lambda rvar: compile(r".*("+rvar+r")\s*\=\s*max\s*\(\s*\1\s*\,\s*([a-z0-9+*\\\-]*)\s*\s*\).*",I) # Tests to add!
  REDN_MIN_RE: Pattern = lambda rvar: compile(r".*("+rvar+r")\s*\=\s*min\s*\(\s*\1\s*\,\s*([a-z0-9+*\\\-]*)\s*\s*\).*",I) # Tests to add!
  REDN_REMOVE_OTHER_RE: Pattern = compile(r".*?(max|min).*",I) # Tests to add!
  CPU_GPU_RE: Pattern = lambda var: compile(r"((^|\s)"+var+r"\s*[a-z0-9\:\,\(\)]*\s*\=\s*[^\>].*)",I|M) # Tests to add!
  GPU_CPU_RE: Pattern = lambda var: compile(r"(([a-z0-9\_\%]*\s*|[a-z0-9\_\%]*\([a-z0-9\:\,]*\)\s*)\=\s*"+var+r".*)",I) # Tests to add!
  ALLOCATABLE_GPU_RE: Pattern = compile(r"(.*)\,\s*allocatable(.*?\_(gpu|managed))",I)
  DEVICE_FLAG_RE: Pattern = compile(r"(\,\s*)(device|managed)",I)
  ALL_GLOBAL_DEVICE_RE: Pattern = compile(r".*?attributes\s*\((.|\n)*?end\s*(subroutine|function).*?\n",I)
  CUDAFOR_RE: Pattern = compile(r"(.*)use CUDAFOR",I)
  STREAM_INT_RE: Pattern = compile(r"integer.*?kind.*?cuda_stream_kind\s*\)(.*)",I)
  STREAM_CREATE_RE: Pattern = compile(r".*cudaStreamCreate\s*\((.*?)\)",I)
  DEVICE_SYNC_RE: Pattern = compile(r"!@cuf iercuda=cudaDeviceSynchronize\(\)",I)
  THREAD_ID_RE: Pattern = lambda idx: compile(r"(blockDim\%"+idx+r"\*\(blockIdx\%"+idx+r"-1\)\+threadIdx\%"+idx+r")(.*)",I)
  DIM_RE: Pattern = compile(r".*dim3.*",I)