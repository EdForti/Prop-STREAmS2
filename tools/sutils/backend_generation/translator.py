import logging
from re import I,compile,search,match,sub,escape
from backend_generation.regular_expressions.regex import FortranRegularExpressions
from tools import format_string,clean_string

class FortranToCpp:
  def __init__(self,serial_code,var_info,array_dict,gpu_array_map,is_reduction=False):
    self._serial_code = serial_code
    self._var_info = var_info
    self._array_dict = array_dict
    self._gpu_array_map = gpu_array_map
    self._translated_code = ""
    self._fregex = FortranRegularExpressions()
    self._is_reduction = is_reduction
    self._translate()
    
  @property
  def translated_code(self):
    return self._translated_code

  def _replace_pow(self,inp):
    def find_var_or_expr(s, reverse=False):
      # search for variable or expression starting in the given string
      # reverse=True means read the string reversed
      if reverse:
        s = s[::-1]
      st = s.replace(" ","")
      if st.startswith("-"):
        raise Exception("Pow converter not working for unary minus exponents")
      if st.startswith("(") or st.startswith(")") or st.startswith("[") or st.startswith("]"):
        level = 0
        for ik,k in enumerate(st):
          if k == "(" or k == "[":
            if reverse:
              level -= 1
            else:
              level += 1
          if k == ")" or k == "]":
            if reverse:
              level += 1
            else:
              level -= 1
          if level == 0:
            array_match = search("^[a-zA-Z0-9\._]+",st[ik+1:])
            var_array = ""
            if array_match:
              var_array = array_match.group(0)
            if reverse:
              v = (st[:ik+1]+var_array)[::-1]
            else:
              v = st[:ik+1]+var_array
            return v
        return v # if expr ends at the end of line
      else:
        v = ""
        for ik,k in enumerate(st):
          if match("[a-zA-Z0-9\._]+",k): 
            v += k
          else:
            if reverse:
              v = v[::-1]
            return v
        return v # if var ends at the end of line
  
    # replaces ** Fortran style exponentiation with pow function call
    # (pow may be poorly efficient but it is another story)
    # algorithm: 
    # (a) search for **
    # (b) find left and right operands (variables or expressions)
    # (c) replace operands and ** with pow syntax
    # (d) iterates over (a-c) until no ** is found
    complete = False
    counter = 0
    max_counter = 1000
    while complete == False and counter < max_counter:
      inp = inp.replace(" ","")
      counter += 1
      complete = True
      last_two = "  "
      for ic,c in enumerate(inp):
        last_two = last_two[1]+c
        if last_two == "**":
          left = inp[0:ic-1]
          right = inp[ic+1:]
          #print(f"found ** {inp}")
          #print(f"found ** {left}")
          #print(f"found ** {right}")
          vr = find_var_or_expr(right)
          vl = find_var_or_expr(left, reverse=True)
          #print(f"{vl}")
          #print(f"{vr}")
          
          if vr.strip() == "2":
            inp_new = f"{left[0:len(left)-len(vl)]} (({vl})*({vl})) {right[len(vr):]}"
          elif type(vr.strip()) == int:
            raise Exception(f"Add pow opertaion for a^{vr.strip()}")
          else:
            inp_new = f"{left[0:len(left)-len(vl)]} pow({vl},{vr}) {right[len(vr):]}"
          #inp_new = f"{left[0:len(left)-len(vl)]} pow({vl},{vr}) {right[len(vr):]}"
          inp = inp_new
          complete = False
          break
    if counter == max_counter:
      logging.error("Failed exponent conversion! Exiting...")
      raise Exception("Error in exponent conversion!, check log files")
    return inp

  def _match_every_line(self,input_string):
    line_list = input_string.split("\n")
    case_counter = 0
    for idx,line in enumerate(line_list):
      if line_list[idx].strip() == "":
        line_list[idx] = line_list[idx].strip().replace("\n","")
        continue
      # Check for call....()
      call_reg = compile(r"call(\s*\()",I)
      if call_reg.search(line_list[idx]):
        line_list[idx] = sub(call_reg,r"\1",line)
        if ":" in line:
          line_list[idx] = line_list[idx].replace(":","1")
          
      # check for temp variables
      tmp_regex = compile(r"(sutils_tmp_(real|int)_\w+)\=(.*?)(\,|\))",I)
      
      # Capture constant args passed through device kernels
      tmp_line_str = tmp_regex.sub(r"\1\4",line_list[idx])
      if tmp_regex.findall(line_list[idx]): 
        final_val_str = ""
        for val in tmp_regex.findall(line_list[idx]):
          val_name = val[0]
          val_type = val[1]
          val_value = val[2]
          val_str = f"{val_type} {val_name} = {val_value}"
          final_val_str += f"{val_str};\n"
        line_list[idx] = f"{final_val_str}{tmp_line_str}\n"
        
      # Check for enddo/endif/else/elseif 
      end_else_if_do_check = line_list[idx].replace(" ","").strip()
      if end_else_if_do_check == "enddo" or end_else_if_do_check == "endif":
        line_list[idx] = "}"
      elif end_else_if_do_check == "else":
        line_list[idx] = "}else {"
      # Check if then....
      if_then_reg = compile(r"(if.*?\(.*?\).*?)then",I)
      
      if if_then_reg.search(line_list[idx]) and "else" not in line_list[idx]:
        line_list[idx] = sub(if_then_reg,r"\1{",line_list[idx])

      else_if_then_reg = compile(r"else\s*if(.*?\(.*?\).*?)then",I)
      if else_if_then_reg.search(line_list[idx]):
        line_list[idx] = sub(else_if_then_reg,r"}else if\1{",line_list[idx])
      # Check for stop statement  
      if "stop" in line_list[idx].strip():
        line_list[idx] = ""
      # write statement
      if "write(" in line_list[idx].strip().replace(" ",""):
        line_list[idx] = 'printf("Error!")'
      # Replace exit with break
      if line.strip().endswith("exit"):
        line_list[idx] = line_list[idx][:-4]
        line_list[idx] += " break"
      # Add SIGN function
      if "sign(" in line_list[idx]:
        line_list[idx] = line_list[idx].replace("sign(","SIGN(")
      # Replace D-40 -> E-40
      if self._fregex.FLOATING_CONST_RE.search(line_list[idx]):
        line_list[idx] = self._fregex.FLOATING_CONST_RE.sub(r"\1E\2",line_list[idx])
      # Check case
      if line_list[idx].startswith("case "):
        if case_counter == 0:
          line_list[idx] = line_list[idx]
          case_counter += 1
        else:
          line_list[idx] = "break;\n" + line_list[idx]
      # Finally do the exponential operator
      if "**" in line_list[idx]:
        line_list[idx] = self._replace_pow(line_list[idx])
      if "}" in line_list[idx] or "{" in line_list[idx] or ":" in line_list[idx]:
        line_list[idx] = line_list[idx]+"\n"
        continue
      else:
        line_list[idx] = line_list[idx]+";\n"

    return "".join(line_list)

  def _symbol_translate(self,txt):
    dictionary = {"_rkind": "0", "!":"//",".and.":"&&",".or.":"||","/=":"!=",".gt.":">",".lt.":"<",".ge.":">=",".le.":"<=",".eq.":"=="}
    match_regex = compile("|".join(map(escape, dictionary)))
    return match_regex.sub(lambda match: dictionary[match.group(0)], txt)

  def _handle_case(self,string):
    string = self._fregex.SELECT_CASE_RE.sub(r"int bc_case = \1\nswitch (bc_case) {",string)
    string = self._fregex.CASE_RE.sub(r"case \1:;",string)
    string = self._fregex.END_CASE_RE.sub("break;\n}",string)

    return string

  def _get_size_for_array(self,length): return r",".join(["[/:a-z0-9+*\(\)\s*_-]*?"]*length)
  
  def _format_gpu_arrays(self,string,gpu_arrays,gpuvar_dict,gpu_array_map):
    for ga in gpu_arrays:
      size_of_array = len(gpuvar_dict[gpu_array_map[ga]][3].split(",")) 
      get_size = self._get_size_for_array(size_of_array)
      match = self._fregex.ARRAY_MAP_RE(ga,get_size).findall(string)
      for m in match:
        m1 = m[0]
        m2 = m[1]
        if ":" in m2:
          search = compile(escape(m1)+r"\s*\("+escape(m2)+r"\)",I)
          replace = r"&("+m1+r"["+gpuvar_dict[gpu_array_map[m1]][0]+r"("+m2+r")])"
          string = sub(search,replace,string)
        else:
          search = compile(escape(m1)+r"\s*\("+escape(m2)+r"\)",I)
          replace = m1+r"["+gpuvar_dict[gpu_array_map[m1]][0]+r"("+m2+r")]"
          string = sub(search,replace,string)
    return clean_string(string)

  def _format_local_arrays(self,string,local_arrays):
    def initialise_array(string,name):
      bracket_match = self._fregex.LOCAL_ARRAY_INIT_RE(name).search(string).group(1)
      split_bracket = bracket_match.split(",")
      new_string = ""
      for i,val in enumerate(split_bracket):
        new_string += f"{name}({i+1}) = {val}\n"
      string = self._fregex.LOCAL_ARRAY_INIT_RE(name).sub(new_string,string)
      return string

    for la in local_arrays:
      name = la[0]
      size = len(la[1].split(","))
      # Check if the array is initialised
      if self._fregex.LOCAL_ARRAY_INIT_RE(name).search(string):
        string = initialise_array(string,name)
      # Now replace () with [__I...()]
      search = self._get_size_for_array(size)
      string = self._fregex.ARRAY_MAP_RE(name,search).sub(name+r"[__LI_"+name.upper()+r"(\2)]",string)
    return string

  def _translate(self):
    # Strip all the whitespaces
    logging.info("Formatting the input string to remove whitespaces, etc.,")
    self._translated_code = format_string(self._serial_code)
    
    logging.info("Extracting GPU arrays and formatting them")
    gpu_arrays = self._var_info["real_arrays"]+self._var_info["int_arrays"]
    if self._is_reduction:
      logging.warning("Manually adding redn_3d_gpu")
      gpu_arrays += ["redn_3d_gpu"]
    # Manually add gpu array and mapping before using this function
    self._translated_code = self._format_gpu_arrays(self._translated_code,gpu_arrays,self._array_dict,self._gpu_array_map)
  
    logging.info("Replacing explicit Fortran symbols with C")
    self._translated_code = self._symbol_translate(self._translated_code)
    
    logging.info("Replacing Do to for in C")
    self._translated_code = self._fregex.DO_RE.sub(r"for(int \1=\2; \1<\3+1; \1++){",self._translated_code)
    
    logging.info("Extracting local arrays and formatting them")
    self._translated_code = self._format_local_arrays(self._translated_code,self._var_info["lreal_arrays"]+self._var_info["linteger_arrays"])
    
    logging.info("Matching case situations with switch")
    self._translated_code = self._handle_case(self._translated_code)
    
    if self._fregex.IF_TAG_RE.search(self._translated_code):
      tag_name = self._fregex.IF_TAG_RE.search(self._translated_code).group(1)
      logging.warning(f"Removing if tag named: {tag_name}")
      self._translated_code = self._fregex.IF_TAG_RE.sub(r"\2",self._translated_code)
      
    # Remove real(x,y) and replace with x
    self._translated_code = self._fregex.CAPTURE_REAL_RE.sub(r"\1",self._translated_code) 
      
    logging.info("Looping through individual lines and making changes")
    self._translated_code = self._match_every_line(self._translated_code)
