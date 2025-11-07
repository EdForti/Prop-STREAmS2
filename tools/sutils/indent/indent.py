import re
import shutil
import os
from tools import check_dir_status,get_files_with_extension,read_file,write_output,handle_line_break

class StreamsIndent:
  def __init__(self):
    pass
    
  def _input_management(self,input_code):
    check_dir_status(input_code,1)
    return input_code

  def indent(self,code_folder,step_indent=2):
    code_folder = self._input_management(code_folder)
    try:
      from colorama import Fore, Back, Style
      info_color  = Fore.CYAN
      warn_color  = Fore.RED
      err_color   = Fore.MAGENTA
      reset_color = Style.RESET_ALL
    except:
      info_color  = ""
      warn_color  = ""
      err_color   = ""
      reset_color  = ""
      
    # rules to increase "more" indent or remove "less" indent
    # three types of match:
    # [1] startswith match
    # [2] exact match
    # [3] regexp match
    more_list = ["attributes(global) ", \
                 "subroutine ", "attributes(device) subroutine", \
                 "function", "attributes(device) function", \
                 "type ", "contains", "interface", \
                 "do "]
    more_list_exact = ["do", "else"]
    more_list_regexp = ["^module(?!( procedure ))", \
                        "^if.*then", "^elseif.*then", "^else if.*then", \
                        "^if *\(.*\&", "^select *case *\(.*\)", "^ *case *\(.*\)", \
                        "^associate *\("]
    
    less_list = ["endmodule ", "end module ", \
                 "endsubroutine ", "end subroutine ", \
                 "endfunction ", "end function", \
                 "endtype ", "end type ", "contains", "end interface", "endinterface",  \
                 "enddo ", "end do ", "endif", "end if", "else if", "elseif", \
                 "endassociate", "end associate", "endselect", "end select"]
    less_list_exact = ["enddo", "else","end select", "endselect", "end do", \
                       "endsubroutine", "end subroutine"]
    less_list_regexp = ["^ *case *\(.*\)"]
  
    finps = get_files_with_extension(f"{code_folder}/src/","F90")
    cpp_files = get_files_with_extension(f"{code_folder}/src/","cpp")

    exclude_comment_indent = []
    
    for finp in finps:
      print(info_color + "-"*72+f"\nProcessing file {finp}..." + reset_color)
      fi = read_file(finp,clean_str=True) 
      
      indent = 0
      i_line = 0
      call_started = False
      for idx,line in enumerate(fi):
        i_line += 1
        ls = line.strip()

        # to decide indentation use ls_d variable where we remove from line:
        # (a) labels of "do" , "if" , "select"
        # (b) starting exclamation mark not corresponding to Cuda directives (normal comment line) 
        if re.search("\w+: do", ls) is not None or \
          re.search("\w+: if", ls) is not None or \
          re.search("\w+: select", ls) is not None:
          ls_d = ":".join(ls.split(":")[1:]).strip()
          #print(ls_d)
        elif ls.startswith("!") and not ls.startswith("!@") and not ls.startswith("!$"):
          if not finp in exclude_comment_indent:
            ls_d = ls[1:].strip()
          else:
            ls_d = ls
        else:
          ls_d = ls

        # decrease indentation
        if any(map(ls_d.startswith, less_list)) or \
          any([ls_d==m for m in less_list_exact]) or \
          any(re.search(r,ls_d) for r in less_list_regexp):
          indent -= step_indent

        # prepare indented line ls_i (first column ! are still in the first column)
        if ls == "":
          #ls_i = "!"
          ls_i = ""
        elif ls.startswith("#"):
          ls_i = ls
        elif ls.startswith("!") and not ls.startswith("!@") and not ls.startswith("!$"):
          #ls_i = "!"+" "*(indent-1)+ls[1:].strip()
          ls_i = " "*(indent)+"!"+ls[1:].strip()
        else:
          ls_i = " "*indent+ls

        # write indented line
        fi[idx] = f"{ls_i}\n"

        # increase indentation
        if any(map(ls_d.startswith, more_list)) or \
          any([ls_d==m for m in more_list_exact]) or \
          any(re.search(r,ls_d) for r in more_list_regexp):
          indent += step_indent

        # manage multiline instructions, in particular "call " and "write "
        # peculiar because it is indented but does not open an indented region 
        # (like subroutine <name> of if ... then for example)
        if any(map(ls_d.startswith, ["call ","write"])) and ls_d[-1] == "&":
          call_started = True
          indent += step_indent
        if call_started and len(ls_d) > 0:
          if ls_d[-1] != "&":
            indent -= step_indent
            call_started = False
      
#       if indent != 0:
#         print(err_color + f"\nWarning! Indent at the end of file is #{indent} instead of zero. Try to disable comment indenting" + reset_color)

      print(f"writing {finp}")
      write_output(finp,handle_line_break("".join(fi)))
    for cppf in cpp_files:
      print(info_color + "-"*72+f"\nProcessing file {cppf}..." + reset_color)
      try:
        os.system(f"clang-format -i {cppf}")
      except:
        raise Exception ("clang-format not found, run pip3 -r install_requirements.txt from sutils/")
        