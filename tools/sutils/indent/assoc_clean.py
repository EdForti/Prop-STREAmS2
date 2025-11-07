import re
import shutil
from tools import check_dir_status,get_files_with_extension,read_file,write_output,handle_line_break

class StreamsAssoclean:
  def __init__(self):
    pass
    
  def _input_management(self,input_code):
    check_dir_status(input_code,1)
    return input_code

  def assoclean(self,code_folder):
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
      
    finps = get_files_with_extension(f"{code_folder}/src/","F90")

    for finp in finps:
      print(info_color + "-"*72+f"\nProcessing file {finp}..." + reset_color)
      fi = read_file(finp,clean_str=True) 
      
      indent = 0
      i_line = 0
      call_started = False
      for idx,line in enumerate(fi):
        i_line += 1
        ls = line.strip()
        
        if re.search(" *end *associate", ls) is not None:
            #print("line with end associate: ",ls)
            pass
        elif re.search("^ *associate *\((.*)\)", ls) is not None:
            #print("line with associate: ",ls)
            ls_vars = re.search(" *associate *\((.*)\)", ls).group(1)
            assoc_dict = {}
            for k_v in ls_vars.split(","):
                k = re.split("=>",k_v)[0].strip()
                v = re.split("=>",k_v)[1].strip()
                assoc_dict[k] = v

            assoc_dict_sca = {}
            assoc_dict_obj = {}
            for k,v in assoc_dict.items():
                obj, sep, item = v.rpartition("%")
                if obj == "":
                    assoc_dict_sca[k] = v
                elif obj in assoc_dict_obj.keys():
                    assoc_dict_obj[obj].append([k,item])
                else:
                    assoc_dict_obj[obj] = [ [k,item] ]
            assoc_dict_clean = dict(scalars=assoc_dict_sca, objects=assoc_dict_obj)

            new_line = "associate( &\n"

            new_assoc_list = []
            for a,b in assoc_dict_clean["scalars"].items() :
                new_assoc_list.append(a + " => " + b)
            if len(new_assoc_list) > 0:
                new_line += "  "
                new_line += ", ".join(new_assoc_list)
                new_line += " ;&\n"

            for obj,v in assoc_dict_clean["objects"].items() :
                new_line += "  from "+obj+" : "
                new_assoc_list = []
                for x in v:
                    a = x[0]
                    b = x[1]
                    if a == b:
                        new_assoc_list.append(a)
                    else:
                        new_assoc_list.append(a + " => " + b)
                new_line += ", ".join(new_assoc_list)+" ;&\n"

            new_line = new_line[0:len(new_line)-3] + "&\n)\n"

            #print(new_line)
            fi[idx] = f"{new_line}\n"
        elif "associated" in ls:
            pass
        elif "associate" in ls and ls.startswith("!"):
            print("associate in comment is ignored")
            pass
        elif "associate" in ls:
            print("associate not as first element of line. Error!")
            print("line with error: ",ls)
            raise
      print(f"writing {finp}")
      write_output(finp,handle_line_break("".join(fi)))

  def assoclean_reverse(self,code_folder):
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
      
    finps = get_files_with_extension(f"{code_folder}/src/","F90")

    for finp in finps:
      print(info_color + "-"*72+f"\nProcessing file {finp}..." + reset_color)
      fi = read_file(finp,clean_str=True) 
      
      indent = 0
      i_line = 0
      call_started = False
      for idx,line in enumerate(fi):
        i_line += 1
        ls = line.strip()
        
        if re.search(" *end *associate", ls) is not None:
            #print("line with end associate: ",ls)
            pass
        elif re.search("^ *associate *\((.*)\)", ls) is not None:
            #print("line with associate: ",ls)
            ls_vars = re.search(" *associate *\((.*)\)", ls).group(1)

            assoc_dict = {}
            for assoc in ls_vars.split(";"):
                if assoc.strip().startswith("from"):
                    assoc_obj = assoc.strip().split(":")
                    assoc_from     = assoc_obj[0].strip()[5:]
                    assoc_obj_list = assoc_obj[1]
                    for item in assoc_obj_list.split(","):
                        if "=>" in item:
                            k = re.split("=>",item)[0].strip()
                            v = re.split("=>",item)[1].strip()
                            assoc_dict[k] = assoc_from+"%"+v
                        else:
                            k = item.strip()
                            v = k
                            assoc_dict[k] = assoc_from+"%"+v
                    
                else:
                    for item in assoc.split(","):
                        k = re.split("=>",item)[0].strip()
                        v = re.split("=>",item)[1].strip()
                        assoc_dict[k] = v
            #print(assoc_dict)

            assoc_list = []
            for a,b in assoc_dict.items():
                assoc_list.append(a + " => " + b)
            new_line = "associate( &\n"
            new_line += ", ".join(assoc_list)
            new_line += "&\n)\n"

            #print(new_line)
            fi[idx] = f"{new_line}\n"
        elif "associated" in ls:
            pass
        elif "associate" in ls and ls.startswith("!"):
            print("associate in comment is ignored")
            pass
        elif "associate" in ls:
            print("associate not as first element of line. Error!")
            print("line with error: ",ls)
            raise

      print(f"writing {finp}")
      write_output(finp,handle_line_break("".join(fi)))