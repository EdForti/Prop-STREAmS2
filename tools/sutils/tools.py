import json
import logging
import sys
import os
import pprint
import fnmatch
from re import I, compile,escape,sub
import re

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

def write_output(filename: str,content: str) -> None:
  f = open(filename, "w")
  f.write(content)
  f.close()

def read_file(filename: str,clean_str=False) -> list:
  read_file = open(filename, "r")
  read_line = read_file.readlines()
  read_file.close()
  if clean_str:
    read_line = "".join(read_line)
    read_line = format_string(read_line)
    read_line = read_line.splitlines(True)
  return read_line

def read_json(filename: str) -> dict :
  with open(filename, "r") as f:
    data = json.load(f)
  return data

def write_json(output_file: str,final_dict: dict) -> None:
  with open(output_file, "w") as f:
    json.dump(final_dict, f, indent=2)

def split_string(strg: str, lenn: int) -> list:
  return [strg[i:i+lenn] for i in range(0, len(strg), lenn)]

def split_string_advanced(strg: str, len_max: int, len_min: int) -> list:
  splitted_lines = []
  favourite_separators = [",",";","+","-","*","/"]

  istart  = 0
  while True:
    # print(istart, len(strg))
    if istart >= len(strg): break
    strg_max = strg[istart:istart+len_max]
    if len(strg_max) < len_max:
      splitted_line = strg_max
      splitted_lines.append(splitted_line)
      break
    imax = len(strg_max)
    istart += imax
    for i in range(min(len_max-len_min,len(strg_max))):
        if strg_max[len(strg_max)-1-i] in favourite_separators:
            # print(strg_max[-1-i])
            imax    = len(strg_max)-i
            istart -= i
            break
    splitted_line = strg_max[0:imax]
    splitted_lines.append(splitted_line)
    # print("istart, splitted_line : ",istart, splitted_line)

  return splitted_lines

def handle_line_break(read_line: str,line_length: int=132) -> str:
  # A very crude implementation for EOL in Fortran. Can improve later
  final_data = ""
  for line in read_line.split("\n"):
    if len(line) >= line_length and "!$omp" not in line:
      lspace = (len(line) - len(line.lstrip()))
      #split_lines = split_string(line,100)
      split_lines = split_string_advanced(line,100,80)
      for idx,l in enumerate(split_lines):
        if "!" in split_lines[0]:
          if idx==0:
            final_data += f"{' '*lspace}{l.strip()}\n"
          else:
            final_data += f"!{' '*lspace}{l.strip()}\n"
        else:
          if idx==0:
            final_data += f"{l}&\n"
          elif idx==len(split_lines)-1:
            final_data += f"{' '*lspace}&{l}\n"
          else:
            final_data += f"{' '*lspace}&{l}&\n"
      continue
    final_data += f"{line}\n"

  return final_data

def one_space_string(input_string: str) -> str: return " ".join(input_string.split())

def setup_logging(string: str) -> None:
  FORMAT = "[%(filename)s: %(lineno)s - %(funcName)s()] - %(levelname)s - %(message)s"
  logging.basicConfig(filename=string,encoding="utf-8", level=logging.DEBUG, format=FORMAT, filemode="w")

def split_list(input_list: list,chunk_size: int)-> list: return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]

def check_dir_status(dir: str, status: int):
  """
  status:
  0 -> dir should not exist
  1 -> dir should exist
  """
  if status == 1:
    if not os.path.exists(dir):
      raise Exception(f"{dir} cannot be found!") 
    return True
  if status == 0:
    if os.path.exists(dir):
      raise Exception(f"{dir} must not exist!. Remove it and try again") 
    return True
  
def reverse_list(lst):
   new_lst = lst[::-1]
   return new_lst
 
def get_files_with_extension(directory, extension):
  file_list = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      if fnmatch.fnmatch(file, f'*.{extension}'):
        file_list.append(os.path.join(root, file))
  return file_list

def format_string(string): 
  # if not "#ifdef" in string:
  #   string_lst = string.split("&") 
  # else: 
  #   return string
  # if len(string_lst) == 1:
  #   return clean_string("".join(string_lst))
  # for idx,line in enumerate(string_lst):
  #   if line.split("\n")[0].strip()=="":
  #       string_lst[idx] = "\n".join(line.split("\n")[1:])
  
  # return clean_string("".join(string_lst))
  
  if "#ifdef" in string:
    return string
  else: 
    return clean_string("\n".join(merge_lines_with_ampersand(string.split("\n"))))

def clean_string(string):
  string_lst = [i.replace("\n","").strip() for i in string.split("\n")]
  return sub(" +"," ","\n".join(string_lst))

def merge_lines_with_ampersand(lst):
  merged_list = []
  i = 0
  while i < len(lst):
    if lst[i].strip().endswith("&") and not lst[i].strip().startswith("!$") and not lst[i].strip().startswith("!@"):
      if lst[i].strip().startswith("&"):
        merged_string = lst[i].strip().lstrip("&").rstrip("&")
      else:
        merged_string = lst[i].strip().rstrip("&")
      inner_count = 1
      while i+inner_count < len(lst) and lst[i+inner_count].strip()[-1].endswith("&"):
        if lst[i+inner_count].strip().startswith("&"):
          merged_string += lst[i+inner_count].strip().lstrip("&").rstrip("&")
        else:
          merged_string += lst[i+inner_count].strip().rstrip("&")
        inner_count+=1
      merged_string += lst[i+inner_count].strip().lstrip("&").rstrip("&")
      merged_list.append(merged_string)
      i += inner_count+1
    else:
      if lst[i].strip().startswith("&"):
        merged_list.append(lst[i].strip().lstrip("&"))
      else:
        merged_list.append(lst[i])
      i += 1
  return merged_list

