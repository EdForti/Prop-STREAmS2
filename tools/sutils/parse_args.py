"""
   Module for processing arguments. 

"""
import argparse
import sys
import os
import shutil

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
CURRENT_PATH = os.path.realpath(os.path.dirname(__file__))

def process_args():
  """Processing user arguments.

     argparse module is used to read the user specified arguments.
  """
  parser = argparse.ArgumentParser(description="sutils: STREAmS utility tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter,epilog="See '<command> --help' to read about a specific sub-command.")
  
  parser.add_argument("--working-dir",default=os.getcwd(),type=str,help="Set working directory")
  parser.add_argument("--equation",help="Equation type",type=str,default="multideal")
  
  subparsers = parser.add_subparsers(dest="act",help="Sub-commands help")
  
  backend_parser = subparsers.add_parser("backend", help="Generate a STREAmS-2 backend",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  backend_parser.add_argument("--generate_amd",help="Generate HIPFort backend for AMD GPUs",action="store_true")
  backend_parser.add_argument("--generate_cpu",help="Generate CPU backend",action="store_true")
  backend_parser.add_argument("--generate_omp",help="Generate OMP backend",action="store_true")
  backend_parser.add_argument("--generate_ompc",help="Generate OMP CPU backend",action="store_true")
  backend_parser.add_argument("--remove_plugins",help="Specify the plugins to remove",choices=["ibm","insitu"],type=str,nargs="+",default=[])
  backend_parser.add_argument("--output_dir",default=os.getcwd(),type=str,help="Output directory for the generated code")
  backend_parser.add_argument("--input_dir",help="Path to the input code",type=str)
  backend_parser.add_argument("--generate_gpu_arrays",help="Generate the linearised GPU array file",action=argparse.BooleanOptionalAction,default="generate_gpu_arrays")
  backend_parser.add_argument("--indent",help="Indent the output code",action="store_true")

  backend_config = backend_parser.add_argument_group("backend config options")
  backend_config.add_argument("--backend_config",default=f"{CURRENT_PATH}/backend_generation/backend_config.toml",type=str,help="Config file for the backend generation")
  backend_config.add_argument("--copy_default_config",help="Copy default backend config to the current directory",action="store_true")
  
  #indent_parser = subparsers.add_parser("indent", help="Generate indented STREAmS-2",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  args = parser.parse_args()

  if not vars(args)["act"]:
    print("\n--------------------------------------------------------------")
    print("\033[1m"+"No arguments provided. Please provide at least one argument."+"\033[0m\n")
    print("\033[1m"+"Type sutils -h"+"\033[0m\n")
    print("--------------------------------------------------------------\n")
    
    sys.exit(0)
  
  if args.copy_default_config:
    shutil.copy2(f"{CURRENT_PATH}/backend_generation/backend_config.toml","./")
    sys.exit(0)
  
  return args
