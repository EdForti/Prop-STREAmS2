import os
import sys
import logging

from backend_generation.backend import InitBackendAmd,InitBackendCpu,InitBackendOmp,InitBackendOmpc,InitBackendOriginal
from backend_generation.generate_backend import GenerateAmd,GenerateCpu,GenerateOmp,GenerateOmpc

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

def backend_entry(args):
  if args.generate_amd:
    logging.info(f"Generating the AMD backend")
    init_obj = InitBackendAmd(args)
    amd_obj = GenerateAmd(init_obj,args)
    amd_obj.generatefiles()
  elif args.generate_cpu:
    logging.info("Generating CPU backend")
    init_obj = InitBackendCpu(args)
    cpu_obj = GenerateCpu(init_obj,args)
    cpu_obj.generatefiles()
  elif args.generate_omp:
    logging.info("Generating OMP backend")
    init_obj = InitBackendOmp(args)
    omp_obj = GenerateOmp(init_obj,args)
    omp_obj.generatefiles()
  elif args.generate_ompc:
    logging.info("Generating OMP-CPU backend")
    init_obj = InitBackendOmpc(args)
    omp_obj = GenerateOmpc(init_obj,args)
    omp_obj.generatefiles()
  else:
    if args.remove_plugins:
      logging.info(f"Removing plugins for the CUDA Fortran backend")
      init_obj = InitBackendOriginal(args)
    else:
      logging.error(f"No backend generation chosen")
      raise Exception("No backend option chosen!")  
