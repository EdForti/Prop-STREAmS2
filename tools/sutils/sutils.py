#!/usr/bin/env python3

import os
import sys

from tools import setup_logging 
from parse_args import process_args
from backend_generation.base import backend_entry

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

def main():
  
  args = process_args()
  if args.act=="backend":
    setup_logging("log_backend.log")
    backend_entry(args)

if __name__ == "__main__":
    main()
