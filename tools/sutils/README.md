# sutils

sutils short for "STREAMS utilities" is a Python3 package developed to support [STREAmS-2](https://github.com/STREAmS-CFD/STREAmS-2). 

## Main features

* Obtaining STREAmS-2 HIPFort backend
* Obtaining STREAmS-2 CPU backend
* Obtaining STREAmS-2 OMP backend (Targetting both CPU and GPU)
* Perform indentation

## Requirements

* Python3 (>=3.11)
* from `sutils/`, run:
```
pip3 install -r install_requirements.txt 
```

## Setup

Add the following to `~/.bashrc`
```
export PATH=$PATH:<path/to/sutils>/bin
```
