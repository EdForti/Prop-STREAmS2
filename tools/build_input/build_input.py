#!/usr/bin/env python3
import re

p_get    = re.compile('call\b* .*cfg%get\("(.*?)","(.*?)",(.*?)\)', re.IGNORECASE)
p_has    = re.compile('cfg%has_key\("(.*?)","(.*?)"\)', re.IGNORECASE)

config   = dict()
opt_dict = dict()

filename = "src/multideal/multideal.F90"
with open(filename, "r") as f:

    for l in f.readlines():
        line = l.strip()
        r_get = p_get.search(line)
        if r_get:
             section_name = r_get.group(1)
             var_name     = r_get.group(2)
             if section_name not in list(config.keys()):
                 config[section_name] = []
             if var_name not in [x["name"] for x in config[section_name]]:
                 config[section_name].append(dict(name=var_name,opt=False))

with open(filename, "r") as f:
    for l in f.readlines():
        line = l.strip()
        r_has = p_has.search(line)
        if r_has:
            opt_dict[r_has.group(1)+"_"+r_has.group(2)] = "yes"


output_file = "multideal_start.ini"

section_names_ordered = ["ref_state", "grid", "bc", "mpi", "controls", \
                         "numerics", "flow", "fluid", "output", "ibmpar", "lespar", "jcfpar", "bl_trip", "limiter", "insitu", "field_info"]

section_names_extracted = list(config.keys())
if set(section_names_extracted) != set(section_names_ordered):
    print("section_names_ordered :",section_names_ordered)
    print("section_names_extracted :",section_names_extracted)
    print("section_names_* DIFF :",set(section_names_extracted).difference(set(section_names_ordered)))
    new_fields = set(section_names_extracted).difference(set(section_names_ordered))
    section_names_ordered += list(new_fields)
    #print("Error! Section name ordering has to be manually updated. Please update section_names_ordered variable above")
    #raise

with open(output_file, "w") as f:
    #for section_name,section_list in config.items():
    f.write("; Singleideal automatically generated\n")
    for section_name in section_names_ordered:
        if section_name != "field_info": # field_info is on another file
            section_list = config[section_name]
            f.write("\n["+section_name+"]\n")
            for var in section_list:
                if section_name+"_"+var["name"] in list(opt_dict.keys()):
                    f.write(var["name"]+" =     ; optional\n")
                else:
                    f.write(var["name"]+" =     ; \n")
