#!/usr/bin/env python3

import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compare two singleideal.ini files')

    parser.add_argument('file1', metavar='1st singleideal', type=str, help='First file to compare')
    parser.add_argument('file2', metavar='2nd singleideal', type=str, help='Second file to compare')

    args = parser.parse_args()

    print(" ========================================================= ")
    print("           Comparing singleideal.ini files                 ")
    print(" ========================================================= ")
    print(" ")

    file1_path = args.file1
    file1_map = dict()
    filter_regex = re.compile(r"\[.*\]|(;.*)")
    with open(file1_path) as f:
        print("Reading file "+file1_path)
        for line_with_comments in f:
            line = line_with_comments.split(";", 1)[0]
            print("line: ",line, line.isspace())
            if not line == "" and not line.isspace() and not filter_regex.match(line):
                splitted = line.strip().split("=",1)
                print("splitted :",splitted)
                key = splitted[0].strip()
                value = splitted[1].strip()
                if key in file1_map:
                    if file1_map[key].__class__.__name__ == [].__class__.__name__:
                        file1_map[key].append(value)
                    else:
                        file1_map[key] = [file1_map[key], value]
                else:
                    file1_map[key] = value

    file2_path = args.file2
    file2_map = dict()
    filter_regex = re.compile(r"\[.*\]|(;.*)")
    with open(file2_path) as f:
        print("Reading file "+file2_path)
        for line_with_comments in f:
            line = line_with_comments.split(";", 1)[0]
            if not line == "" and not line.isspace() and not filter_regex.match(line):
                splitted = line.strip().split("=",1)
                key = splitted[0].strip()
                value = splitted[1].strip()
                if key in file2_map:
                    if file2_map[key].__class__.__name__ == [].__class__.__name__:
                        file2_map[key].append(value)
                    else:
                        file2_map[key] = [file2_map[key], value]
                else:
                    file2_map[key] = value

    file1_keys = set(file1_map.keys())
    #print(file1_keys)
    file2_keys = set(file2_map.keys())

    print(" ")
    print(" ========================================================= ")
    print("             Common keys               ")
    print(" ========================================================= ")
    print(" ")

    for key in (file1_keys & file2_keys):
        if file1_map[key] != file2_map[key]:
            print(" {}".format(key))
            print("  {}".format(file1_map[key]))
            print("  {}".format(file2_map[key]))

    print(" ")
    print(" ========================================================= ")
    print("     Keys unique to "+file1_path)
    print(" ========================================================= ")
    print(" ")

    for key in (file1_keys - file2_keys):
        print(" {} = {}".format(key,str(file1_map[key])))

    print(" ")
    print(" ========================================================= ")
    print("     Keys unique to "+file2_path)
    print(" ========================================================= ")
    print(" ")

    for key in (file2_keys - file1_keys):
        print(" {} = {}".format(key,str(file2_map[key])))

    print(" ")
    print(" ========================================================= ")
    print("            Summary                ")
    print(" ========================================================= ")
    print(" ")
    print(" Common keys:    "+str(len(file1_keys & file2_keys)))
    print(" Extra in file1: "+str(len(file1_keys - file2_keys)))
    print(" Extra in file2: "+str(len(file2_keys - file1_keys)))
    print(" ")

if __name__ == "__main__":
    main()
