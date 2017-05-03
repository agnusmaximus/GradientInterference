from __future__ import print_function
import sys
import glob

def extract_info(f):
    lines = []
    for line in f:
        if "IInfo" in line:
            lines.append(line)
    return lines

def output_info(data, f):
    for line in data:
        print(line, file=f)

dname = sys.argv[1]
files = glob.glob(dname + "/*")
for fname in files:
    f = open(fname, "r")
    info = extract_info(f)
    f.close()
    f = open(fname, "w")
    output_info(info, f)
    f.close()
