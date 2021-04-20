from os import listdir
from os.path import isfile, join

mypath = "./reviews"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

i = 0
for fileName in onlyfiles:
    i += 1
    print(i)
    with open(join(mypath, fileName), "r") as f:
        lines = f.readlines()
        print(lines)
        print()
