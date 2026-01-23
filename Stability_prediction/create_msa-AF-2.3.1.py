import sys
import random
import os

# full msa file
msa_i=sys.argv[1]
# number of sequences
nseq=int(sys.argv[2])

# prepare data dictionary
data_dict={}
# initialize
## GS lines + sequences
data_dict[0]=[]

# open output msa
f=open("uniref90_hits.sto", "w")
# write header
f.write("# STOCKHOLM 1.0\n\n")

# set counter to 1
ic=1
# loop over msa file
for lines in open(msa_i, "r").readlines():
    riga=lines.strip().split()
    # skip empty lines
    if(len(riga)==0): continue
    # skip last line
    if(riga[0]=="//"): continue
    # skip first line
    if(len(riga)==3 and riga[1]=="STOCKHOLM"): continue
    # check if GS line
    if(riga[0]=="#=GS"):
      data_dict[0].append(lines.strip())
    else:
    # append to sequences
      if ic in data_dict:
         data_dict[ic].append(lines.strip())
      else:
         data_dict[ic] = [lines.strip()]
    # increment counter
    if(riga[0]=="#=GC"): ic+=1

# check if target sequence is included in #=GS lines
# check first name of #=GS sequence
str1=data_dict[0][0].split()[1]
# check first name of second block
str2=data_dict[1][0].split()[0]
# are the two names identical?
if(str1==str2): off = 0
else:           off = 1

# now extract random sequences
seq = random.sample(range(1-off,len(data_dict[0])), nseq)

# now write output msa
# #=GS fields
# only if target sequence is present
if(off==0): f.write("%s\n" % data_dict[0][0])
# extracted sequences
for s in seq:
    f.write("%s\n" % data_dict[0][s])
f.write("\n")
# get number of blocks
nblocks=len(data_dict.keys())
# loop over blocks
for i in range(1,nblocks):
    # this is always the target sequence
    f.write("%s\n" % data_dict[i][0])
    # loop over sequences
    for s in seq:
        f.write("%s\n" % data_dict[i][s+off])
    # last line 
    f.write("%s\n" % data_dict[i][-1])
    # if not last iter, add newline
    if(i!=(nblocks-1)): f.write("\n")
# last line
f.write("//\n")
# close file
f.close() 

# copy to mgnify_hits.sto
os.popen('cp uniref90_hits.sto mgnify_hits.sto')
