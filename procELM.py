from sys import argv
f=open(argv[1])

seq = {}
pos = {}

for line in f:
    parts = line.strip().split("\t") #1: UPID, 2: Pos, 6: Seq
    if parts[1] not in seq:
        seq[parts[1]] = parts[6]
    if parts[1] in pos:
        pos[parts[1]].append(parts[2])
    else:
        pos[parts[1]] = [parts[2]]

f.close()
    
g=open(argv[2]+".fasta","w")
for s in seq.keys():
    g.write(">%s\n%s\n"%(s,seq[s]))
g.close()

h=open(argv[2]+".csv","w")
for p in pos.keys():
    for n in pos[p]:
        h.write("%s,%s\n"%(p,n))
h.close()
