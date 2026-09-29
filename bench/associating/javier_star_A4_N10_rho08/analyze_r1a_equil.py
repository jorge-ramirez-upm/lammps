#!/usr/bin/env python3
"""Summarize R1-A active/self/intermolecular network snapshots."""
import argparse, csv, glob, re
p=argparse.ArgumentParser(); p.add_argument('snapshots',nargs='+'); p.add_argument('--out',default='r1a_network_summary.csv'); a=p.parse_args()
files=sum((glob.glob(x) for x in a.snapshots),[]); rows=[]
for path in sorted(files):
    lines=open(path).read().splitlines(); m=re.search(r'timestep (\d+) bonds (\d+)',lines[0]); selfb=inter=0
    for line in lines[2:]:
        x=line.split()
        if len(x)==4:
            if x[2]==x[3]: selfb+=1
            else: inter+=1
    rows.append(dict(step=int(m.group(1)),active=int(m.group(2)),self=selfb,inter=inter,file=path))
with open(a.out,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=['step','active','self','inter','file']); w.writeheader(); w.writerows(rows)
for r in rows: print('step={step} active={active} self={self} inter={inter}'.format(**r))
