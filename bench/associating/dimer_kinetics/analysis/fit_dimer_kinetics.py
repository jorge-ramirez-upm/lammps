#!/usr/bin/env python3
"""Fit A + A <=> B trajectories from in.dimer_kinetics.lmp (stdlib only)."""
import argparse, csv, glob, math, os, re, statistics, sys

def read(path):
    rows=[]
    with open(path) as f:
        for line in f:
            x=line.split()
            if len(x)==7 and x[0] != '#':
                try:
                    z=tuple(map(float,x)); rows.append((z[0],z[0]*0.005,*z[1:]))
                except ValueError: pass
    if len(rows)<10: raise ValueError('%s has too few data rows' % path)
    return rows

def bmodel(t, c, kf, kb):
    # dB/dt=kf(c-2B)^2-kb B, B(0)=0.  Roots b- (physical), b+.
    disc=kb*kb+8.0*kb*kf*c
    root=math.sqrt(disc)
    bm=(4*kf*c+kb-root)/(8*kf)
    bp=(4*kf*c+kb+root)/(8*kf)
    r=(bm/bp)*math.exp(-root*t)
    return (bm-r*bp)/(1-r)

def sse(rows, lkf, lkb):
    kf,kb=math.exp(lkf),math.exp(lkb); c=rows[0][4]+2*rows[0][5]
    return sum((r[5]-bmodel(r[1],c,kf,kb))**2 for r in rows)

def fit(rows):
    # small deterministic coordinate search in log-rate space; no scipy dependency.
    best=(float('inf'), -5., -5.)
    for a in range(-10,3,2):
      for b in range(-10,3,2):
        z=sse(rows,a,b)
        if z<best[0]: best=(z,float(a),float(b))
    _,x,y=best; step=2.
    while step>1e-5:
      changed=False
      for dx,dy in ((step,0),(-step,0),(0,step),(0,-step)):
        z=sse(rows,x+dx,y+dy)
        if z<best[0]: best=(z,x+dx,y+dy); x,y=x+dx,y+dy; changed=True
      if not changed: step*=0.5
    return math.exp(x),math.exp(y),best[0]

def meanse(xs):
    m=statistics.mean(xs)
    return m, (statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else float('nan'))
def linfit(xs,ys):
    xm,ym=statistics.mean(xs),statistics.mean(ys)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    return ym-slope*xm,slope

def main():
 p=argparse.ArgumentParser(); p.add_argument('files',nargs='+'); p.add_argument('--summary',default='summary.csv'); args=p.parse_args()
 files=sum((glob.glob(x) for x in args.files),[]); out=[]
 for path in sorted(set(files)):
    m=re.search(r'Ea([0-9.]+)_Ee([0-9.]+)_N([0-9]+)_r([0-9]+)',os.path.basename(path))
    if not m: continue
    rows=read(path); kf,kb,err=fit(rows); c=rows[0][4]+2*rows[0][5]
    # Last half is equilibrium; block its concentration samples into 10 blocks.
    eq=rows[len(rows)//2:]; nb=10; w=max(1,len(eq)//nb)
    vals=[statistics.mean(r[5] for r in eq[i:i+w])/(statistics.mean(r[4] for r in eq[i:i+w])**2) for i in range(0,len(eq),w) if eq[i:i+w]]
    keq,keqse=meanse(vals)
    mass=max(abs(r[4]+2*r[5]-c) for r in rows)
    # dimer change and creation-break diagnostics must agree after the first record.
    d0=rows[0][3]; e0=rows[0][6]-rows[0][7]
    diag=max(abs((r[3]-d0)-((r[6]-r[7])-e0)) for r in rows)
    out.append(dict(file=path,Ea=float(m.group(1)),Ee=float(m.group(2)),Nevery=int(m.group(3)),rep=int(m.group(4)),kf=kf,kb=kb,Keq_kin=kf/kb,Keq_eq=keq,Keq_eq_block_se=keqse,fit_sse=err,mass_error=mass,diagnostic_error=diag))
 fields=list(out[0]) if out else []
 with open(args.summary,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
 for x in out: print('{file}: kf={kf:.5g} kb={kb:.5g} Keq={Keq_kin:.5g} Keq_eq={Keq_eq:.5g} mass={mass_error:.2e} diag={diagnostic_error:.2e}'.format(**x))
 groups={}
 for x in out: groups.setdefault((x['Ea'],x['Ee'],x['Nevery']),[]).append(x)
 with open(os.path.splitext(args.summary)[0]+'_conditions.csv','w',newline='') as f:
    fields=['Ea','Ee','Nevery','replicas','kf','kf_se','kb','kb_se','Keq_kin','Keq_kin_se','Keq_eq','Keq_eq_se']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for key,g in sorted(groups.items()):
      z={'Ea':key[0],'Ee':key[1],'Nevery':key[2],'replicas':len(g)}
      for name in ('kf','kb','Keq_kin','Keq_eq'): z[name],z[name+'_se']=meanse([q[name] for q in g])
      w.writerow(z)
 # requested log-linear reports, only when sufficient distinct control values exist
 for label, selector, xfun, yfun in [('K1-A',lambda q:q['Ee']==4 and q['Nevery']==100,lambda q:q['Ea'],lambda q:math.log(q['kf'])),('K1-B-kb',lambda q:q['Ea']==4 and q['Nevery']==100,lambda q:q['Ee'],lambda q:math.log(q['kb'])),('K1-B-Keq',lambda q:q['Ea']==4 and q['Nevery']==100,lambda q:q['Ee'],lambda q:math.log(q['Keq_eq']))]:
    g=[q for q in out if selector(q)]; by={}
    for q in g: by.setdefault(xfun(q),[]).append(yfun(q))
    if len(by)>1:
      xs=sorted(by); a,b=linfit(xs,[statistics.mean(by[x]) for x in xs]); print('%s: intercept %.5g slope %.5g' % (label,a,b))
 if out and (max(q['mass_error'] for q in out)>1e-8 or max(q['diagnostic_error'] for q in out)>1e-8): sys.exit('consistency check failed')
if __name__=='__main__': main()
