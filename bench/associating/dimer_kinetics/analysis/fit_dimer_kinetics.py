#!/usr/bin/env python3
"""Fit A + A <=> B trajectories; works without scipy or matplotlib."""
import argparse, csv, glob, math, os, re, statistics, sys

def read(path, dt):
    rows=[]
    with open(path) as f:
        for line in f:
            x=line.split()
            if len(x)==7 and x[0] != '#':
                try:
                    z=tuple(map(float,x)); rows.append((z[0],z[0]*dt,*z[1:]))
                except ValueError: pass
    if len(rows)<10: raise ValueError('%s has too few data rows' % path)
    return rows

def bmodel(t,c,kf,kb):
    disc=kb*kb+8*kb*kf*c; root=math.sqrt(disc)
    bm=(4*kf*c+kb-root)/(8*kf); bp=(4*kf*c+kb+root)/(8*kf)
    r=(bm/bp)*math.exp(-root*t)
    return (bm-r*bp)/(1-r)

def sse(rows,x,y):
    kf,kb=math.exp(x),math.exp(y); c=rows[0][4]+2*rows[0][5]
    return sum((r[5]-bmodel(r[1],c,kf,kb))**2 for r in rows)

def fit(rows):
    best=(float('inf'),-5.,-5.)
    for x in range(-10,3,2):
        for y in range(-10,3,2): best=min(best,(sse(rows,x,y),float(x),float(y)))
    _,x,y=best; step=2.
    while step>1e-5:
        changed=False
        for dx,dy in ((step,0),(-step,0),(0,step),(0,-step)):
            z=sse(rows,x+dx,y+dy)
            if z<best[0]: best=(z,x+dx,y+dy); x,y=x+dx,y+dy; changed=True
        if not changed: step*=.5
    return math.exp(x),math.exp(y),best[0]

def meanse(xs):
    return statistics.mean(xs), statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else float('nan')
def linfit(xs,ys):
    xm,ym=statistics.mean(xs),statistics.mean(ys)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    return ym-slope*xm,slope

def qvalue(ea, every, nu0, temp, dt): return -math.expm1(-nu0*math.exp(-ea/temp)*every*dt)
def condition(rows):
    c=rows[0][4]+2*rows[0][5]; eq=rows[len(rows)//2:]; width=max(1,len(eq)//10)
    blocks=[statistics.mean(r[5] for r in eq[i:i+width])/(statistics.mean(r[4] for r in eq[i:i+width])**2) for i in range(0,len(eq),width)]
    mass=max(abs(r[4]+2*r[5]-c) for r in rows); d0,e0=rows[0][3],rows[0][6]-rows[0][7]
    diag=max(abs((r[3]-d0)-((r[6]-r[7])-e0)) for r in rows)
    return meanse(blocks)+(mass,diag,rows[-1][6],rows[-1][7])

def main():
    p=argparse.ArgumentParser(); p.add_argument('files',nargs='+'); p.add_argument('--summary',default='summary.csv'); p.add_argument('--dt',type=float,default=.005); p.add_argument('--nu0',type=float,default=20); p.add_argument('--temperature',type=float,default=1); a=p.parse_args()
    out=[]; files=sum((glob.glob(x) for x in a.files),[])
    pattern=re.compile(r'(?:rho([0-9.]+)_)?Ea([0-9.]+)_Ee([0-9.]+)_N([0-9]+)_r([0-9]+)')
    for path in sorted(set(files)):
        m=pattern.search(os.path.basename(path))
        if not m: continue
        rho=float(m.group(1) or .05); ea,ee,every,rep=float(m.group(2)),float(m.group(3)),int(m.group(4)),int(m.group(5))
        rows=read(path,a.dt); kf,kb,err=fit(rows); keq,keq_block_se,mass,diag,created,broken=condition(rows); q=qvalue(ea,every,a.nu0,a.temperature,a.dt)
        variance=sum((r[5]-statistics.mean(x[5] for x in rows))**2 for r in rows)
        out.append(dict(file=path,rho=rho,Ea=ea,Ee=ee,Nevery=every,rep=rep,q=q,kf=kf,kb=kb,kf_over_q=kf/q,kb_over_q=kb/q,Keq_kin=kf/kb,Keq_eq=keq,Keq_eq_block_se=keq_block_se,fit_sse=err,fit_r2=1-err/variance if variance else float('nan'),mass_error=mass,diagnostic_error=diag,creations=created,breaks=broken))
    if not out: sys.exit('no matching trajectory files')
    with open(a.summary,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    groups={}
    for x in out: groups.setdefault((x['rho'],x['Ea'],x['Ee'],x['Nevery']),[]).append(x)
    names=('q','kf','kb','kf_over_q','kb_over_q','Keq_kin','Keq_eq','fit_sse','fit_r2','creations','breaks')
    cp=os.path.splitext(a.summary)[0]+'_conditions.csv'
    with open(cp,'w',newline='') as f:
        fields=['rho','Ea','Ee','Nevery','replicas']+[v for n in names for v in (n,n+'_se')]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for key,g in sorted(groups.items()):
            z=dict(zip(('rho','Ea','Ee','Nevery'),key)); z['replicas']=len(g)
            for n in names: z[n],z[n+'_se']=meanse([x[n] for x in g])
            w.writerow(z)
    trends=[]
    for rho in sorted({x['rho'] for x in out}):
      for label,select,xkey,ykey in (
        ('activation_lnkf',lambda z:z['Ee']==4 and z['Nevery']==100,'Ea','kf'),
        ('activation_lnq',lambda z:z['Ee']==4 and z['Nevery']==100,'Ea','q'),
        ('activation_kf_over_q',lambda z:z['Ee']==4 and z['Nevery']==100,'Ea','kf_over_q'),
        ('activation_kb_over_q',lambda z:z['Ee']==4 and z['Nevery']==100,'Ea','kb_over_q'),
        ('binding_lnkb',lambda z:z['Ea']==4 and z['Nevery']==100,'Ee','kb'),
        ('binding_lnKeq_kin',lambda z:z['Ea']==4 and z['Nevery']==100,'Ee','Keq_kin'),
        ('binding_lnKeq_eq',lambda z:z['Ea']==4 and z['Nevery']==100,'Ee','Keq_eq')):
        g=[z for z in out if z['rho']==rho and select(z)]; by={}
        for z in g: by.setdefault(z[xkey],[]).append(math.log(z[ykey]))
        if len(by)>1:
            xs=sorted(by); intercept,slope=linfit(xs,[statistics.mean(by[x]) for x in xs])
            trends.append(dict(rho=rho,series=label,points=len(xs),intercept=intercept,slope=slope))
    tp=os.path.splitext(a.summary)[0]+'_trends.csv'
    with open(tp,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=['rho','series','points','intercept','slope']); w.writeheader(); w.writerows(trends)
    for z in trends: print('rho={rho:g} {series}: intercept={intercept:.5g} slope={slope:.5g}'.format(**z))
    if max(x['mass_error'] for x in out)>1e-8 or max(x['diagnostic_error'] for x in out)>1e-8: sys.exit('consistency check failed')
if __name__=='__main__': main()
