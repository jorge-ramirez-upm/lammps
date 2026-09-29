#!/usr/bin/env python3
"""K1 event/exposure and analytic-transient estimators (stdlib only)."""
import argparse, csv, glob, math, os, random, re, statistics, sys

def read(path, dt):
    rows=[]
    for line in open(path):
        x=line.split()
        if len(x)==7 and x[0] != '#':
            try:
                z=tuple(map(float,x)); rows.append((z[0],z[0]*dt,*z[1:]))
            except ValueError: pass
    if len(rows)<10: raise ValueError('%s has too few rows' % path)
    return rows

def trap(rows, value):
    return sum((rows[i+1][1]-rows[i][1])*(value(rows[i])+value(rows[i+1]))/2 for i in range(len(rows)-1))
def volume(rows):
    r=rows[0]; return (r[2]+2*r[3])/(r[4]+2*r[5])
def bmodel(t,c,kf,kb):
    d=kb*kb+8*kb*kf*c; root=math.sqrt(d)
    bm=(4*kf*c+kb-root)/(8*kf); bp=(4*kf*c+kb+root)/(8*kf); z=(bm/bp)*math.exp(-root*t)
    return (bm-z*bp)/(1-z)
def sse(rows,x,y):
    kf,kb=math.exp(x),math.exp(y); c=rows[0][4]+2*rows[0][5]
    return sum((r[5]-bmodel(r[1],c,kf,kb))**2 for r in rows)
def fit(rows):
    best=(float('inf'),-5.,-5.)
    for x in range(-10,3,2):
        for y in range(-10,3,2): best=min(best,(sse(rows,x,y),float(x),float(y)))
    _,x,y=best; step=2.
    for _ in range(60):
        if step<=1e-5: break
        changed=False
        for dx,dy in ((step,0),(-step,0),(0,step),(0,-step)):
            z=sse(rows,x+dx,y+dy)
            if z<best[0]: best=(z,x+dx,y+dy); x,y=x+dx,y+dy; changed=True
        if not changed: step*=.5
    return math.exp(x),math.exp(y),best[0]
def meanse(xs):
    xs=[x for x in xs if math.isfinite(x)]
    return (statistics.mean(xs), statistics.stdev(xs)/math.sqrt(len(xs)) if len(xs)>1 else float('nan'))
def linfit(xs,ys):
    xm,ym=statistics.mean(xs),statistics.mean(ys); slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
    return ym-slope*xm,slope
def qvalue(ea,every,nu0,temp,dt): return -math.expm1(-nu0*math.exp(-ea/temp)*every*dt)
def rel(a,b): return a/b-1 if b else float('nan')
def event_rates(rows):
    v=volume(rows); made=rows[-1][6]-rows[0][6]; broke=rows[-1][7]-rows[0][7]
    # Exact finite-N formation exposure: integral N_A(N_A-1)/V dt.
    hf=trap(rows,lambda r:r[2]*(r[2]-1)/v); hb=trap(rows,lambda r:r[3])
    # Quantify trapezoid discretization by comparison with endpoint rectangles.
    def rect(fn,right): return sum((rows[i+1][1]-rows[i][1])*fn(rows[i+1 if right else i]) for i in range(len(rows)-1))
    hfl,hfr=rect(lambda r:r[2]*(r[2]-1)/v,False),rect(lambda r:r[2]*(r[2]-1)/v,True)
    hbl,hbr=rect(lambda r:r[3],False),rect(lambda r:r[3],True)
    exposure_error=max(abs(hf-hfl)/hf,abs(hf-hfr)/hf,abs(hb-hbl)/hb if hb else 0,abs(hb-hbr)/hb if hb else 0)
    kf=made/hf if hf else float('nan'); kb=broke/hb if hb else float('nan')
    return made,broke,hf,hb,kf,kb,exposure_error
def equilibrium(rows):
    eq=rows[len(rows)//2:]; w=max(1,len(eq)//10)
    values=[statistics.mean(r[5] for r in eq[i:i+w])/(statistics.mean(r[4] for r in eq[i:i+w])**2) for i in range(0,len(eq),w)]
    return meanse(values)
def bootstrap(points, reference=None, nboot=2000, seed=202510):
    """Replica-level bootstrap of y=a+m*x, or y=C+reference(x)."""
    rng=random.Random(seed); slopes=[]; intercepts=[]
    for _ in range(nboot):
        ys=[statistics.mean(rng.choice(v) for _ in v) for _,v in points]
        xs=[x for x,_ in points]
        if reference is None: a,m=linfit(xs,ys); slopes.append(m); intercepts.append(a)
        else: intercepts.append(statistics.mean(y-reference(x) for x,y in zip(xs,ys)))
    def boot(xs): return statistics.mean(xs), statistics.stdev(xs)
    if reference is None: return boot(intercepts)+boot(slopes)
    return boot(intercepts)+(float('nan'),float('nan'))
def trend(rows,rho,label,select,xkey,ykey,reference=None):
    by={}
    for r in rows:
        if r['rho']==rho and select(r) and r[ykey]>0: by.setdefault(r[xkey],[]).append(math.log(r[ykey]))
    if len(by)<2: return None
    points=sorted(by.items()); inter,inter_se,slope,slope_se=bootstrap(points,reference)
    mean_y=[statistics.mean(v) for _,v in points]; xs=[x for x,_ in points]
    residual=[y-(inter+(reference(x) if reference else slope*x)) for x,y in zip(xs,mean_y)]
    return dict(rho=rho,series=label,points=len(points),intercept=inter,intercept_se=inter_se,slope=slope,slope_se=slope_se,rmse=math.sqrt(sum(x*x for x in residual)/len(residual)))
def main():
    p=argparse.ArgumentParser(); p.add_argument('files',nargs='+'); p.add_argument('--summary',default='summary.csv'); p.add_argument('--dt',type=float,default=.005); p.add_argument('--nu0',type=float,default=20); p.add_argument('--temperature',type=float,default=1); a=p.parse_args()
    pat=re.compile(r'(?:rho([0-9.]+)_)?Ea([0-9.]+)_Ee([0-9.]+)_N([0-9]+)_r([0-9]+)'); out=[]
    for path in sorted(set(sum((glob.glob(x) for x in a.files),[]))):
        m=pat.search(os.path.basename(path))
        if not m: continue
        rho=float(m.group(1) or .05); ea,ee,every,rep=float(m.group(2)),float(m.group(3)),int(m.group(4)),int(m.group(5)); rows=read(path,a.dt)
        kff,kbf,s=fit(rows); made,broke,hf,hb,kfe,kbe,exerr=event_rates(rows); q=qvalue(ea,every,a.nu0,a.temperature,a.dt); keq,keqse=equilibrium(rows)
        var=sum((r[5]-statistics.mean(z[5] for z in rows))**2 for r in rows); c=rows[0][4]+2*rows[0][5]
        mass=max(abs(r[4]+2*r[5]-c) for r in rows); d0,e0=rows[0][3],rows[0][6]-rows[0][7]; diag=max(abs((r[3]-d0)-((r[6]-r[7])-e0)) for r in rows)
        out.append(dict(file=path,rho=rho,Ea=ea,Ee=ee,Nevery=every,rep=rep,q=q,event_creations=made,event_breaks=broke,forward_exposure=hf,backward_exposure=hb,exposure_trapezoid_rel_error=exerr,kf_event=kfe,kb_event=kbe,Keq_event=kfe/kbe,kf_event_over_q=kfe/q,kb_event_over_q=kbe/q,kf_fit=kff,kb_fit=kbf,Keq_fit=kff/kbf,kf_event_fit_rel=rel(kfe,kff),kb_event_fit_rel=rel(kbe,kbf),Keq_event_fit_rel=rel(kfe/kbe,kff/kbf),Keq_eq=keq,Keq_eq_block_se=keqse,fit_sse=s,fit_r2=1-s/var if var else float('nan'),mass_error=mass,diagnostic_error=diag))
    if not out: sys.exit('no matching trajectories')
    with open(a.summary,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    groups={}
    for r in out: groups.setdefault((r['rho'],r['Ea'],r['Ee'],r['Nevery']),[]).append(r)
    names=('q','event_creations','event_breaks','forward_exposure','backward_exposure','exposure_trapezoid_rel_error','kf_event','kb_event','Keq_event','kf_event_over_q','kb_event_over_q','kf_fit','kb_fit','Keq_fit','kf_event_fit_rel','kb_event_fit_rel','Keq_event_fit_rel','Keq_eq','fit_sse','fit_r2')
    cp=os.path.splitext(a.summary)[0]+'_conditions.csv'
    with open(cp,'w',newline='') as f:
        fields=['rho','Ea','Ee','Nevery','replicas']+[z for n in names for z in (n,n+'_se')]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for key,g in sorted(groups.items()):
            z=dict(zip(('rho','Ea','Ee','Nevery'),key)); z['replicas']=len(g)
            for n in names: z[n],z[n+'_se']=meanse([r[n] for r in g])
            w.writerow(z)
    trends=[]
    for rho in sorted({r['rho'] for r in out}):
        act=lambda r:r['Ee']==4 and r['Nevery']==100; bind=lambda r:r['Ea']==4 and r['Nevery']==100
        trends += [x for x in (
          trend(out,rho,'activation_event_lnkf',act,'Ea','kf_event'),
          trend(out,rho,'activation_event_lnq_offset',act,'Ea','kf_event',lambda x:math.log(qvalue(x,100,a.nu0,a.temperature,a.dt))),
          trend(out,rho,'activation_event_ln_kf_over_q',act,'Ea','kf_event_over_q'),
          trend(out,rho,'activation_event_ln_kb_over_q',act,'Ea','kb_event_over_q'),
          trend(out,rho,'binding_event_lnkb',bind,'Ee','kb_event'),
          trend(out,rho,'binding_lnKeq_event',bind,'Ee','Keq_event'),
          trend(out,rho,'binding_lnKeq_fit',bind,'Ee','Keq_fit'),
          trend(out,rho,'binding_lnKeq_eq',bind,'Ee','Keq_eq')) if x]
    tp=os.path.splitext(a.summary)[0]+'_trends.csv'
    with open(tp,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=['rho','series','points','intercept','intercept_se','slope','slope_se','rmse']); w.writeheader(); w.writerows(trends)
    for x in trends: print('rho={rho:g} {series}: intercept={intercept:.4g}+/-{intercept_se:.2g} slope={slope:.4g}+/-{slope_se:.2g}'.format(**x))
    if max(r['mass_error'] for r in out)>1e-8 or max(r['diagnostic_error'] for r in out)>1e-8: sys.exit('consistency check failed')
if __name__=='__main__': main()
