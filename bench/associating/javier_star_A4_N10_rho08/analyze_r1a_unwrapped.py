#!/usr/bin/env python3
"""Topology-consistent unwrapped R1-A.1 conformation/stationarity summary."""
import argparse, csv, glob, gzip, math, os, re


def bonds(path):
    graph={}; active=False
    for line in open(path):
        if line.startswith('Bonds'): active=True; continue
        if not active or not line.split(): continue
        x=line.split()
        if len(x)>=4:
            a,b=int(x[2]),int(x[3]); graph.setdefault(a,[]).append(b); graph.setdefault(b,[]).append(a)
    return graph


def frames(path):
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
        while True:
            line=f.readline()
            if not line: return
            if not line.startswith('ITEM: TIMESTEP'): continue
            step=int(f.readline()); f.readline(); n=int(f.readline())
            f.readline(); box=[tuple(map(float,f.readline().split()[:2])) for _ in range(3)]
            names=f.readline().split()[2:]; index={x:i for i,x in enumerate(names)}
            need=('id','mol','xu','yu','zu')
            if any(x not in index for x in need): raise ValueError('dump needs '+str(need))
            bymol={}
            for _ in range(n):
                x=f.readline().split(); tag=int(x[index['id']]); mol=int(x[index['mol']])
                bymol.setdefault(mol,{})[tag]=tuple(float(x[index[k]]) for k in ('xu','yu','zu'))
            yield step,box,bymol


def unwrap(points, box, graph):
    """Repair only periodic integer shifts of xu using permanent FENE bonds."""
    length=[hi-lo for lo,hi in box]; wrapped={tag:tuple(lo+(q-lo)%l for q,lo,l in zip(p,(b[0] for b in box),length)) for tag,p in points.items()}
    root=next(iter(points)); out={root:points[root]}; todo=[root]
    while todo:
        a=todo.pop()
        for b in graph.get(a,[]):
            if b not in points or b in out: continue
            d=[]
            for k in range(3):
                v=wrapped[b][k]-wrapped[a][k]; v-=length[k]*math.floor(v/length[k]+.5); d.append(v)
            out[b]=tuple(out[a][k]+d[k] for k in range(3)); todo.append(b)
    if len(out)!=len(points): raise ValueError('permanent-bond graph does not connect a star')
    return out


def stats(values):
    n=len(values); mean=sum(values)/n; sd=math.sqrt(sum((x-mean)**2 for x in values)/(n-1)) if n>1 else 0.
    return mean,sd,sd/math.sqrt(n)


def network(path):
    lines=open(path).read().splitlines(); m=re.search(r'timestep (\d+) bonds (\d+)',lines[0]); selfb=sum(1 for line in lines[2:] if len(line.split())==4 and line.split()[2]==line.split()[3])
    return int(m.group(1)),int(m.group(2)),selfb,int(m.group(2))-selfb


def diagnostic(path):
    return [(int(x[0]),*[float(y) for y in x[1:6]]) for line in open(path) if not line.startswith('#') if len(x:=line.split())>=6]


def trend(name, rows, value):
    rows=rows[len(rows)//2:]; x=[r[0] for r in rows]; y=[value(r) for r in rows]; n=len(x); xm=sum(x)/n; ym=sum(y)/n; den=sum((z-xm)**2 for z in x)
    slope=sum((z-xm)*(w-ym) for z,w in zip(x,y))/den if den else 0.; rms=math.sqrt(sum((w-(ym+slope*(z-xm)))**2 for z,w in zip(x,y))/n)
    return dict(observable=name,n=n,mean=ym,slope_per_step=slope,late_rms=rms,change=y[-1]-y[0])


def main():
    p=argparse.ArgumentParser(); p.add_argument('trajectory'); p.add_argument('--data',required=True,help='original permanent-bond data file'); p.add_argument('--diagnostics'); p.add_argument('--snapshots',nargs='*',default=[]); p.add_argument('--out-prefix'); a=p.parse_args()
    prefix=a.out_prefix or os.path.splitext(a.trajectory)[0]; graph=bonds(a.data); summary=[]; raw=[]; origin=None
    for step,box,bymol in frames(a.trajectory):
        cms={}; rg2=[]; rg=[]
        for mol,points in bymol.items():
            pts=list(unwrap(points,box,graph).values()); cm=tuple(sum(q[k] for q in pts)/len(pts) for k in range(3)); cms[mol]=cm
            value=sum(sum((q[k]-cm[k])**2 for k in range(3)) for q in pts)/len(pts); rg2.append(value); rg.append(math.sqrt(value)); raw.append(dict(step=step,molecule=mol,rg2=value,rg=math.sqrt(value)))
        if origin is None: origin=cms
        if set(cms)!=set(origin): raise ValueError('star molecule set changed between frames')
        msd=sum(sum((cms[m][k]-origin[m][k])**2 for k in range(3)) for m in cms)/len(cms); m2,s2,e2=stats(rg2); mr,sr,er=stats(rg)
        summary.append(dict(step=step,nstars=len(cms),mean_rg2=m2,std_rg2=s2,sem_rg2=e2,mean_rg=mr,std_rg=sr,sem_rg=er,com_msd_from_start=msd,min_rg=min(rg),max_rg=max(rg)))
    if not summary: raise SystemExit('no coordinate frames')
    for suffix,rows in (('.r1a_conformation.csv',summary),('.r1a_star_values.csv',raw)):
        with open(prefix+suffix,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report=[trend('mean_rg2',[(r['step'],r['mean_rg2']) for r in summary],lambda r:r[1]),trend('com_msd_from_start',[(r['step'],r['com_msd_from_start']) for r in summary],lambda r:r[1])]
    if a.diagnostics:
        d=diagnostic(a.diagnostics); report += [trend('temperature',d,lambda r:r[1]),trend('potential_energy',d,lambda r:r[2]),trend('active_bonds',d,lambda r:r[3]),trend('cumulative_creations',d,lambda r:r[4]),trend('cumulative_breaks',d,lambda r:r[5])]
    files=sum((glob.glob(x) for x in a.snapshots),[])
    if files:
        n=sorted(network(x) for x in files); report += [trend('self_bonds',n,lambda r:r[2]),trend('intermolecular_bonds',n,lambda r:r[3])]
    with open(prefix+'.r1a_stationarity.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
    print('frames=%d stars=%d rg2=%g..%g com_msd_final=%g' % (len(summary),summary[0]['nstars'],summary[0]['mean_rg2'],summary[-1]['mean_rg2'],summary[-1]['com_msd_from_start']))
    for r in report: print('{observable}: mean={mean:.8g} slope/step={slope_per_step:.4g} late_rms={late_rms:.4g}'.format(**r))

if __name__=='__main__': main()
