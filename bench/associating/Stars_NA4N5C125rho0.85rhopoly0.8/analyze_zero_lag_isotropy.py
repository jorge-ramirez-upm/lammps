#!/usr/bin/env python3
"""R1-B2 instantaneous stress, covariance, and network-isotropy analysis."""
import argparse, csv, glob, os, re
import numpy as np

NAMES = ('total', 'kinetic', 'wca', 'permanent', 'associating')
TENSORS = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')

def load(path):
    a = np.loadtxt(path, comments='#')
    return a[None, :] if a.ndim == 1 else a

def metrics(x):
    shear = x[:, 3:6]
    normal = np.column_stack((x[:, 0]-x[:, 1], x[:, 0]-x[:, 2], x[:, 1]-x[:, 2]))
    cs, cn = np.mean(shear*shear), np.mean(normal*normal)
    return cs, cn, cn/(4*cs) if cs else np.nan

def sem(x): return np.std(x, ddof=1)/np.sqrt(len(x)) if len(x) > 1 else np.nan

def read_bonds(data):
    out=[]; active=False
    for line in open(data):
        if line.strip() == 'Bonds': active=True; continue
        if active and line.strip() and line.split()[0].isdigit():
            p=line.split(); out.append((int(p[2]), int(p[3])))
    return out

def frames(path):
    with open(path) as f:
        while True:
            if not f.readline(): return
            step=int(f.readline()); f.readline(); n=int(f.readline())
            f.readline(); f.readline(); f.readline(); f.readline()
            cols=f.readline().split()[2:]
            a=np.array([[float(v) for v in f.readline().split()] for _ in range(n)])
            yield step, {c:a[:, i] for i,c in enumerate(cols)}

def network(path):
    return {(int(a),int(b)) for a,b,*_ in (line.split() for line in open(path)) if a.isdigit()}

def orientation(pos, bonds):
    ids={int(i):k for k,i in enumerate(pos['id'])}; q=[]
    xyz=np.column_stack((pos['xu'],pos['yu'],pos['zu']))
    for a,b in bonds:
        if a in ids and b in ids:
            d=xyz[ids[b]]-xyz[ids[a]]; r=np.linalg.norm(d)
            if r: q.append(np.outer(d,d)/(r*r))
    return np.mean(q,axis=0) if q else np.full((3,3),np.nan)

def structural(traj, data, prefix):
    permanent=read_bonds(data); rows=[]; prev=None
    for step,pos in frames(traj):
        path='%s.network.%d.dat'%(prefix,step); active=network(path) if os.path.exists(path) else None
        for name,bonds in (('permanent_orientation',permanent),) + ((('active_orientation',active),) if active is not None else ()):
            q=orientation(pos,bonds); rows.append(dict(step=step, tensor=name, **{TENSORS[i]:q[(0,1,2,0,0,1)[i],(0,1,2,1,2,2)[i]] for i in range(6)}))
        if prev is not None and active is not None:
            rows.append(dict(step=step,tensor='network_persistence',xx=len(active&prev)/len(prev) if prev else np.nan,yy=np.nan,zz=np.nan,xy=np.nan,xz=np.nan,yz=np.nan))
        if active is not None: prev=active
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('raw'); p.add_argument('--out', default='r1b2'); p.add_argument('--block-steps',type=int,default=10000)
    p.add_argument('--trajectory'); p.add_argument('--data'); a=p.parse_args(); raw=load(a.raw); _, keep=np.unique(raw[:,0],return_index=True); raw=raw[np.sort(keep)]
    step=raw[:,0].astype(int); values={n:raw[:,5+6*i:11+6*i] for i,n in enumerate(NAMES)}
    summed=sum(values[n] for n in NAMES[1:]); residual=values['total']-summed
    rows=[]; blocks=[]
    for name,x in values.items():
        cs,cn,r=metrics(x); rows.append(dict(component=name,Cs0=cs,Cn0=cn,R0=r,**{'mean_'+t:x[:,i].mean() for i,t in enumerate(TENSORS)}))
        for b in range((step[-1]//a.block_steps)+1):
            y=x[(step//a.block_steps)==b]
            if len(y) >= a.block_steps:
                q=metrics(y); blocks.append(dict(block=b,component=name,n=len(y),Cs0=q[0],Cn0=q[1],R0=q[2]))
    total_blocks=np.array([r['R0'] for r in blocks if r['component']=='total'])
    with open(a.out+'.components.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with open(a.out+'.blocks.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=blocks[0]); w.writeheader(); w.writerows(blocks)
    cross=[]
    for kind,idx in (('shear',(3,4,5)),('normal_difference',None)):
        z={n:(values[n][:,idx] if idx else np.column_stack((values[n][:,0]-values[n][:,1],values[n][:,0]-values[n][:,2],values[n][:,1]-values[n][:,2]))) for n in NAMES[1:]}
        for i,x in enumerate(NAMES[1:]):
            for y in NAMES[1:]: cross.append(dict(kind=kind,row=x,column=y,value=np.mean(z[x]*z[y])))
    with open(a.out+'.cross_terms.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=cross[0]); w.writeheader(); w.writerows(cross)
    diag=[]
    for b in range((step[-1]//a.block_steps)+1):
        m=(step//a.block_steps)==b
        if m.any():
            cs,cn,r=metrics(values['total'][m]); diag.append(dict(block=b,active=raw[m,2].mean(),pe=raw[m,1].mean(),R0=r,Cs0=cs,Cn0=cn))
    with open(a.out+'.stationarity.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=diag[0]); w.writeheader(); w.writerows(diag)
    summary=dict(samples=len(raw),blocks=len(total_blocks),block_steps=a.block_steps,R0=float(total_blocks.mean()),R0_sem=sem(total_blocks),max_tensor_sum_residual=np.abs(residual).max())
    if a.trajectory and a.data:
        sr=structural(a.trajectory,a.data,os.path.splitext(a.raw)[0])
        with open(a.out+'.structural.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=sr[0]); w.writeheader(); w.writerows(sr)
        for kind in ('permanent_orientation','active_orientation'):
            q=np.array([[r[t] for t in ('xx','yy','zz')] for r in sr if r['tensor']==kind]); summary[kind+'_max_abs_from_1_3']=float(np.abs(q.mean(0)-1/3).max())
    ok=abs(summary['R0']-1) <= 2*summary['R0_sem'] if np.isfinite(summary['R0_sem']) else False
    summary['gate']='A: total statistically compatible with isotropy' if ok else 'pending/failed: inspect components, cross_terms.csv, and structural.csv'
    with open(a.out+'.summary.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=summary); w.writeheader(); w.writerow(summary)
    print('R0=%g +/- %g (%d blocks); max tensor sum residual=%g; %s'%(summary['R0'],summary['R0_sem'],summary['blocks'],summary['max_tensor_sum_residual'],summary['gate']))
    try:
        import matplotlib.pyplot as plt
        for key in ('R0','Cs0','Cn0','active','pe'):
            plt.figure(); plt.plot([r['block'] for r in diag],[r[key] for r in diag]); plt.xlabel('block'); plt.ylabel(key); plt.savefig(a.out+'.'+key+'.png',dpi=140); plt.close()
    except ImportError: pass
if __name__ == '__main__': main()
