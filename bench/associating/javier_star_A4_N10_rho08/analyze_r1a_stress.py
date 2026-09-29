#!/usr/bin/env python3
"""Reconstruct the canonical R1 modulus from a six-channel .gt file."""
import argparse, csv, math, os, sys

def load(path):
    rows=[]
    for line in open(path):
        if line.startswith('#'): continue
        try:
            x=list(map(float,line.split()))
            if len(x)==7: rows.append(x)
        except ValueError: pass
    if not rows: raise SystemExit('no six-channel correlation rows in '+path)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('gt'); p.add_argument('--volume',type=float,required=True); p.add_argument('--temperature',type=float,default=1); p.add_argument('--run-steps',type=int,required=True); p.add_argument('--dt',type=float,default=.01); a=p.parse_args()
    out=[]; runtime=a.run_steps*a.dt
    for t,cxy,cxz,cyz,cnxy,cnxz,cnyz in load(a.gt):
        gs=a.volume*(cxy+cxz+cyz)/(5*a.temperature); gn=a.volume*(cnxy+cnxz+cnyz)/(30*a.temperature)
        out.append(dict(time=t,Cxy=cxy,Cxz=cxz,Cyz=cyz,CNxy=cnxy,CNxz=cnxz,CNyz=cnyz,G_shear=gs,G_normal=gn,G=gs+gn,support_upper=max(0.,1-t/runtime)))
    base=os.path.splitext(a.gt)[0]; path=base+'.r1a_modulus.csv'
    with open(path,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    useful=max((r['time'] for r in out if r['support_upper'] >= .1),default=0.)
    def spread(keys):
        # Normalize scatter by the three-channel zero-lag mean, not by C(t):
        # the latter crosses zero and spuriously diverges in noisy long-lag bins.
        scale=abs(sum(out[0][k] for k in keys)/3)
        vals=[math.sqrt(sum((r[k]-sum(r[j] for j in keys)/3)**2 for k in keys)/3) for r in out if r['support_upper'] >= .1]
        return math.sqrt(sum(x*x for x in vals)/len(vals))/(scale+1e-30)
    print('rows=%d formal_max_lag=%g useful_lag_support>=0.1=%g' % (len(out),out[-1]['time'],useful))
    print('useful_lag_rms_channel_scatter_over_C0 shear=%g normal=%g' % (spread(('Cxy','Cxz','Cyz')),spread(('CNxy','CNxz','CNyz'))))
    print('wrote '+path)
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();
        for k in ('Cxy','Cxz','Cyz','CNxy','CNxz','CNyz','G_shear','G_normal','G'): ax.plot([r['time'] for r in out],[r[k] for r in out],label=k)
        ax.set_xscale('log'); ax.set_xlabel('time'); ax.set_ylabel('correlation / G(t)'); ax.legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(base+'.r1a_modulus.png',dpi=160)
    except ImportError: pass
if __name__=='__main__': main()
