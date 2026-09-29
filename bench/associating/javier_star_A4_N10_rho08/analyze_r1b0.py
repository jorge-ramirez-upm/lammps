#!/usr/bin/env python3
"""R1-B0 offline stress autocorrelation, isotropy, and online comparison."""
import argparse, csv, os
import numpy as np

NAMES=('xy','xz','yz','nxy','nxz','nyz')
def table(path):
    a=np.loadtxt(path,comments='#'); return a[None,:] if a.ndim==1 else a
def acf(x):
    n=len(x); size=1<<(2*n-1).bit_length(); f=np.fft.rfft(x,size); return np.fft.irfft(f*f.conj(),size)[:n]/np.arange(n,0,-1)
def main():
    p=argparse.ArgumentParser(); p.add_argument('raw'); p.add_argument('gt'); p.add_argument('--dt',type=float,default=.01); p.add_argument('--volume',type=float,default=51250.5882353); p.add_argument('--temperature',type=float,default=1.); p.add_argument('--early-time',type=float,default=.16); p.add_argument('--ratio-floor',type=float,default=.05); p.add_argument('--out-prefix'); a=p.parse_args()
    raw=table(a.raw); online=table(a.gt); prefix=a.out_prefix or os.path.splitext(a.raw)[0]
    if raw.shape[1]<10 or online.shape[1]<7: raise SystemExit('expected raw step+9 fields and gt time+6 fields')
    means=raw[:,1:7].mean(0); series=raw[:,4:10]; offline=np.array([acf(series[:,i]) for i in range(6)]).T
    lag=np.rint(online[:,0]/a.dt).astype(int); valid=lag<len(offline); off=offline[lag[valid]]; on=online[valid,1:7]; early=online[valid,0]<=a.early_time
    rows=[]; metrics=[]
    for i,name in enumerate(NAMES):
        d=on[:,i]-off[:,i]; signal=np.abs(off[:,i])>=.05*abs(off[0,i]); select=early & signal
        metrics.append(dict(channel=name,max_abs=float(np.max(abs(d[early]))),rms=float(np.sqrt(np.mean(d[early]**2))),max_rel=float(np.max(abs(d[select]/off[select,i]))) if np.any(select) else float('nan'),n_early=int(early.sum())))
    shear=on[:,:3].mean(1); normal=on[:,3:].mean(1); ratio=np.full(len(shear),np.nan); good=np.abs(shear)>=a.ratio_floor*abs(shear[0]); ratio[good]=normal[good]/(4*shear[good])
    gs=a.volume*on[:,:3].sum(1)/(5*a.temperature); gn=a.volume*on[:,3:].sum(1)/(30*a.temperature)
    for i,t in enumerate(online[valid,0]): rows.append(dict(time=t,**{f'online_{NAMES[j]}':on[i,j] for j in range(6)},**{f'offline_{NAMES[j]}':off[i,j] for j in range(6)},shear_mean=shear[i],normal_mean_over4=normal[i]/4,R_iso=ratio[i],G_shear=gs[i],G_normal=gn[i],G=gs[i]+gn[i]))
    with open(prefix+'.comparison.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    with open(prefix+'.metrics.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    print('normal_means pxx=%g pyy=%g pzz=%g; shear_means pxy=%g pxz=%g pyz=%g' % tuple(means))
    print('R_iso(t=0)=%g; useful ratio mean=%g std=%g n=%d' % (ratio[0],np.nanmean(ratio[good]),np.nanstd(ratio[good]),good.sum()))
    for m in metrics: print('{channel}: max_abs={max_abs:.4g} rms={rms:.4g} max_rel={max_rel:.4g} n={n_early}'.format(**m))
    try:
        import matplotlib.pyplot as plt
        t=online[valid,0]; pos=t>0
        def save(name, ys, labels, log=True):
            fig,ax=plt.subplots(); [ax.plot(t,y,label=l) for y,l in zip(ys,labels)];
            if log: ax.set_xscale('log')
            ax.legend();ax.set_xlabel('time');fig.tight_layout();fig.savefig(prefix+'.'+name+'.png',dpi=160)
        save('shear',[on[:,i] for i in range(3)],NAMES[:3]); save('normal_over4',[on[:,i]/4 for i in range(3,6)],NAMES[3:]); save('orientation',[shear,normal/4],['shear mean','normal mean / 4']); save('modulus',[gs,gn,gs+gn],['G_shear','G_normal','G'])
        save('online_offline',[on[:,0],off[:,0],on[:,3],off[:,3]],['online xy','offline xy','online nxy','offline nxy'])
    except ImportError: pass
if __name__=='__main__': main()
