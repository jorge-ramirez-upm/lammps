#!/usr/bin/env python3
"""Replica-uncertainty R1-B1 online/offline and isotropy analysis."""
import argparse,csv,glob,os
import numpy as np
N=('xy','xz','yz','nxy','nxz','nyz')
def load(p):
 a=np.loadtxt(p,comments='#');return a[None,:] if a.ndim==1 else a
def ac(x):
 n=len(x);m=1<<(2*n-1).bit_length();f=np.fft.rfft(x,m);return np.fft.irfft(f*f.conj(),m)[:n]/np.arange(n,0,-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('raw',nargs='+');p.add_argument('--dt',type=float,default=.01);p.add_argument('--volume',type=float,default=976.1176470588);p.add_argument('--out',default='r1b1');a=p.parse_args()
 reps=[]
 for raw in sum((glob.glob(x) for x in a.raw),[]):
  gt=raw[:-4]+'.gt';r=load(raw);g=load(gt);lag=np.rint(g[:,0]/a.dt).astype(int);off=np.array([ac(r[:,4+i]) for i in range(6)]).T[lag];on=g[:,1:7]
  reps.append((raw,r,on,off,g[:,0]))
 t=reps[0][4];on=np.array([x[2] for x in reps]);off=np.array([x[3] for x in reps]); cs=on[:,:,:3].mean(2);cn=on[:,:,3:].mean(2)/4;d=cn-cs
 mean=lambda x:x.mean(0);sem=lambda x:x.std(0,ddof=1)/np.sqrt(len(x))
 ms,mn,md=mean(cs),mean(cn),mean(d);ss,sn,sd=sem(cs),sem(cn),sem(d); good=np.abs(ms)>2*ss; ratio=np.full(len(t),np.nan);ratio[good]=mn[good]/ms[good]
 rows=[dict(time=t[i],shear=ms[i],shear_sem=ss[i],normal_over4=mn[i],normal_sem=sn[i],D=md[i],D_sem=sd[i],R_iso=ratio[i]) for i in range(len(t))]
 with open(a.out+'.isotropy.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 metrics=[]
 for raw,r,o,fv,_ in reps:
  early=t<=.16;dif=o-fv;metrics.append(dict(replica=raw,max_abs=np.max(abs(dif[early])),rms=np.sqrt(np.mean(dif[early]**2)),max_rel=np.max(abs(dif[early]/fv[early]))))
 with open(a.out+'.online_offline.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=metrics[0]);w.writeheader();w.writerows(metrics)
 print('replicas=%d Riso0=%g useful_D_over_SE_max=%g useful_points=%d'%(len(reps),ratio[0],np.max(abs(md[good]/sd[good])),good.sum()))
 for m in metrics:print('{replica} max_abs={max_abs:g} rms={rms:g} max_rel={max_rel:g}'.format(**m))
 try:
  import matplotlib.pyplot as q
  def plot(name,y,e=0):
   q.figure();q.plot(t,y);e != 0 and q.fill_between(t,y-e,y+e,alpha=.25);q.xscale('log');q.axhline(0,color='k');q.savefig(a.out+'.'+name+'.png',dpi=140);q.close()
  plot('isotropy',md,sd);plot('ratio',ratio);plot('modulus',a.volume*(3*ms)/(5)+a.volume*(12*mn)/(30))
 except ImportError:pass
if __name__=='__main__':main()
