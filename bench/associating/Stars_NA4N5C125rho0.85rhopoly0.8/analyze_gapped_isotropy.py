#!/usr/bin/env python3
"""R1-B2.2 weakly correlated, gapped-window stress ACF analysis."""
import argparse,csv,glob
import numpy as np

LENGTH=50000; MAXLAG=25000
SCHEMES={'A':(10000,240000,470000,700000,930000),'B':(110000,320000,530000,740000,950000)}
def load(p):
 x=np.loadtxt(p,comments='#');_,i=np.unique(x[:,0],return_index=True);return x[np.sort(i)]
def ac(x,n):
 m=1<<(2*len(x)-1).bit_length();f=np.fft.rfft(x,m);return np.fft.irfft(f*f.conj(),m)[:n]/np.arange(len(x),len(x)-n,-1)
def one(x):
 s=np.mean([ac(x[:,j],MAXLAG) for j in (3,4,5)],0)
 n=np.mean([ac(x[:,0]-x[:,1],MAXLAG),ac(x[:,0]-x[:,2],MAXLAG),ac(x[:,1]-x[:,2],MAXLAG)],0)/4
 return s,n
def net(pat):
 d={}
 for p in glob.glob(pat):
  step=int(p.rsplit('.',2)[1]);d[step]={(int(q[0]),int(q[1])) for q in (z.split() for z in open(p)) if q and q[0].isdigit()}
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument('raw');p.add_argument('--out',default='r1b22');p.add_argument('--dt',type=float,default=.01);p.add_argument('--network-glob',required=True);a=p.parse_args()
 x=load(a.raw)[:,5:11]; e=net(a.network_glob); summaries=[]
 for name,starts in SCHEMES.items():
  z=np.array([one(x[s:s+LENGTH]) for s in starts]);cs=z[:,0];cn=z[:,1];d=cn-cs
  mean=lambda y:y.mean(0); sem=lambda y:y.std(0,ddof=1)/len(y)**.5
  cm,ce,nm,ne,dm,de=mean(cs),sem(cs),mean(cn),sem(cn),mean(d),sem(d);good=np.abs(cm)>2*ce;ratio=np.full(MAXLAG,np.nan);ratio[good]=nm[good]/cm[good]
  rows=[dict(time=i*a.dt,Cs=cm[i],Cs_sem=ce[i],Cn_over4=nm[i],Cn_over4_sem=ne[i],D=dm[i],D_sem=de[i],R_iso=ratio[i],useful=int(good[i])) for i in range(MAXLAG)]
  with open('%s.%s.csv'%(a.out,name),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
  pairs=[]
  for i,s in enumerate(starts):
   for t in starts[i+1:]: pairs.append(dict(scheme=name,start_step=s,end_step=s+LENGTH,separation_steps=t-s,separation_time=(t-s)*a.dt,Q=len(e[s]&e[t])/len(e[s])))
  with open('%s.%s.windows.csv'%(a.out,name),'w',newline='') as f:w=csv.DictWriter(f,fieldnames=pairs[0]);w.writeheader();w.writerows(pairs)
  sig=np.abs(dm[good]/de[good]);k=np.where(good)[0][np.argmax(sig)]
  summaries.append(dict(scheme=name,window_starts=' '.join(map(str,starts)),R0=ratio[0],useful_to_time=np.where(good)[0][-1]*a.dt,max_abs_D_over_sem=sig.max(),max_time=k*a.dt,max_D=dm[k],frac_within_1=np.mean(sig<=1),frac_within_2=np.mean(sig<=2),mean_pair_Q=np.mean([r['Q'] for r in pairs]),min_pair_Q=np.min([r['Q'] for r in pairs])))
 with open(a.out+'.summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
 try:
  import matplotlib.pyplot as q
  for title,key in (('correlations','Cs'),('difference','D'),('ratio','R_iso')):
   q.figure()
   for s in SCHEMES:
    r=np.genfromtxt('%s.%s.csv'%(a.out,s),delimiter=',',names=True);q.plot(r['time'],r[key],label='gapped '+s)
    if key!='R_iso':q.fill_between(r['time'],r[key]-r[key+'_sem'],r[key]+r[key+'_sem'],alpha=.2)
   if key=='correlations':q.plot(r['time'],r['Cn_over4'],ls='--',label='Cn/4 (B)')
   q.axhline(0,color='k',lw=.5);q.legend();q.xlabel('time');q.savefig(a.out+'.'+title+'.png',dpi=140);q.close()
  q.figure()
  for s in SCHEMES:
   r=np.genfromtxt('%s.%s.csv'%(a.out,s),delimiter=',',names=True);q.plot(r['time'],np.abs(r['D']/r['D_sem']),label='gapped '+s)
  for b in (50,100):
   r=np.genfromtxt(a.raw.replace('.r1b2.raw','.r1b21.%dk.csv'%b),delimiter=',',names=True);q.plot(r['time'],np.abs(r['D']/r['D_sem']),label='contiguous %dk'%b,alpha=.6)
  q.axhline(2,color='k',lw=.5);q.legend();q.xlabel('time');q.ylabel('|D|/SEM');q.savefig(a.out+'.comparison.png',dpi=140);q.close()
 except ImportError:pass
 for r in summaries:print(r)
if __name__=='__main__':main()
