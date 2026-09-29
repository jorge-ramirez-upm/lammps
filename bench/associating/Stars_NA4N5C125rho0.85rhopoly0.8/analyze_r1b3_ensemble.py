#!/usr/bin/env python3
"""R1-B3: replicas, never internal blocks, are the statistical units."""
import argparse,csv,glob,os
import numpy as np
def ac(x):
 n=len(x);m=1<<(2*n-1).bit_length();f=np.fft.rfft(x,m);return np.fft.irfft(f*f.conj(),m)[:n]/np.arange(n,0,-1)
def load(p):
 x=np.loadtxt(p,comments='#');return x[None,:] if x.ndim==1 else x
def main():
 p=argparse.ArgumentParser();p.add_argument('runs',nargs='?',default='r1b3_runs');p.add_argument('--out',default='r1b3');p.add_argument('--dt',type=float,default=.01);a=p.parse_args()
 paths=sorted(glob.glob(os.path.join(a.runs,'replica*/production.raw')));assert len(paths)==6,'need six completed production.raw files'
 z=[];checks=[]
 for path in paths:
  x=load(path);ch=x[:,4:10];aa=np.array([ac(ch[:,i]) for i in range(6)]).T;s=np.mean(aa[:,:3],0);n=np.mean(aa[:,3:],0)/4;z.append((s,n));q=dict(replica=os.path.basename(os.path.dirname(path)),R0=n[0]/s[0])
  gt=path[:-3]+'gt'
  if os.path.exists(gt):
   g=load(gt);lag=np.rint(g[:,0]/a.dt).astype(int);d=aa[lag]-g[:,1:7];q.update(online_max_abs=float(np.abs(d).max()),online_max_rel=float(np.max(np.abs(d/g[:,1:7]))))
  checks.append(q)
 z=np.array(z);cs=z[:,0];cn=z[:,1];d=cn-cs;mean=lambda x:x.mean(0);sd=lambda x:x.std(0,ddof=1);se=lambda x:sd(x)/len(x)**.5
 cm,nm,dm=mean(cs),mean(cn),mean(d);ce,ne,de=se(cs),se(cn),se(d);good=np.abs(cm)>2*ce;r=np.full(len(cm),np.nan);r[good]=nm[good]/cm[good]
 rows=[dict(time=i*a.dt,Cs=cm[i],Cs_sd=sd(cs)[i],Cs_sem=ce[i],Cn_over4=nm[i],Cn_over4_sd=sd(cn)[i],Cn_over4_sem=ne[i],D=dm[i],D_sd=sd(d)[i],D_sem=de[i],R_iso=r[i],useful=int(good[i])) for i in range(len(cm))]
 with open(a.out+'.ensemble.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with open(a.out+'.replicas.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=checks[0]);w.writeheader();w.writerows(checks)
 sig=np.abs(dm[good]/de[good]);i=np.where(good)[0][np.argmax(sig)];summary=dict(replicas=6,R0_mean=float(np.mean([q['R0'] for q in checks])),R0_sd=float(np.std([q['R0'] for q in checks],ddof=1)),max_abs_D_over_sem=float(sig.max()),max_time=i*a.dt,max_D=dm[i],fraction_within_1=float(np.mean(sig<=1)),fraction_within_2=float(np.mean(sig<=2)))
 with open(a.out+'.summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary);w.writeheader();w.writerow(summary)
 print(summary)
if __name__=='__main__':main()
