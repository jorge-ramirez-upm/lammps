#!/usr/bin/env python3
"""R1-B2.1 block-ACF isotropy and network-persistence analysis."""
import argparse, csv, glob, os
import numpy as np

def load_raw(path):
    x=np.loadtxt(path,comments='#'); _,i=np.unique(x[:,0],return_index=True)
    return x[np.sort(i)]

def ac(x,n):
    m=1<<(2*len(x)-1).bit_length(); f=np.fft.rfft(x,m)
    return np.fft.irfft(f*f.conj(),m)[:n]/np.arange(len(x),len(x)-n,-1)

def block_acf(x,block,maxlag):
    out=[]
    for start in range(0,len(x)-block+1,block):
        y=x[start:start+block]
        shear=np.mean([ac(y[:,j],maxlag) for j in (3,4,5)],axis=0)
        normal=np.mean([ac(y[:,0]-y[:,1],maxlag),ac(y[:,0]-y[:,2],maxlag),ac(y[:,1]-y[:,2],maxlag)],axis=0)/4
        out.append((shear,normal))
    return np.array(out)

def stats(a):
    return a.mean(0),a.std(0,ddof=1)/np.sqrt(len(a))

def nets(pattern):
    ans=[]
    for p in glob.glob(pattern):
        step=int(p.rsplit('.',2)[1]); e=set()
        for line in open(p):
            q=line.split()
            if q and q[0].isdigit(): e.add((int(q[0]),int(q[1])))
        ans.append((step,e))
    return sorted(ans)

def persistence(networks):
    rows=[]
    for d in range(1,len(networks)):
        q=[len(networks[i][1]&networks[i+d][1])/len(networks[i][1]) for i in range(len(networks)-d)]
        rows.append((networks[d][0]-networks[0][0],np.mean(q),np.std(q,ddof=1)/np.sqrt(len(q)) if len(q)>1 else np.nan,len(q)))
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('raw'); p.add_argument('--out',default='r1b21')
    p.add_argument('--dt',type=float,default=.01); p.add_argument('--max-lag',type=int,default=25000)
    p.add_argument('--network-glob'); a=p.parse_args(); raw=load_raw(a.raw); x=raw[:,5:11]
    schemes=(50000,100000); summaries=[]
    for block in schemes:
        z=block_acf(x,block,a.max_lag); cs,csse=stats(z[:,0]); cn,cnse=stats(z[:,1]); d=z[:,1]-z[:,0]; dm,dse=stats(d)
        good=np.abs(cs)>2*csse; ratio=np.full(a.max_lag,np.nan); ratio[good]=cn[good]/cs[good]
        rows=[dict(time=i*a.dt,Cs=cs[i],Cs_sem=csse[i],Cn_over4=cn[i],Cn_over4_sem=cnse[i],D=dm[i],D_sem=dse[i],R_iso=ratio[i],useful=int(good[i])) for i in range(a.max_lag)]
        path='%s.%dk.csv'%(a.out,block//1000)
        with open(path,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        sig=np.abs(dm[good]/dse[good]); summaries.append(dict(block_steps=block,blocks=len(z),R0=ratio[0],useful_to_time=(np.where(good)[0][-1]*a.dt if good.any() else np.nan),max_abs_D_over_sem=sig.max() if len(sig) else np.nan,frac_within_1=float(np.mean(sig<=1)) if len(sig) else np.nan,frac_within_2=float(np.mean(sig<=2)) if len(sig) else np.nan))
    if a.network_glob:
        pr=persistence(nets(a.network_glob));
        with open(a.out+'.persistence.csv','w',newline='') as f:
            w=csv.writer(f);w.writerow(('lag_steps','lag_time','Q','Q_sem','pairs'));w.writerows((s,s*a.dt,q,e,n) for s,q,e,n in pr)
        for threshold in (.9,.75,.5):
            hit=next((s*a.dt for s,q,_,_ in pr if q<threshold),np.nan); summaries[0]['Q_below_%g'%threshold]=hit
    with open(a.out+'.summary.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=sorted({k for r in summaries for k in r}));w.writeheader();w.writerows(summaries)
    try:
        import matplotlib.pyplot as q
        for name,key in (('correlations','Cs'),('difference','D'),('ratio','R_iso')):
            q.figure()
            for block in schemes:
                r=np.genfromtxt('%s.%dk.csv'%(a.out,block//1000),delimiter=',',names=True); t=r['time']; y=r[key]
                q.plot(t,y,label='%dk'% (block//1000))
                if key!='R_iso': q.fill_between(t,y-r[key+'_sem'],y+r[key+'_sem'],alpha=.2)
            if key=='correlations':
                r=np.genfromtxt('%s.50k.csv'%a.out,delimiter=',',names=True);q.plot(r['time'],r['Cn_over4'],label='Cn/4 (50k)',ls='--')
            q.axhline(0,color='k',lw=.5);q.legend();q.xlabel('time');q.savefig(a.out+'.'+name+'.png',dpi=140);q.close()
        if a.network_glob:
            r=np.genfromtxt(a.out+'.persistence.csv',delimiter=',',names=True);q.figure();q.plot(r['lag_time'],r['Q']);q.axhline(.9,color='k',lw=.5);q.axhline(.75,color='k',lw=.5);q.axhline(.5,color='k',lw=.5);q.xlabel('lag time');q.ylabel('Q');q.savefig(a.out+'.persistence.png',dpi=140);q.close()
    except ImportError: pass
    for r in summaries: print(r)
if __name__=='__main__': main()
