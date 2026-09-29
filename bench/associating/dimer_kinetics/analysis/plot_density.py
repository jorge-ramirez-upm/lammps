#!/usr/bin/env python3
"""Optional K1-E error-bar plots. Input is *_conditions.csv."""
import csv, math, os, sys
try: import matplotlib.pyplot as plt
except ImportError: sys.exit('matplotlib is optional; install it to make plots')
rows=list(csv.DictReader(open(sys.argv[1]))); trends=list(csv.DictReader(open(sys.argv[1].replace('_conditions.csv','_trends.csv'))))
def pick(rho, kind): return sorted([r for r in rows if r['rho']==rho and r['Nevery']=='100' and ((kind=='a' and r['Ee']=='4.0') or (kind=='b' and r['Ea']=='4.0'))],key=lambda r:float(r['Ea' if kind=='a' else 'Ee']))
def errplot(kind,key,label,log=False,overlay_q=False):
    fig,ax=plt.subplots()
    for rho in sorted({r['rho'] for r in rows},key=float):
        q=pick(rho,kind); x=[float(r['Ea' if kind=='a' else 'Ee']) for r in q]; y=[float(r[key]) for r in q]; e=[float(r[key+'_se']) for r in q]
        ax.errorbar(x,y,yerr=e,fmt='o',capsize=3,label='rho='+rho)
        if overlay_q:
            c=sum(math.log(v)-math.log(float(r['q'])) for r,v in zip(q,y))/len(q); xx=[min(x)+i*(max(x)-min(x))/100 for i in range(101)]
            ax.plot(xx,[math.exp(c)*(-math.expm1(-20*math.exp(-z)*.5)) for z in xx],'-',alpha=.7)
    ax.set(xlabel='Ea' if kind=='a' else 'Ee',ylabel=label); ax.set_yscale('log' if log else 'linear'); ax.legend(); fig.tight_layout(); fig.savefig(sys.argv[1]+'_'+key+'.png',dpi=160)
errplot('a','kf_event','kf event',True,True)
errplot('a','kf_event_over_q','kf event / q',True)
errplot('a','kb_event_over_q','kb event / q',True)
fig,ax=plt.subplots()
for rho in sorted({r['rho'] for r in rows},key=float):
    q=pick(rho,'b'); x=[float(r['Ee']) for r in q]
    for key,mark in [('Keq_event','o'),('Keq_eq','s')]: ax.errorbar(x,[float(r[key]) for r in q],yerr=[float(r[key+'_se']) for r in q],fmt=mark,capsize=3,label=key+' rho='+rho)
    for t in trends:
        if t['rho']==rho and t['series']=='binding_lnKeq_event':
            xx=[min(x),max(x)]; ax.plot(xx,[math.exp(float(t['intercept'])+float(t['slope'])*z) for z in xx],'--',alpha=.7)
ax.set(xlabel='Ee',ylabel='Keq',yscale='log'); ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(sys.argv[1]+'_equilibrium.png',dpi=160)
