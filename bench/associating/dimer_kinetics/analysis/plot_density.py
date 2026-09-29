#!/usr/bin/env python3
"""Optional K1-D plots; numerical analysis does not require matplotlib."""
import csv, sys
try: import matplotlib.pyplot as plt
except ImportError: sys.exit('matplotlib is optional; install it to make plots')
rows=list(csv.DictReader(open(sys.argv[1])))
for series, x, y in [('activation','Ea','kf_over_q'),('binding','Ee','Keq_eq')]:
    plt.figure()
    for rho in sorted({r['rho'] for r in rows},key=float):
        q=[r for r in rows if r['rho']==rho and ((series=='activation' and r['Ee']=='4.0') or (series=='binding' and r['Ea']=='4.0')) and r['Nevery']=='100']
        plt.plot([float(r[x]) for r in q],[float(r[y]) for r in q],'o-',label='rho='+rho)
    plt.xlabel(x); plt.ylabel(y); plt.yscale('log'); plt.legend(); plt.tight_layout(); plt.savefig(sys.argv[1]+'_'+series+'.png',dpi=160)
