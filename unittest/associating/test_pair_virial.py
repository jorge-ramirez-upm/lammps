#!/usr/bin/env python3
"""Tensor regression: pair associating must tally all pressure virial terms."""
import math, re, subprocess, sys
lmp=sys.argv[1]; r=(.30,.40,-.50); k=30.; r0=1.5; volume=1000.
deck=f'''units lj
atom_style atomic
atom_modify map yes
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 4 5 6
create_atoms 1 single {4+r[0]} {5+r[1]} {6+r[2]}
mass 1 1
pair_style associating
pair_coeff * * {k} {r0} 1
fix a all associating/kinetics debug_pair 1 2
compute ptotal all pressure NULL virial
compute passoc all pressure NULL pair
thermo_style custom step c_ptotal[1] c_ptotal[2] c_ptotal[3] c_ptotal[4] c_ptotal[5] c_ptotal[6] c_passoc[1] c_passoc[2] c_passoc[3] c_passoc[4] c_passoc[5] c_passoc[6]
thermo_modify norm no
run 0
'''
p=subprocess.run([lmp,'-log','none'],input=deck,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
assert p.returncode==0,p.stdout
line=next(x for x in p.stdout.splitlines() if re.match(r'^\s*0\s',x)); fields=[float(x)*volume for x in line.split()[1:13]]
got,component=fields[:6],fields[6:]
fac=-k/(1-sum(x*x for x in r)/r0**2); force=[fac*x for x in r]
expected=[r[0]*force[0],r[1]*force[1],r[2]*force[2],r[0]*force[1],r[0]*force[2],r[1]*force[2]]
for label,a,b in zip(('xx','yy','zz','xy','xz','yz'),got,expected):
    assert math.isclose(a,b,rel_tol=2e-6,abs_tol=2e-6),(label,a,b,p.stdout)
for label,a,b in zip(('xx','yy','zz','xy','xz','yz'),got,component):
    assert math.isclose(a,b,rel_tol=0.,abs_tol=1e-12),(label,a,b,p.stdout)
