import sys,os,json; H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,H); os.chdir(H)
import numpy as np
from simrule import load, score
D=load(); E=1e-12
def norm(m): return m/m.sum(1,keepdims=True)
def ent(m): return -(m*np.log(m+E)).sum(1,keepdims=True)/np.log(m.shape[1])
def arith(w): return lambda b:[norm((1-w)*s+w*r) for r,s in zip(b['raw'],b['slp'])]
def geo(w): return lambda b:[norm(np.exp((1-w)*np.log(s+E)+w*np.log(r+E))) for r,s in zip(b['raw'],b['slp'])]
def entmix(a): return lambda b:[norm((1-a*ent(s))*s+a*ent(s)*r) for r,s in zip(b['raw'],b['slp'])]
def lenmix(c): return lambda b:[norm((1-c/len(s))*s+(c/len(s))*r) if len(s)>c else r for r,s in zip(b['raw'],b['slp'])]
def pmean(p,w=0.3): return lambda b:[norm(((1-w)*s**p+w*r**p)**(1/p)) for r,s in zip(b['raw'],b['slp'])]
T={'arith0.1':arith(.1),'arith0.3':arith(.3),'arith0.5':arith(.5),'geo0.1':geo(.1),'geo0.3':geo(.3),'geo0.5':geo(.5),
 'entmix0.5':entmix(.5),'lenmix1':lenmix(1),'pmean0.5':pmean(.5),'pmean2':pmean(2)}
R=[]
for k,f in T.items():
    s=score(f,D); s['idea']=k; R.append(s); print(s,flush=True)
json.dump(R,open('rules100/theme3_mix.json','w'))
