import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simrule import load, slp, sk, bacc, V
D = load()
def sk2(P, T=1.75, it=10, gamma=1.0, topk=None):
    Q0 = np.clip(P,1e-12,None)**(1/T); Q0 /= Q0.sum(1,keepdims=True); c=np.ones(V); n=len(Q0)
    if topk: thr=np.sort(Q0,1)[:,-topk][:,None]
    for _ in range(it):
        Q=Q0*c; Q/=Q.sum(1,keepdims=True)
        cs=(Q*(Q0>=thr)).sum(0)*(Q.sum()/ (Q*(Q0>=thr)).sum()) if topk else Q.sum(0)
        c*=(n/V)/cs.clip(1e-12)
    Q=Q0*c**gamma; return Q/Q.sum(1,keepdims=True)
def logbal(P,T=1.75):
    L=np.log(np.clip(P,1e-12,None))/T; L=L-L.mean(0,keepdims=True); Q=np.exp(L-L.max(1,keepdims=True)); return Q/Q.sum(1,keepdims=True)
def grouped(b, key):
    M=b["slp"]; keys=[key(m,i) for m in M for i in range(len(m))]; P=np.concatenate(M); keys=np.array(keys); out=np.zeros_like(P)
    for k in np.unique(keys):
        ix=keys==k; out[ix]=sk(P[ix])
    return out
def twostage(b):
    raws=[]; R=np.concatenate(b["raw"]); Rb=sk(R,T=1.0); i=0
    for r in b["raw"]: raws.append(Rb[i:i+len(r)]); i+=len(r)
    return sk(np.concatenate(slp(raws)[0]))
tests = {
 "sk it=1": lambda b: sk(np.concatenate(b["slp"]),it=1),
 "sk it=3": lambda b: sk(np.concatenate(b["slp"]),it=3),
 "sk it=30": lambda b: sk(np.concatenate(b["slp"]),it=30),
 "partial offsets gamma=0.75": lambda b: sk(np.concatenate(b["slp"]),gamma=0.75),
 "over offsets gamma=1.25": lambda b: sk(np.concatenate(b["slp"]),gamma=1.25),
 "log-space column-mean balancing": lambda b: logbal(np.concatenate(b["slp"])),
 "balance on top-20 mass only": lambda b: sk2(np.concatenate(b["slp"]),topk=20),
 "two-stage (raw sk T1 -> SLP -> sk)": twostage,
 "per-sentence-length groups": lambda b: grouped(b, lambda m,i: len(m)),
 "per-position groups (first/mid/last)": lambda b: grouped(b, lambda m,i: 0 if i==0 else (2 if i==len(m)-1 else 1)),
}
res=[]
base=np.array([bacc(sk(np.concatenate(b["slp"])),np.concatenate(b["y"])) for b in D])
for name,f in tests.items():
    new=np.array([bacc(f(b),np.concatenate(b["y"])) for b in D]); d=new-base
    r=dict(idea=name,delta=round(d.mean(),4),se=round(d.std()/np.sqrt(len(d)),4),wins=int((d>0).sum()))
    r["promising"]=bool(r["delta"]>=0.002 and r["delta"]>2*r["se"] and r["wins"]>=16); print(r,flush=True); res.append(r)
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"theme1_results.json"),"w"),indent=1)
