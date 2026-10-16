import sys, json, numpy as np
sys.path.insert(0, "/home/rohit/projects/libribrain-2026/docs/home_stretch/tuning")
from simrule import load, score
D = load()
def per(f):
    def r(b):
        M = b["slp"]; P = np.concatenate(M); Q = f(np.clip(P,1e-12,None)); out=[]; i=0
        for m in M: out.append(Q[i:i+len(m)]); i+=len(m)
        return out
    return r
def ent(P): return -(P*np.log(P)).sum(1)/np.log(P.shape[1])
def pw(P,e): Q=P**e[:,None]; return Q/Q.sum(1,keepdims=True)
def rank(x): return np.argsort(np.argsort(x))/(len(x)-1)
def gap(P): s=np.sort(P,1); return s[:,-1]-s[:,-2]
T={
 "ent_temp_sharpen_lowent(a=0.5)": lambda P: pw(P, 1+0.5*(1-ent(P))),
 "ent_temp_flatten_highent(a=0.5)": lambda P: pw(P, 1-0.5*ent(P)),
 "sharpen_top20pct_conf(x1.5)": lambda P: pw(P, np.where(rank(P.max(1))>=0.8,1.5,1.0)),
 "flatten_bottom20pct_conf(x0.6)": lambda P: pw(P, np.where(rank(P.max(1))<=0.2,0.6,1.0)),
 "sharp_top20_flat_bot20": lambda P: pw(P, np.where(rank(P.max(1))>=0.8,1.5,np.where(rank(P.max(1))<=0.2,0.6,1.0))),
 "margin_rank_power(0.7..1.3)": lambda P: pw(P, 0.7+0.6*rank(gap(P))),
 "rowweight_by_margin_mix_uniform": lambda P: (lambda w: w[:,None]*P+(1-w[:,None])/P.shape[1])(0.5+0.5*rank(gap(P))),
 "top2_gap_small_boost_second(0.5)": lambda P: (lambda s,g: P+np.where((g<0.05)[:,None],0.5*(P==s[:,[-2]])*s[:,[-1]],0))(np.sort(P,1),gap(P)),
 "ent_temp_strong(0.6+0.8*(1-H))": lambda P: pw(P, 0.6+0.8*(1-ent(P))),
 "inverse_flatten_confident(1-0.3*conf_rank)": lambda P: pw(P, 1-0.3*rank(P.max(1))),
}
res=[]
for k,f in T.items():
    s=score(per(f),D); s["idea"]=k; res.append(s); print(json.dumps(s),flush=True)
json.dump(res,open("/home/rohit/projects/libribrain-2026/docs/home_stretch/tuning/rules100/theme6_results.json","w"),indent=1)
