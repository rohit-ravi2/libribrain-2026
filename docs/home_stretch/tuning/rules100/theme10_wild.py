import sys, numpy as np; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from simrule import load, sk, bacc, V
D = load()
def lp(P): return np.log(np.clip(P,1e-12,None))
def topk_quota(P, it=30, lr=0.5):
    # adjust log column bias so each class appears in top-10 of n*10/V rows
    L = lp(sk(P)); b = np.zeros(V); n=len(L); tgt = n*10/V
    for _ in range(it):
        S = L + b; top = np.argpartition(-S,10,1)[:,:10]
        cnt = np.bincount(top.ravel(), minlength=V)
        b -= lr*np.log((cnt+1)/(tgt+1))
    return L + b
def soft_quota(P, it=30, tau=0.5):
    # Sinkhorn on soft top-10 indicator expectation
    L = lp(sk(P)); b=np.zeros(V); n=len(L); tgt=n*10/V
    for _ in range(it):
        S=L+b; th=np.sort(S,1)[:,-10:-9]+0  # 10th largest
        th=(np.sort(S,1)[:,-10]+np.sort(S,1)[:,-11])/2
        I=1/(1+np.exp(-(S-th[:,None])/tau)); b -= 0.5*np.log((I.sum(0)+1)/(tgt+1))
    return L+b
def colz(P):
    L=lp(sk(P)); return (L-L.mean(0))/L.std(0)
def rankmix(P):
    Q=sk(P); R=Q.argsort(0).argsort(0)/len(Q); return lp(Q)+0.5*R
def geo(P,raw): return sk(np.sqrt(np.clip(P,1e-12,None)*np.clip(raw,1e-12,None)))
tests = {
 'sk_it50': lambda P,R: sk(P,it=50),
 'sk_gamma1.2': lambda P,R: sk(P,gamma=1.2),
 'sk_gamma0.8': lambda P,R: sk(P,gamma=0.8),
 'top10_hard_quota': lambda P,R: topk_quota(P),
 'top10_hard_quota_lr0.2': lambda P,R: topk_quota(P,lr=0.2),
 'top10_soft_quota_tau0.5': lambda P,R: soft_quota(P),
 'colz_logprob': lambda P,R: colz(P),
 'rank_mix0.5': lambda P,R: rankmix(P),
 'geo_slp_raw_sk': lambda P,R: geo(P,R),
 'sk_then_top10_quota_on_raw_slp_avg': lambda P,R: topk_quota(0.5*P+0.5*R),
}
base=[]; res={k:[] for k in tests}
for b in D:
    y=np.concatenate(b['y']); P=np.concatenate(b['slp']); R=np.concatenate(b['raw'])
    b0=bacc(sk(P),y); base.append(b0)
    for k,f in tests.items(): res[k].append(bacc(f(P,R),y)-b0)
for k,d in res.items():
    d=np.array(d); m,se,w=d.mean(),d.std()/np.sqrt(len(d)),int((d>0).sum())
    print(f"{k}\t{m:+.4f}\t{se:.4f}\t{w}\t{m>=0.002 and m>2*se and w>=16}")
