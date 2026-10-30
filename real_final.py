import numpy as np, pandas as pd, math, json, warnings
warnings.filterwarnings('ignore')
from kum import *
df=pd.read_csv('/home/claude/case0802.csv')
V=[30,32,34,36]
T={v:np.sort(df[df.Voltage==v].Time.values) for v in V}
C0,T2m=250.0,100.0
def make(c=C0,T2m=T2m):
    bl=[]
    for v in V:
        t=T[v]; n=len(t); r=math.ceil(0.7*n); x=t/c
        if t[r-1]<T2m: bl.append(dict(x=x[:r],R=np.r_[np.zeros(r-1),n-r],tau=x[r-1],C=0,case=2,n=n))
        else:
            k2=int((t<=T2m).sum()); bl.append(dict(x=x[:k2],R=np.zeros(k2),tau=T2m/c,C=n-k2,case=3,n=n))
    return bl
bl=make()
res={}
print('m',[len(b['x']) for b in bl],'case',[b['case'] for b in bl],'C',[b['C'] for b in bl])
a,l,bd=mle(bl); ar,lr,_=mle(bl,True)
print('MLE',a,l,'\nRMLE',ar,lr)
th,se,lo,hi,llo,lhi=fisher_ci(bl,a,l)
print('Fisher normal'); [print(round(th[i],3),round(lo[i],3),round(hi[i],3),' log:',round(llo[i],3),round(lhi[i],3)) for i in range(5)]
print('pivot alpha CI'); 
for b in bl: print(len(b['x']),b['case'],np.round(alpha_ci(b),3))
print('pivot lambda CI at alpha-hat (plug-in)')
for b in bl: print(np.round(lam_ci_known_alpha(b,a),3))
# LRT common alpha
ll_c=ell(a,l,bl)
ll_s=0; ahat_i=[]
for b in bl:
    ai,_,_ = mle([b]); ahat_i.append(ai); ll_s+=prof_block(ai,b)
LR=2*(ll_s-ll_c); print('block alphas',np.round(ahat_i,3),'LR',LR,'p',stats.chi2.sf(LR,3))
# GOF: normalized spacings
def spacings(blocks,a,lam):
    Z=[]
    for b,l in zip(blocks,lam):
        x,R,n=b['x'],b['R'],b['n']; Y=-l*np.log1p(-x**a); gam=n-np.arange(len(x))-np.r_[0,np.cumsum(R)[:-1]]
        Z.append(gam*np.diff(np.r_[0,Y]))
    return np.concatenate(Z)
Z=spacings(bl,a,l); ks=stats.kstest(Z,'expon').statistic; print('KS spacings',ks)
rng=np.random.default_rng(5); cnt=0; B=300; nb=0
for _ in range(B):
    nbl=[]
    for b,lm in zip(bl,l):
        n=b['n']; r=math.ceil(0.7*n); Rp=np.r_[np.zeros(r-1),n-r]
        nbl.append(gen_block(rng,n,r,0.2,T2m/C0,lm,a,Rp))
    if min(len(b['x']) for b in nbl)<2: continue
    ab,lb,_=mle(nbl); Zb=spacings(nbl,ab,lb); nb+=1; cnt+= stats.kstest(Zb,'expon').statistic>=ks
print('GOF bootstrap p',cnt/nb,nb)
# Bayes
rng=np.random.default_rng(11)
out={}
for nm,(a0,b0,ai,bi) in {'G(1,1)':(1,1,1,1),'G(2,1)':(2,1,2,1),'G(0.1,0.1)':(.1,.1,.1,.1),'G(1,0.1)':(1,.1,1,.1)}.items():
    al,lam=bayes_draws(bl,rng,N=1000000,a0=a0,b0=b0,ai=[ai]*4,bi=[bi]*4)
    ok=np.all(np.diff(lam,axis=1)>=0,axis=1)
    print(nm,'P(order)',round(ok.mean(),4))
    for lab,sel in [('unres',slice(None)),('res',ok)]:
        A=al[sel]; L=lam[sel]
        print(' ',lab,'alpha',round(A.mean(),3),np.round(hpd(A[:400000]),3),'lam',np.round(L.mean(0),3),[tuple(np.round(hpd(L[:400000,i] if lab=='unres' else L[:min(len(L),400000),i]),2)) for i in range(4)])
    pr=np.mean(lam[:,2]<=lam[:,3]); print('  P(l3<=l4)',round(pr,3),'P(l1<=l2)',round(np.mean(lam[:,0]<=lam[:,1]),3),'P(l2<=l3)',round(np.mean(lam[:,1]<=lam[:,2]),3))
# sensitivity to divisor
for c in [250,500,1000]:
    b2=make(c); a2,l2,bd2=mle(b2); print('c',c,round(a2,3),np.round(l2,2),bd2)
