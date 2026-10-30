import numpy as np, sys, json, time, warnings
warnings.filterwarnings('ignore')
from kum import *
n,r=60,40
Rplan=np.r_[np.ones(10),np.zeros(r-11),0]   # planned: 1 unit removed at each of first 10 failures; rest at r-th
Rplan[-1]=n-r-10
SET={'A':dict(lam=[2.0,4.0],T1=[0.20,0.15],T2=[0.60,0.50]),'B':dict(lam=[2.5,3.0],T1=[0.20,0.20],T2=[0.60,0.60])}
ALPHA=2.0
PRIORS={'Gamma(2,1)':(2,1),'Gamma(1,1)':(1,1),'Gamma(0.1,0.1)':(.1,.1),'Gamma(4,2)':(4,2),'Gamma(8,2)':(8,2)}
def rep(seed,S,do_bayes):
    rng=np.random.default_rng(seed); s=SET[S]
    bl=[gen_block(rng,n,r,s['T1'][i],s['T2'][i],s['lam'][i],ALPHA,Rplan) for i in range(2)]
    if min(len(b['x']) for b in bl)<2: return None
    o=dict(case=[b['case'] for b in bl],m=[len(b['x']) for b in bl])
    a,l,bd=mle(bl); ar,lr,_=mle(bl,True)
    o['mle']=[a,*l]; o['rmle']=[ar,*lr]; o['pooled']=bool(l[0]>l[1]); o['bd']=bd
    th,se,lo,hi,llo,lhi=fisher_ci(bl,a,l)
    tr=np.r_[ALPHA,s['lam']]
    o['fi_norm']=[bool(lo[j]<=tr[j]<=hi[j]) for j in range(3)]; o['fi_log']=[bool(llo[j]<=tr[j]<=lhi[j]) for j in range(3)]
    o['fi_len']=[float(hi[j]-lo[j]) for j in range(3)]; o['fi_loglen']=[float(lhi[j]-llo[j]) for j in range(3)]
    pa=[alpha_ci(b) for b in bl]
    o['pv_a']=[(bool(p[0]<=ALPHA<=p[1]) if not np.isnan(p[0]) else None) for p in pa]; o['pv_a_len']=[float(p[1]-p[0]) for p in pa]
    pl=[lam_ci_known_alpha(b,ALPHA) for b in bl]
    o['pv_l']=[bool(pl[i][0]<=s['lam'][i]<=pl[i][1]) for i in range(2)]
    if do_bayes:
        o['bayes']={}
        for nm,(a0,b0) in PRIORS.items():
            if not do_bayes=='all' and nm!='Gamma(2,1)': continue
            al,lam=bayes_draws(bl,rng,N=20000,a0=a0,b0=b0,ai=[a0]*2,bi=[b0]*2)
            ok=lam[:,0]<=lam[:,1]
            d={'u':[al.mean(),*lam.mean(0)],'porder':float(ok.mean())}
            if ok.sum()>500: d['r']=[al[ok].mean(),*lam[ok].mean(0)]
            if nm=='Gamma(2,1)':
                d['u_cov']=[bool(hpd(v)[0]<=t<=hpd(v)[1]) for v,t in zip([al,lam[:,0],lam[:,1]],tr)]
                if ok.sum()>500: d['r_cov']=[bool(hpd(v)[0]<=t<=hpd(v)[1]) for v,t in zip([al[ok],lam[ok,0],lam[ok,1]],tr)]
            o['bayes'][nm]=d
    return o
if __name__=='__main__':
    S,B,bay=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    bay={'0':False,'1':True,'all':'all'}[bay]
    t0=time.time(); out=[]; fail=0
    for i in range(B):
        o=rep(1000+i,S,bay)
        if o is None: fail+=1
        else: out.append(o)
    json.dump(dict(S=S,fail=fail,res=out),open(f'sim_{S}_{sys.argv[3]}.json','w'))
    print(S,B,'done',time.time()-t0)
