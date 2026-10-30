import numpy as np
from scipy import optimize, stats, special

# ---------- data generation (block GAPHC, adaptive) ----------
def gen_block(rng, n, r, T1, T2, lam, alpha, Rplan):
    atrisk, Y, xs, Rs = n, 0.0, [], []
    for j in range(1, r+1):
        Y += rng.exponential()/atrisk
        x = (1-np.exp(-Y/lam))**(1/alpha)
        if x > T2:                       # Case III: stop at T2
            m=len(xs)
            return dict(x=np.array(xs),R=np.array(Rs,float),tau=T2,C=atrisk,case=3,n=n)
        xs.append(x)
        if j < r:
            Rj = Rplan[j-1] if x < T1 else 0
        else:
            Rj = atrisk-1
        Rj = min(Rj, atrisk-1)
        Rs.append(Rj); atrisk -= 1+Rj
    case = 1 if xs[-1] < T1 else 2
    return dict(x=np.array(xs),R=np.array(Rs,float),tau=xs[-1],C=0,case=case,n=n)

# ---------- likelihood pieces ----------
def Wf(a, b):
    x,R,tau,C = b['x'],b['R'],b['tau'],b['C']
    s = np.sum((1+R)*np.log1p(-x**a))
    if C>0: s += C*np.log1p(-tau**a)
    return -s
def prof_block(a,b):
    """profile-in-lambda loglik of block at alpha a (lambda=m/W)"""
    x=b['x']; m=len(x); W=Wf(a,b)
    return m*np.log(a)+(a-1)*np.sum(np.log(x))-np.sum(np.log1p(-x**a))+m*np.log(m/W)-m
def ell(a, lam, blocks):
    t=0
    for l,b in zip(lam,blocks):
        x=b['x']; m=len(x)
        t+= m*np.log(a)+m*np.log(l)+(a-1)*np.sum(np.log(x))-np.sum(np.log1p(-x**a))-l*Wf(a,b)
    return t

def pava(ratio, w, m):
    # isotonic (increasing) regression of lambda_i=m_i/W_i with weights W_i; pooled = sum m / sum W
    blocks=[[m[i],w[i],[i]] for i in range(len(m))]
    out=[]
    for bl in blocks:
        out.append(bl)
        while len(out)>1 and out[-2][0]/out[-2][1] > out[-1][0]/out[-1][1]:
            b2=out.pop(); b1=out.pop()
            out.append([b1[0]+b2[0],b1[1]+b2[1],b1[2]+b2[2]])
    lam=np.empty(len(m))
    for mm,ww,idx in out: lam[idx]=mm/ww
    return lam

LG = np.linspace(np.log(0.02),np.log(60),600)
def _argmax_log(f):
    g=np.array([f(np.exp(u)) for u in LG])
    g[~np.isfinite(g)]=-np.inf
    i=int(np.argmax(g))
    if i==0 or i==len(LG)-1: return np.exp(LG[i]), True
    r=optimize.minimize_scalar(lambda u:-f(np.exp(u)),bracket=(LG[i-1],LG[i],LG[i+1]),tol=1e-10)
    return float(np.exp(r.x)), False

def mle(blocks, restricted=False):
    m=np.array([len(b['x']) for b in blocks],float)
    def lam_of(a):
        W=np.array([Wf(a,b) for b in blocks]); l=m/W
        return pava(l,W,m) if restricted else l
    f=lambda a: ell(a,lam_of(a),blocks)
    ag=np.exp(LG); Wg=np.column_stack([Wgrid(b,ag) for b in blocks]); lg=m/Wg
    g=np.empty(len(ag))
    for t,a_ in enumerate(ag):
        lt=pava(lg[t],Wg[t],m) if restricted else lg[t]
        g[t]=sum(mm*np.log(a_)+mm*np.log(l_)+(a_-1)*np.log(b['x']).sum()-np.log1p(-b['x']**a_).sum()-l_*Wg[t,i] for i,(mm,l_,b) in enumerate(zip(m,lt,blocks)))
    g[~np.isfinite(g)]=-np.inf; i=int(np.argmax(g))
    if i==0 or i==len(LG)-1: a,bd=ag[i],True
    else:
        r_=optimize.minimize_scalar(lambda u:-f(np.exp(u)),bracket=(LG[i-1],LG[i],LG[i+1]),tol=1e-10); a,bd=float(np.exp(r_.x)),False
    return a, lam_of(a), bd

def fisher_ci(blocks, a, lam, z=1.959964):
    th=np.r_[a,lam]; k=len(lam)
    f=lambda t: ell(t[0],t[1:],blocks)
    H=np.zeros((k+1,k+1)); h=1e-4*np.maximum(1,np.abs(th))
    for i in range(k+1):
        for j in range(i,k+1):
            def v(si,sj):
                t=th.copy(); t[i]+=si*h[i]; t[j]+=sj*h[j]; return f(t)
            H[i,j]=H[j,i]=(v(1,1)-v(1,-1)-v(-1,1)+v(-1,-1))/(4*h[i]*h[j])
    cov=np.linalg.inv(-H); se=np.sqrt(np.diag(cov))
    return th, se, th-z*se, th+z*se, np.exp(np.log(th)-z*se/th), np.exp(np.log(th)+z*se/th)

# ---------- pivotal inference (per block, working assumption) ----------
def pivot_R(b):
    """realized removals with terminal censoring absorbed into last failure (identical to R when C=0)"""
    R=b['R'].copy(); R[-1]+=b['C']; return R
def Wtilde(a,b):
    return -np.sum((1+pivot_R(b))*np.log1p(-b['x']**a))
def lam_ci_known_alpha(b,a,gam=0.05):
    m=len(b['x']); W=Wtilde(a,b)
    return stats.chi2.isf(1-gam/2,2*m)/(2*W), stats.chi2.isf(gam/2,2*m)/(2*W)
def Tfun(a,b):
    x=b['x']; R=pivot_R(b); m=len(x); n=b['n']
    l=np.log1p(-x**a)
    return np.sum((1+R)*l/l[0])/(n*(m-1))-1/(m-1)
def alpha_ci(b,gam=0.05):
    m=len(b['x'])
    if m<2: return (np.nan,np.nan)
    q=[stats.f.isf(1-gam/2,2*m-2,2),stats.f.isf(gam/2,2*m-2,2)]
    grid=np.exp(np.linspace(np.log(1e-3),np.log(1e3),400))
    Tg=np.array([Tfun(a,b) for a in grid])
    sol=[]
    for t in q:
        s=Tg-t; idx=np.where(np.sign(s[:-1])*np.sign(s[1:])<0)[0]
        if len(idx)==0: sol.append(np.nan); continue
        i=idx[0]; sol.append(optimize.brentq(lambda a:Tfun(a,b)-t,grid[i],grid[i+1],xtol=1e-12))
    return tuple(sorted(sol)) if not np.isnan(sol).any() else (np.nan,np.nan)

# ---------- Bayes: collapsed sampler ----------
AG = np.exp(np.linspace(np.log(0.02),np.log(30),4000))
def Wgrid(b,ag):
    x=b['x']; L=np.log1p(-np.exp(np.outer(ag,np.log(x))))
    W=-(L@(1+b['R']))
    if b['C']>0: W-=b['C']*np.log1p(-np.exp(ag*np.log(b['tau'])))
    return W
def post_grid(blocks, a0,b0, ai,bi):
    m=np.array([len(b['x']) for b in blocks]); M=m.sum()
    Wg=np.column_stack([Wgrid(b,AG) for b in blocks])
    lp=(a0+M-1)*np.log(AG)-b0*AG
    for i,b in enumerate(blocks):
        lx=np.log(b['x']); L=np.log1p(-np.exp(np.outer(AG,lx)))
        lp+=(AG-1)*lx.sum()-L.sum(1)
        lp-=(ai[i]+m[i])*np.log(bi[i]+Wg[:,i])
    lp+=np.log(AG)
    w=np.exp(lp-lp.max()); w/=w.sum()
    return w,Wg,m
def bayes_draws(blocks,rng,N=20000,a0=2.,b0=1.,ai=None,bi=None):
    k=len(blocks); ai=ai or [2.]*k; bi=bi or [1.]*k
    w,Wg,m=post_grid(blocks,a0,b0,ai,bi)
    g=rng.choice(len(AG),size=N,p=w)
    al=AG[g]
    lam=np.column_stack([rng.gamma(ai[i]+m[i],1/(bi[i]+Wg[g,i])) for i in range(k)])
    return al,lam
def hpd(s,g=0.05):
    s=np.sort(s); n=len(s); k=int(np.floor((1-g)*n)); w=s[k:]-s[:n-k]; i=int(np.argmin(w)); return s[i],s[i+k]
