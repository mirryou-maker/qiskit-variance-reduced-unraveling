import numpy as np
from scipy.optimize import brentq
def vs(t): return 1-np.exp(-4*t)
def vp(t): return np.exp(-2*t)*(1-np.exp(-2*t))
def va(t,th): return (1+np.exp(-8*t*np.cos(th)**2))/2-np.exp(-4*t)
th=0.3
print("gt  proj/std  analog/std(th=.3)")
for t in [0.1,0.2,0.25,0.35,0.5,0.7,1.0,1.5]:
    print(t, round(vp(t)/vs(t),4), round(va(t,th)/vs(t),4))
f=lambda t,th: va(t,th)/vs(t)-vp(t)/vs(t)
for th in [0.05,0.1,0.2,0.3,0.5,0.7]:
    try: print("theta0",th,"crossover gt*",round(brentq(lambda t:f(t,th),1e-4,5),4))
    except Exception as e: print(th,e)
print("exact speedup gt=1.5:", vs(1.5)/vp(1.5))
for n in [8,10,12,16,20]:
    print(n,"N_std exp",round(vs(1.5)/n/1e-4,1),"N_proj exp",round(vp(1.5)/n/1e-4,1))
# GHZ: m=exp(-2 n gt)
for nt in [0.5,1.0,1.5]:
    m=np.exp(-2*nt); print("GHZ ngt",nt,"speedup exact",(1-m*m)/(m*(1-m)))
# sampling distribution of speedup estimator, a3c protocol N=600 both, and a4b protocol (Aer 20000 shots binary vs proj N=600)
rng=np.random.default_rng(0); q=np.exp(-3)
for n in [8,12,16,20]:
    sp=[]; sp2=[]
    for r in range(4000):
        # std: per-qubit +-1 with P(+)=(1+q)/2
        s=(2*(rng.random((600,n))<(1+q)/2)-1).mean(1)
        p=(rng.random((600,n))<q).mean(1)
        sp.append(s.var(ddof=1)/p.var(ddof=1))
    sp=np.array(sp); print("n",n,"N=600 speedup est: median",np.median(sp).round(2),"95%CI",np.percentile(sp,[2.5,97.5]).round(1))
# GHZ c1 gt=0.15 n=10 : Bernoulli(m) N=2000
m=np.exp(-3); sp=[]
for r in range(4000):
    s=2*(rng.random(2000)<(1+m)/2)-1; p=(rng.random(2000)<m)*1.0
    sp.append(s.var()/p.var())
print("GHZ gt.15 N=2000 speedup 95%CI",np.percentile(sp,[2.5,50,97.5]).round(1), "P(>=27.2)",np.mean(np.array(sp)>=27.2))
# erosion exact mean
def bloch_run(theta,tau=0.5):
    x,z=1.0,0.0
    for w in range(3):
        x*=np.exp(-2*tau)
        if w<2 and theta!=0:
            # RY(theta): x' = x cos + z sin ; z' = -x sin + z cos
            x,z = x*np.cos(theta)+z*np.sin(theta), -x*np.sin(theta)+z*np.cos(theta)
    return x
for th in [0,0.3,0.6,1.0,np.pi/2]: print("erosion theta",round(th,3),"exact mean <X>",round(bloch_run(th),4))
