
#!/usr/bin/env python3
import sys,os,time,json,colorsys,warnings; warnings.filterwarnings('ignore')
from pathlib import Path; from collections import defaultdict
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F,torch.optim as optim
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt,matplotlib.patches as mpatches
from sklearn.manifold import TSNE; from sklearn.metrics import confusion_matrix

sys.path.insert(0,'/home/claude/pac_net')

# ── TinyCNN backbone (no download, fast CPU ~115ms/batch) ──────────────────
class TinyCNN(nn.Module):
    def __init__(self,frozen=False):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.out_dim=128; self.frozen=frozen
        if frozen:
            for p in self.parameters(): p.requires_grad_(False)
    def forward(self,x):
        ctx=torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx: return self.net(x).flatten(1)

import pac_net.models.backbone as _bb, pac_net.models.pac_net as _pm
_bb.build_backbone=lambda name='tiny',frozen=True: TinyCNN(frozen=False)
_pm.build_backbone=lambda name='tiny',frozen=True: TinyCNN(frozen=False)

from pac_net.models.pac_net       import PACNet,PACNetConfig
from pac_net.losses.combined_loss import PACNetLoss
from pac_net.metrics.calibration  import calibration_report,reliability_data,ece as _ece
from pac_net.metrics.ood          import ood_report,density_score,msp_score,energy_score
from pac_net.metrics.statistical  import classification_report_full,shot_breakdown,wilcoxon_holm
from pac_net.utils.seed           import set_seed

OUT=Path('/home/claude/demo_results'); (OUT/'figures').mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cpu'); SEED=42; set_seed(SEED)
PAL=['#0077BB','#33BBEE','#009988','#EE7733','#CC3311','#EE3377','#BBBBBB','#44BB99']

print("="*65)
print("  PAC-Net  Prototype-Anchored Calibration Network")
print("  Full Pipeline Demo  |  Synthetic Medical Imaging Data")
print("="*65)

# ── Synthetic dataset ──────────────────────────────────────────────────────
class SynMed(Dataset):
    def __init__(self,cc,img=32,hoff=0.0,seed=42):
        rng=np.random.default_rng(seed); C=len(cc); imgs,lbls=[],[]
        for c,n in enumerate(cc):
            r,g,b=colorsys.hsv_to_rgb((c/C+hoff)%1.,0.75,0.85)
            for _ in range(n):
                im=rng.normal([[[r]],[[g]],[[b]]],0.06,(3,img,img)).astype(np.float32)
                x=np.linspace(0,2*np.pi*(c+1),img,dtype=np.float32)
                im[0]+=0.05*np.sin(x)[None,:]; imgs.append(torch.tensor(im.clip(0,1))); lbls.append(c)
        p=rng.permutation(len(imgs)); self.images=[imgs[i] for i in p]; self.labels=[lbls[i] for i in p]
        self.class_counts=list(cc); self.num_classes=C
    def __len__(self): return len(self.images)
    def __getitem__(self,i): return self.images[i],self.labels[i]

class Spl(Dataset):
    def __init__(self,par,idx):
        self.p=par; self.idx=idx
        cc=[0]*par.num_classes
        for i in idx: cc[par.labels[i]]+=1
        self.class_counts=cc; self.num_classes=par.num_classes
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        j=self.idx[i]; return self.p.images[j],self.p.labels[j]

def strat_split(ds,vr=.15,tr=.15,seed=42):
    rng=np.random.default_rng(seed); by=defaultdict(list)
    for i in range(len(ds)): by[ds.labels[i]].append(i)
    T,V,E=[],[],[]
    for idxs in by.values():
        a=np.array(idxs); rng.shuffle(a)
        nt,nv=max(1,int(len(a)*tr)),max(1,int(len(a)*vr))
        E.extend(a[:nt]); V.extend(a[nt:nt+nv]); T.extend(a[nt+nv:])
    return T,V,E

def wld(ds,bs=64,samp=False,shuf=False):
    s=None
    if samp:
        wts=1./torch.tensor(ds.class_counts,dtype=torch.float)
        sw=torch.tensor([wts[ds.p.labels[i]] for i in ds.idx])
        s=WeightedRandomSampler(sw,len(sw),replacement=True)
    return DataLoader(ds,batch_size=bs,sampler=s,shuffle=(shuf and s is None),drop_last=False)

HAM_C=[10,15,25,10,30,200,10]; HAM_N=['akiec','bcc','bkl','df','mel','nv','vasc']
APT_C=[100,30,60,20,25];       APT_N=['No_DR','Mild','Moderate','Severe','Prolif_DR']
OCT_C=[200,120,90,250];        OCT_N=['CNV','DME','DRUSEN','NORMAL']

print("\n[Data] Generating synthetic datasets...")
ham=SynMed(HAM_C,hoff=0.0,seed=SEED); apt=SynMed(APT_C,hoff=0.25,seed=SEED); oct=SynMed(OCT_C,hoff=0.52,seed=SEED)

ti_h,vi_h,ei_h=strat_split(ham,seed=SEED)
ti_a,vi_a,ei_a=strat_split(apt,seed=SEED)
_,_,ei_o=strat_split(oct,seed=SEED)

ham_tr,ham_vl,ham_te=Spl(ham,ti_h),Spl(ham,vi_h),Spl(ham,ei_h)
apt_tr,apt_vl,apt_te=Spl(apt,ti_a),Spl(apt,vi_a),Spl(apt,ei_a)
oct_te=Spl(oct,ei_o)

ham_trl=wld(ham_tr,samp=True); ham_vll=wld(ham_vl); ham_tel=wld(ham_te)
oct_tel=wld(oct_te); apt_trl=wld(apt_tr,samp=True); apt_vll=wld(apt_vl); apt_tel=wld(apt_te)

print(f"  HAM-like   train={len(ham_tr)} val={len(ham_vl)} test={len(ham_te)}  IR={max(HAM_C)/min(HAM_C):.0f}:1")
print(f"  APTOS-like train={len(apt_tr)} val={len(apt_vl)} test={len(apt_te)}  IR={max(APT_C)/min(APT_C):.0f}:1")
print(f"  OCT-like   test={len(oct_te)} (OOD eval only)")

# ── Model helpers ─────────────────────────────────────────────────────────
def bld(nc,K=5,gt=False):
    cfg=PACNetConfig(num_classes=nc,backbone='tiny',frozen_backbone=False,
                     proj_dim=256,proj_hidden=512,num_prototypes=K,use_global_temp=gt)
    return PACNet(cfg).to(DEVICE)

@torch.no_grad()
def pred(model,ldr):
    model.eval(); P,L,D,G=[],[],[],[]
    for x,y in ldr:
        c,r,d,f=model(x.to(DEVICE),return_all=True)
        P.append(F.softmax(c,dim=-1).cpu().numpy()); L.append(y.numpy())
        D.append(d.cpu().numpy()); G.append(r.cpu().numpy())
    return np.concatenate(P),np.concatenate(L),np.concatenate(D),np.concatenate(G)

def mets(P,L,N,cc=None):
    m={**classification_report_full(P,L,N),**calibration_report(P,L)}
    if cc: m.update(shot_breakdown(P.argmax(-1),L,cc,many_thresh=100,few_thresh=15))
    return m

def mkc(cc,lf=0.5,lr=0.3,ld=0.1,tau=1.0):
    return PACNetLoss(cc,lambda_focal=lf,lambda_reliability=lr,lambda_diversity=ld,la_tau=tau)

def train_fn(model,crit,trl,vll,N,epochs=12,lr=3e-4,tag=''):
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
    best,state,hist=float('inf'),None,[]
    for ep in range(1,epochs+1):
        model.train(); tot=0; nb=0
        for x,y in trl:
            x,y=x.to(DEVICE),y.to(DEVICE); c,r,d=model(x)
            loss,_=crit(c,r,y,model.proto_bank)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            tot+=loss.item(); nb+=1
        sch.step(); tot/=nb
        if ep%2==0 or ep==epochs:
            vP,vL,_,_=pred(model,vll)
            vb=classification_report_full(vP,vL,N)['BalancedAccuracy']; ve=_ece(vP,vL)
            sc=ve-vb
            if sc<best: best=sc; state={k:v.clone() for k,v in model.state_dict().items()}
            hist.append({'ep':ep,'loss':round(tot,4),'val_bacc':round(vb,4),'val_ece':round(ve,4)})
            print(f"    {tag} Ep{ep:02d}  loss={tot:.4f}  val_BAcc={vb:.4f}  val_ECE={ve:.4f}"+(" ✓" if sc==best else ""))
    if state: model.load_state_dict(state)
    return hist

def qtrain(model,crit,epochs=6):
    opt=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sch=CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
    for _ in range(epochs):
        model.train()
        for x,y in ham_trl:
            x,y=x.to(DEVICE),y.to(DEVICE); c,r,d=model(x)
            loss,_=crit(c,r,y,model.proto_bank)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        sch.step()

# ── Train primary models ───────────────────────────────────────────────────
print("\n"+"="*65); print("  [Train] HAM10000-like  (7 cls, IR 20:1)"); print("="*65)
set_seed(SEED); ham_m=bld(7); ham_c=mkc(ham_tr.class_counts)
t0=time.time(); ham_hist=train_fn(ham_m,ham_c,ham_trl,ham_vll,HAM_N,epochs=12,tag='HAM')
print(f"  Done in {time.time()-t0:.1f}s")

print("\n"+"="*65); print("  [Train] APTOS-like  (5 cls, IR 5:1)"); print("="*65)
set_seed(SEED); apt_m=bld(5); apt_c=mkc(apt_tr.class_counts)
t0=time.time(); apt_hist=train_fn(apt_m,apt_c,apt_trl,apt_vll,APT_N,epochs=12,tag='APT')
print(f"  Done in {time.time()-t0:.1f}s")

# ── Predict ────────────────────────────────────────────────────────────────
ham_P,ham_L,ham_D,ham_G=pred(ham_m,ham_tel)
apt_P,apt_L,apt_D,apt_G=pred(apt_m,apt_tel)
oct_P,oct_L,oct_D,oct_G=pred(ham_m,oct_tel)

# ── E1 ─────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E1] Primary Classification"); print("="*65)
e1h=mets(ham_P,ham_L,HAM_N,ham_tr.class_counts)
e1a=mets(apt_P,apt_L,APT_N,apt_tr.class_counts)
for ds,m in [('HAM10000-like',e1h),('APTOS-like',e1a)]:
    print(f"\n  -- {ds} --")
    for k in ['Accuracy','BalancedAccuracy','MacroF1','MacroAUC','ECE','MCE','ACE','Brier','NLL']:
        if k in m: print(f"    {k:<22} {m[k]:.4f}")

# ── E2 ─────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E2] Shot Breakdown"); print("="*65)
e2={k:v for k,v in e1h.items() if any(s in k for s in ['many','medium','few','n_'])}
for k,v in e2.items():
    print(f"  {k:<20}  {v:.4f}" if isinstance(v,float) else f"  {k:<20}  {v}")

# ── E3 ─────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E3] Calibration Detail"); print("="*65)
for ds,P,L in [('HAM',ham_P,ham_L),('APTOS',apt_P,apt_L)]:
    cal=calibration_report(P,L); print(f"\n  -- {ds} --")
    for k,v in cal.items(): print(f"    {k:<8}  {v:.4f}")

# ── E4 ─────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E4] Calibration Shift  HAM→OCT"); print("="*65)
ci=calibration_report(ham_P,ham_L); co=calibration_report(oct_P,oct_L)
e4={k:{'in':ci[k],'ood':co[k],'shift':abs(co[k]-ci[k])} for k in ['ECE','Brier','NLL']}
for k,v in e4.items(): print(f"  {k}  in={v['in']:.4f}  ood={v['ood']:.4f}  delta={v['shift']:.4f}")

# ── E5 ─────────────────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E5] OOD Detection"); print("="*65)
e5={}
for nm,(si,so) in [
    ('PAC-Net Density',(density_score(ham_D),density_score(oct_D))),
    ('MSP Baseline',   (msp_score(ham_P),    msp_score(oct_P))),
    ('Energy Baseline',(energy_score(ham_G), energy_score(oct_G)))]:
    r=ood_report(si,so); e5[nm]=r
    print(f"  {nm:<22}  AUROC={r['AUROC']:.4f}  AUPR={r['AUPR_in']:.4f}  FPR@95={r['FPR@95TPR']:.4f}")

# ── E7 Ablation ─────────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E7] Ablation Study  (8 variants + FULL)"); print("="*65)

class NullPB:
    def diversity_loss(self): return torch.tensor(0.)

class LinNet(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.backbone=TinyCNN(frozen=False)
        self.proj=nn.Sequential(nn.Linear(128,256),nn.BatchNorm1d(256),nn.GELU(),nn.Linear(256,256))
        self.head=nn.Linear(256,nc); self._pb=NullPB()
    @property
    def proto_bank(self): return self._pb
    def forward(self,x,return_all=False):
        f=self.backbone(x); p=F.normalize(self.proj(f),dim=-1); lg=self.head(p)
        dn=torch.zeros(x.shape[0],lg.shape[1])
        return (lg,lg,dn,p) if return_all else (lg,lg,dn)

cc=ham_tr.class_counts; bkw=dict(lambda_focal=0.5,lambda_reliability=0.3,lambda_diversity=0.1,la_tau=1.0)
def vbuild(nm):
    if nm=='A1': return LinNet(7).to(DEVICE),PACNetLoss(cc,**bkw)
    if nm=='A2': return bld(7),PACNetLoss(cc,la_tau=0.,lambda_focal=0.5,lambda_reliability=0.3,lambda_diversity=0.1)
    if nm=='A3': return bld(7),PACNetLoss(cc,lambda_reliability=0.,lambda_focal=0.5,lambda_diversity=0.1,la_tau=1.)
    if nm=='A4': return bld(7),PACNetLoss(cc,lambda_diversity=0.,lambda_focal=0.5,lambda_reliability=0.3,la_tau=1.)
    if nm=='A5': return bld(7),PACNetLoss(cc,**bkw)
    if nm=='A6': return bld(7,K=1),PACNetLoss(cc,**bkw)
    if nm=='A7': return bld(7,K=10),PACNetLoss(cc,**bkw)
    if nm=='A8': return bld(7,gt=True),PACNetLoss(cc,**bkw)
    return bld(7),PACNetLoss(cc,**bkw)

abl={}; t0a=time.time()
for nm in ['FULL','A1','A2','A3','A4','A5','A6','A7','A8']:
    set_seed(SEED); m,c=vbuild(nm); qtrain(m,c,epochs=6)
    with torch.no_grad(): P,L,_,_=pred(m,ham_tel)
    r=mets(P,L,HAM_N)
    abl[nm]={k:r[k] for k in ['BalancedAccuracy','MacroF1','ECE','Brier','NLL'] if k in r}
    print(f"  {nm:<6}  BAcc={r['BalancedAccuracy']:.4f}  MacroF1={r['MacroF1']:.4f}  ECE={r['ECE']:.4f}  Brier={r['Brier']:.4f}")
print(f"  Ablation: {time.time()-t0a:.1f}s")

# ── E8 Statistical significance ────────────────────────────────────────────
print("\n"+"="*65); print("  [E8] Statistical Significance  (3 seeds)"); print("="*65)
sr={'FULL':[],'A1':[],'A2':[]}
for s in [42,43,44]:
    for nm in ['FULL','A1','A2']:
        set_seed(s); m,c=vbuild(nm); qtrain(m,c,epochs=5)
        with torch.no_grad(): P,L,_,_=pred(m,ham_tel)
        sr[nm].append(classification_report_full(P,L,HAM_N)['BalancedAccuracy'])
e8=wilcoxon_holm(sr['FULL'],{'A1 (no prototype)':sr['A1'],'A2 (no logit adj)':sr['A2']})
for nm,r in e8.items():
    print(f"  FULL vs {nm:<22}  p_adj={r['p_adj']:.4f}  r={r['effect_size_r']:.3f}  {'significant' if r['significant'] else 'not significant'}")
print(f"  FULL seeds: {[round(v,4) for v in sr['FULL']]}")
print(f"  A1   seeds: {[round(v,4) for v in sr['A1']]}")
print(f"  A2   seeds: {[round(v,4) for v in sr['A2']]}")

# ── E9 Sensitivity ─────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E9] Sensitivity: K prototypes"); print("="*65)
sens={}
for K in [1,3,5,10]:
    set_seed(SEED); m=bld(7,K=K); c=PACNetLoss(cc,**bkw); qtrain(m,c,epochs=6)
    with torch.no_grad(): P,L,_,_=pred(m,ham_tel)
    cal=calibration_report(P,L); clf=classification_report_full(P,L,HAM_N)
    sens[K]={'BAcc':clf['BalancedAccuracy'],'ECE':cal['ECE']}
    print(f"  K={K:<3}  BAcc={clf['BalancedAccuracy']:.4f}  ECE={cal['ECE']:.4f}"+(" <- default" if K==5 else ""))

# ── E11 Efficiency ─────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E11] Efficiency"); print("="*65)
tr_p=sum(p.numel() for p in ham_m.parameters() if p.requires_grad)
tot_p=sum(p.numel() for p in ham_m.parameters())
dummy=torch.randn(64,3,32,32); ham_m.eval()
with torch.no_grad():
    for _ in range(5): ham_m(dummy)
t0=time.perf_counter()
with torch.no_grad():
    for _ in range(50): ham_m(dummy)
thr=(50*64)/(time.perf_counter()-t0)
print(f"  Total params    : {tot_p:,}"); print(f"  Trainable params: {tr_p:,}")
print(f"  CPU Throughput  : {thr:.0f} img/s  (bs=64, 32x32)")
e11={'total':tot_p,'trainable':tr_p,'throughput':round(thr)}

# ── E10 Figures ────────────────────────────────────────────────────────────
print("\n"+"="*65); print("  [E10] Generating Figures"); print("="*65)

# ── Fig 1: t-SNE ──────────────────────────────────────────────────────────
print("  -> Fig 1: t-SNE")
ham_m.eval()
with torch.no_grad(): _,_,_,feats=pred(ham_m,ham_tel)
protos=ham_m.proto_bank.prototypes.detach().cpu().numpy()
C2,K2,D2=protos.shape; pf=protos.reshape(C2*K2,D2)
pf=pf/(np.linalg.norm(pf,axis=-1,keepdims=True)+1e-8)
ns=min(150,len(feats)); idx_s=np.random.default_rng(0).choice(len(feats),ns,replace=False)
all_pts=np.concatenate([feats[idx_s],pf],axis=0)
tsne=TSNE(n_components=2,random_state=0,perplexity=min(20,ns//4),n_iter=300,verbose=0)
emb=tsne.fit_transform(all_pts); es,ep=emb[:ns],emb[ns:]; ls=ham_L[idx_s]
cnts=np.array(ham_tr.class_counts); med=np.median(cnts)
fig,ax=plt.subplots(figsize=(9,7))
for c,nm in enumerate(HAM_N):
    mask=ls==c; col=PAL[c%len(PAL)]
    ax.scatter(es[mask,0],es[mask,1],c=col,s=22 if cnts[c]<med else 10,alpha=0.6,linewidths=0,
               label=f'{nm} ({"tail" if cnts[c]<med else "head"})')
for c in range(C2):
    pe=ep[c*K2:(c+1)*K2]
    ax.scatter(pe[:,0],pe[:,1],marker='*',s=250,c=PAL[c%len(PAL)],edgecolors='k',linewidths=0.5,zorder=5)
ax.legend(fontsize=8,ncol=2,framealpha=0.7); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('t-SNE: PAC-Net Prototype Space  (* = prototypes)',fontsize=12)
plt.tight_layout(); fig.savefig(OUT/'figures'/'fig1_tsne.png',dpi=180,bbox_inches='tight'); plt.close()

# ── Fig 2: Reliability ────────────────────────────────────────────────────
print("  -> Fig 2: Reliability diagrams")
_,_,_,vlog=pred(ham_m,ham_vll); _,vlbl,_,_=pred(ham_m,ham_vll)
Tp=nn.Parameter(torch.tensor(1.5)); tso=optim.LBFGS([Tp],lr=0.01,max_iter=100)
vlt=torch.tensor(vlog,dtype=torch.float32); vll2=torch.tensor(vlbl,dtype=torch.long)
def cl():
    tso.zero_grad(); loss=F.cross_entropy(vlt/Tp,vll2); loss.backward(); return loss
tso.step(cl); Tbest=Tp.item()
ts_P=F.softmax(torch.tensor(ham_G,dtype=torch.float32)/Tbest,dim=-1).numpy()
unc_P=F.softmax(torch.tensor(ham_G,dtype=torch.float32),dim=-1).numpy()
mths={'Uncalibrated':(unc_P,ham_L),f'Temp.Scaling T={Tbest:.2f}':(ts_P,ham_L),'PAC-Net (ours)':(ham_P,ham_L)}
fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,(mn,(P,L)),col in zip(axes,mths.items(),['#CC3311','#EE7733','#0077BB']):
    rel=reliability_data(P,L,n_bins=12); bc,ba,edges=rel['bin_conf'],rel['bin_acc'],rel['bin_edges']; w=np.diff(edges)
    ax.bar(edges[:-1],ba,width=w,align='edge',color=col,alpha=0.72,edgecolor='white')
    ax.bar(edges[:-1],bc,width=w,align='edge',color='none',edgecolor='crimson',linestyle='--',linewidth=1.2)
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
    ax.set_title(f'{mn}\nECE={_ece(P,L):.4f}',fontsize=10); ax.set_xlabel('Confidence'); ax.set_xlim(0,1); ax.set_ylim(0,1)
axes[0].set_ylabel('Accuracy'); fig.suptitle('Reliability Diagrams',fontsize=12,y=1.01)
plt.tight_layout(); fig.savefig(OUT/'figures'/'fig2_reliability.png',dpi=180,bbox_inches='tight'); plt.close()

# ── Fig 3: Confusion matrix ───────────────────────────────────────────────
print("  -> Fig 3: Confusion matrix")
cm=confusion_matrix(ham_L,ham_P.argmax(-1),normalize='true')
fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels(HAM_N,rotation=40,ha='right',fontsize=9); ax.set_yticklabels(HAM_N,fontsize=9)
for i in range(7):
    for j in range(7):
        v=cm[i,j]; ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,color='white' if v>0.55 else 'black')
ax.set_xlabel('Predicted',fontsize=10); ax.set_ylabel('True',fontsize=10)
ax.set_title('Normalised Confusion Matrix',fontsize=11)
plt.tight_layout(); fig.savefig(OUT/'figures'/'fig3_confusion.png',dpi=180,bbox_inches='tight'); plt.close()

# ── Fig 4: Density OOD ────────────────────────────────────────────────────
print("  -> Fig 4: Density distribution")
di=density_score(ham_D); do=density_score(oct_D)
bns=np.linspace(min(di.min(),do.min()),max(di.max(),do.max()),50)
fig,ax=plt.subplots(figsize=(8,4))
ax.hist(di,bins=bns,density=True,alpha=0.65,color='#0077BB',label='In-dist (HAM-like)')
ax.hist(do,bins=bns,density=True,alpha=0.65,color='#EE7733',label='OOD (OCT-like)')
ax.axvline(di.mean(),color='#0077BB',ls='--',lw=1.8); ax.axvline(do.mean(),color='#EE7733',ls='--',lw=1.8)
ax.set_xlabel('Max Prototype Density',fontsize=11); ax.set_ylabel('Density',fontsize=11)
ax.set_title('PAC-Net Anomaly Score: In-Distribution vs OOD',fontsize=12); ax.legend(fontsize=10)
plt.tight_layout(); fig.savefig(OUT/'figures'/'fig4_density_ood.png',dpi=180,bbox_inches='tight'); plt.close()

# ── Fig 5: Per-class ECE ─────────────────────────────────────────────────
print("  -> Fig 5: Per-class ECE")
pce=[_ece(np.stack([1-ham_P[:,c],ham_P[:,c]],axis=-1),(ham_L==c).astype(int)) for c in range(7)]
cols=['#0077BB' if cnts[c]>=med else '#EE7733' for c in range(7)]
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(HAM_N,pce,color=cols,edgecolor='white')
ax.set_xlabel('Class',fontsize=11); ax.set_ylabel('ECE',fontsize=11)
ax.set_title('Per-Class Calibration Error',fontsize=12); ax.tick_params(axis='x',rotation=25)
ax.legend(handles=[mpatches.Patch(color='#0077BB',label='Head'),
                   mpatches.Patch(color='#EE7733',label='Tail')],fontsize=10)
plt.tight_layout(); fig.savefig(OUT/'figures'/'fig5_per_class_ece.png',dpi=180,bbox_inches='tight'); plt.close()

# ── Fig 6: Training curves ────────────────────────────────────────────────
print("  -> Fig 6: Training curves")
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
eps_h=[h['ep'] for h in ham_hist]; eps_a=[h['ep'] for h in apt_hist]
a1.plot(eps_h,[h['loss'] for h in ham_hist],'#0077BB',marker='o',ms=4,label='HAM loss')
a1r=a1.twinx(); a1r.plot(eps_h,[h['val_ece'] for h in ham_hist],'#EE7733',marker='s',ms=4,ls='--',label='HAM val ECE')
a1.set_xlabel('Epoch'); a1.set_ylabel('Loss',color='#0077BB'); a1r.set_ylabel('Val ECE',color='#EE7733')
a1.set_title('HAM: Loss & Calibration During Training',fontsize=11)
a2.plot(eps_h,[h['val_bacc'] for h in ham_hist],'#009988',marker='o',ms=4,label='HAM val BAcc')
a2.plot(eps_a,[h['val_bacc'] for h in apt_hist],'#CC3311',marker='s',ms=4,label='APT val BAcc')
a2.set_xlabel('Epoch'); a2.set_ylabel('Balanced Accuracy'); a2.legend(fontsize=9)
a2.set_title('Balanced Accuracy During Training',fontsize=11)
plt.tight_layout(); fig.savefig(OUT/'figures'/'fig6_training_curves.png',dpi=180,bbox_inches='tight'); plt.close()

# ── Fig 7: Ablation bar ───────────────────────────────────────────────────
print("  -> Fig 7: Ablation bar chart")
nms=['FULL','A1','A2','A3','A4','A5','A6','A7','A8']
baccs=[abl[n]['BalancedAccuracy'] for n in nms]; eces=[abl[n]['ECE'] for n in nms]
x=np.arange(len(nms)); w=0.35
fig,ax=plt.subplots(figsize=(10,4))
ax.bar(x-w/2,baccs,w,color='#0077BB',alpha=0.8,label='Bal.Acc')
ax2v=ax.twinx(); ax2v.bar(x+w/2,eces,w,color='#EE7733',alpha=0.8,label='ECE')
ax.set_xticks(x); ax.set_xticklabels(nms,fontsize=10)
ax.set_ylabel('Balanced Accuracy',color='#0077BB',fontsize=10); ax2v.set_ylabel('ECE',color='#EE7733',fontsize=10)
ax.set_title('Ablation Study Results',fontsize=12)
# Highlight FULL bar
ax.patches[0].set_edgecolor('black'); ax.patches[0].set_linewidth(2.5)
ax.legend(loc='upper left',fontsize=9); ax2v.legend(loc='upper right',fontsize=9)
plt.tight_layout(); fig.savefig(OUT/'figures'/'fig7_ablation.png',dpi=180,bbox_inches='tight'); plt.close()
print("  All 7 figures saved.")

# ── Save JSON & print master table ─────────────────────────────────────────
results={'E1_ham':e1h,'E1_apt':e1a,'E2':e2,'E4':e4,'E5':e5,
         'E7':abl,'E9':sens,'E11':e11,
         'E8':{k:{kk:(float(vv) if isinstance(vv,(float,np.floating)) else vv)
                   for kk,vv in v.items()} for k,v in e8.items()},
         'training_history':{'ham':ham_hist,'apt':apt_hist}}
with open(OUT/'results.json','w') as f: json.dump(results,f,indent=2,default=str)

W=65
print('\n'+'='*W)
print('  COMPLETE RESULTS TABLE')
print('='*W)
def row(lbl,val,note=''):
    s=f"  {lbl:<30} {val}"; return s+note

print('\n  E1 — Primary Classification')
print(f"  {'Metric':<22} HAM10000-like    APTOS-like")
print('  '+'-'*55)
for k in ['BalancedAccuracy','MacroF1','ECE','Brier','NLL']:
    hv=e1h.get(k,'n/a'); av=e1a.get(k,'n/a')
    hv=f"{hv:.4f}" if isinstance(hv,float) else 'n/a'
    av=f"{av:.4f}" if isinstance(av,float) else 'n/a'
    print(f"  {k:<22} {hv:<16} {av}")

print('\n  E2 — Shot Breakdown (HAM)')
for k,v in e2.items():
    print(f"  {k:<22} {v:.4f}" if isinstance(v,float) else f"  {k:<22} {v}")

print('\n  E4 — Calibration Shift (HAM -> OCT-like)')
for k,v in e4.items(): print(f"  {k}  in={v['in']:.4f}  ood={v['ood']:.4f}  delta={v['shift']:.4f}")

print('\n  E5 — OOD Detection')
print(f"  {'Method':<22} {'AUROC':>8} {'AUPR_in':>9} {'FPR@95':>8}")
print('  '+'-'*50)
for nm,r in e5.items(): print(f"  {nm:<22} {r['AUROC']:>8.4f} {r['AUPR_in']:>9.4f} {r['FPR@95TPR']:>8.4f}")

print('\n  E7 — Ablation Study')
print(f"  {'Variant':<8} {'BAcc':>8} {'MacroF1':>9} {'ECE':>8} {'Brier':>8}")
print('  '+'-'*45)
for nm in ['FULL','A1','A2','A3','A4','A5','A6','A7','A8']:
    r=abl[nm]; note='  <- proposed' if nm=='FULL' else ''
    print(f"  {nm:<8} {r['BalancedAccuracy']:>8.4f} {r['MacroF1']:>9.4f} {r['ECE']:>8.4f} {r['Brier']:>8.4f}{note}")

print('\n  E8 — Statistical Significance (3 seeds)')
for nm,r in e8.items():
    sig='SIGNIFICANT' if r['significant'] else 'not sig.'
    print(f"  FULL vs {nm[:30]:<30}  p={r['p_adj']:.4f}  {sig}")

print('\n  E9 — Sensitivity: K prototypes')
for K,v in sens.items():
    print(f"  K={K}   BAcc={v['BAcc']:.4f}  ECE={v['ECE']:.4f}"+('  <- default' if K==5 else ''))

print('\n  E11 — Efficiency')
print(f"  Total params      {e11['total']:>12,}")
print(f"  Trainable params  {e11['trainable']:>12,}")
print(f"  CPU throughput    {e11['throughput']:>12,} img/s")
print('\n'+'='*W)
print(f"\n[DONE]  Full pipeline complete.")
print(f"  Results  ->  {OUT/'results.json'}")
print(f"  Figures  ->  {OUT/'figures/'} (7 PNG files)")
