 
import common as C, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
for ds in ['vqa_rad','path_vqa','omni']:
    s=C.load_raw(ds); tr,va,te,rows,audit=C.get_level(ds,'L4_fair',s)
    d=np.load(C.CACHE/('trained_%s_L4_fair.npz'%ds)); ni=np.load(C.CACHE/('trained_noimg_%s_L4_fair.npz'%ds))
    te=d['idx_te']; mlp=d['mlp_score']; noimg=ni['score']
    trs=set(int(x) for x in tr)
    pos={}; neg={}
    for j,i in enumerate(te):
        (pos if rows[i]['label']==0 else neg)[rows[i]['pid']]=j
    ps=[p for p in pos if p in neg and C.answer_is_yesno(rows[te[pos[p]]]['answer'])]
    if len(ps)<50: continue
    dm=np.array([mlp[pos[p]]-mlp[neg[p]] for p in ps]); dn=np.array([noimg[pos[p]]-noimg[neg[p]] for p in ps])
    qtr=[rows[te[pos[p]]]['question'] for p in ps if int(te[pos[p]]) in trs]
    ytr=[1 if C.answer_polarity(rows[te[pos[p]]]['answer'])=='y' else 0 for p in ps if int(te[pos[p]]) in trs]
    qte=[rows[te[pos[p]]]['question'] for p in ps if int(te[pos[p]]) not in trs]
    yte_=[1 if C.answer_polarity(rows[te[pos[p]]]['answer'])=='y' else 0 for p in ps if int(te[pos[p]]) not in trs]
    A=float('nan')
    if len(set(ytr))>1 and len(qte)>50 and len(set(yte_))>1:
        pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2),LogisticRegression(max_iter=2000))
        pipe.fit(qtr,ytr); A=roc_auc_score(yte_,pipe.predict_proba(qte)[:,1])
    print('%-9s pairs=%4d  P(full correct)=%.3f  P(noimg correct)=%.3f  question->polarity AUROC=%.3f (base=%.3f, ntest=%d)'%(
        ds,len(ps),1-(dm>0).mean(),1-(dn>0).mean(),A,(np.mean(yte_) if qte else float('nan')),len(qte)))
