"""Frozen, prediction-reuse reviewer experiments. No manuscript modifications."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('RFSC_WORKSPACE_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
from bps_gnn.io import read_frozen_prediction, read_labels_for_ids, sha256_file, freeze_prediction_file
from bps_gnn.projection import boundary_projection, select_active_edges
from bps_gnn.temporal import temporal_split_ids
from scripts.fit_response_free_risk_model import features as original_features

OUT = Path(os.environ.get('RFSC_AUDIT_OUTPUT', ROOT / 'reproduced/structural_audits')).resolve()
TOL = 1e-10
FEATURES = ['correction_abs_scaled','base_disagreement_scaled','active_degree',
            'log_component_size','component_density','component_span_scaled']
METRICS = ['overall_mae','cliff_endpoint_mae','pair_delta_mae']
SOURCES = {}


def record(path):
    path = Path(path)
    key = str(path.relative_to(ROOT))
    if key not in SOURCES:
        SOURCES[key] = {'sha256':sha256_file(path),'bytes':path.stat().st_size}
    return path


def readpred(path):
    record(path)
    return read_frozen_prediction(Path(path))


def readtruth(root,target,ids):
    record(Path(root)/target/'labels_private.parquet')
    return read_labels_for_ids(root,target,ids)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding='utf-8')


def arrhash(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()


def graph(index, frame):
    n = len(index)
    lookup = pd.Index(index)
    e = frame.loc[frame.mol_id_lo.isin(lookup) & frame.mol_id_hi.isin(lookup)].copy()
    left, right = lookup.get_indexer(e.mol_id_lo), lookup.get_indexer(e.mol_id_hi)
    assert (left >= 0).all() and (right >= 0).all()
    return left, right, e


def components(n, left, right):
    adj = coo_matrix((np.ones(2*len(left)),(np.r_[left,right],np.r_[right,left])),shape=(n,n)).tocsr()
    count, label = connected_components(adj,directed=False)
    return count, label


def project(y, left, right, scale=1.5):
    _, label = components(len(y),left,right)
    size = np.bincount(label)
    center = np.bincount(label,weights=y) / size
    return y+(scale-1.)*(y-center[label]), label


def feature_array(y, corrected, svm, fixed, left, right):
    delta = y[right]-y[left]
    active = np.abs(delta) >= .5
    lo, hi = left[active], right[active]
    count, label = components(len(y),lo,hi)
    size = np.bincount(label,minlength=count)
    degree = np.bincount(np.r_[lo,hi],minlength=len(y))
    edge_count = np.bincount(label[lo],minlength=count)
    density = np.zeros(count)
    mask = size > 1
    density[mask] = 2*edge_count[mask]/(size[mask]*(size[mask]-1))
    minimum = np.full(count,np.inf)
    maximum = np.full(count,-np.inf)
    np.minimum.at(minimum,label,y)
    np.maximum.at(maximum,label,y)
    scale = max(float(np.std(y)),1e-8)
    return np.column_stack((np.abs(corrected-y)/scale,np.abs(svm-fixed)/scale,
        degree,np.log1p(size[label]),density[label],(maximum-minimum)[label]/scale))


def metrics(y, truth, left, right):
    dt = truth[right]-truth[left]
    cliff = np.abs(dt) >= 1.
    ep = np.unique(np.r_[left[cliff],right[cliff]])
    return {'n_nodes':len(y),'n_mmp_edges':len(left),'n_cliff_edges':int(cliff.sum()),
        'overall_mae':float(np.mean(np.abs(y-truth))),
        'cliff_endpoint_mae':float(np.mean(np.abs(y[ep]-truth[ep]))) if len(ep) else np.nan,
        'pair_delta_mae':float(np.mean(np.abs((y[right]-y[left]-dt)[cliff]))) if cliff.any() else np.nan,
        'all_mmp_pair_mae':float(np.mean(np.abs(y[right]-y[left]-dt))) if len(left) else np.nan}


def hierarchical(frame, by, columns):
    # Equal seeds per fold, equal folds per target, equal targets per cohort.
    level1=frame.groupby(by+['target','fold'],dropna=False,as_index=False)[columns].mean()
    targets=level1.groupby(by+['target'],dropna=False,as_index=False)[columns].mean()
    summary=targets.groupby(by,dropna=False,as_index=False)[columns].mean()
    return targets, summary


def distributions(values):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return dict(count=0,mean=np.nan,median=np.nan,p95=np.nan,maximum=np.nan)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),maximum=float(a.max()))


def aq_runs(limit=None):
    runs=sorted((ROOT/'experiments/eswa_aqsol_external/predictions').glob('*/fold_*/seed_*/run.json'))
    return runs[:limit] if limit else runs


def load_aq(runpath):
    run=json.loads(record(runpath).read_text())
    assert run['test_labels_materialized'] is False
    p=runpath.parent
    base=readpred(p/'unscaled_reference/predictions.parquet')
    pred={m:readpred(p/f'{m}/predictions.parquet').reindex(base.index) for m in ('bps','svm','rf','gine')}
    edges=pd.read_parquet(record(ROOT/f'graphs/mmp/{run["target"]}/eligible_mmp_edges.parquet'))
    left,right,e=graph(base.index,edges)
    scorepath=ROOT/f'experiments/eswa_response_free_risk/aqsol_scores/{run["target"]}/fold_{run["fold"]}/seed_{run["seed"]}/risk_scores.parquet'
    md=json.loads(scorepath.with_suffix('.parquet.sha256.json').read_text())
    assert md['response_columns_read'] is False and sha256_file(scorepath)==md['risk_score_sha256']
    scores=pd.read_parquet(record(scorepath)).set_index('mol_id').risk_score.reindex(base.index)
    assert scores.notna().all()
    return run,base,pred,left,right,e,scores


def mechanism(args,dest):
    rows=[]
    bins=[]
    work=[]
    for name in ('development','phase4_secondary'):
        paths=sorted((ROOT/f'experiments/svm_bps_selected/{name}').glob('*/fold_*/seed_*/run.json'))
        for p in paths:
            r=json.loads(p.read_text())
            ginroot='experiments/h1_oracle/gine_scg_panel' if name=='development' else 'experiments/phase4_confirmatory/gine_scg'
            gpath=ROOT/ginroot/r['target']/f'fold_{r["fold"]}'/f'seed_{r["seed"]}'/'predictions.parquet'
            work.append(('MoleculeACE',r,p.parent,ROOT/'data/processed/moleculeace',ROOT/'graphs/mmp',gpath))
    base_temporal=ROOT/'experiments/chembl36_temporal_earliest'
    for p in sorted((base_temporal/'predictions').glob('*/seed_*/run.json')):
        r=json.loads(p.read_text());r['fold']=0
        gpath=p.parent/'gine/predictions.parquet'
        work.append(('corrected_ChEMBL',r,p.parent,ROOT/'data/processed/chembl36_temporal_earliest',ROOT/'graphs/chembl36_temporal_mmp',gpath))
    if args.smoke:
        work=[next(w for w in work if w[0]==ds) for ds in ('MoleculeACE','corrected_ChEMBL')]
    for wi,(ds,r,p,dataroot,mmp,gpath) in enumerate(work):
        base=readpred(p/('svm_fixed_mixture' if ds=='MoleculeACE' else 'unscaled_reference')/'predictions.parquet')
        truth=readtruth(dataroot,r['target'],base.index).set_index('mol_id').y_pactivity.reindex(base.index).to_numpy()
        edges=pd.read_parquet(record(mmp/r['target']/'eligible_mmp_edges.parquet'))
        left,right,e=graph(base.index,edges)
        models={'mixture':base,'SVM':readpred(p/('svm_ecfp' if ds=='MoleculeACE' else 'svm')/'predictions.parquet'),
                'GINE':readpred(gpath)}
        dt=truth[right]-truth[left]
        for model,series in models.items():
            y=series.reindex(base.index).to_numpy()
            delta=y[right]-y[left]
            active=np.abs(delta)>=.5
            corrected,label=project(y,left[active],right[active])
            if model=='mixture':
                cpath=p/('svm_bps_selected' if ds=='MoleculeACE' else 'bps')/'predictions.parquet'
                existing=readpred(cpath).reindex(base.index).to_numpy()
                assert np.max(np.abs(existing-corrected),initial=0)<1e-9
            newdelta=corrected[right]-corrected[left]
            before=np.abs(delta-dt);after=np.abs(newdelta-dt)
            identity={'dataset':ds,'target':r['target'],'fold':r['fold'],'seed':r['seed'],'model':model}
            category=np.digitize(np.abs(delta),[.25,.5,.75,1.],right=False)
            for scope, subset in (('all_MMP',np.ones(len(dt),dtype=bool)),('true_cliff',np.abs(dt)>=1.)):
                mask=active & subset
                n=int(mask.sum())
                under=np.abs(delta)<np.abs(dt)-TOL
                over=np.abs(delta)>np.abs(dt)+TOL
                improved=after<before-TOL;worsened=after>before+TOL
                summary={**identity,'edge_scope':scope,'n_all_edges':int(subset.sum()),'active_edges':n}
                for k,m in [('underestimated',under),('overestimated',over),('equal_amplitude',~(under|over)),
                            ('improved',improved),('worsened',worsened),('unchanged',~(improved|worsened)),
                            ('same_nonzero_sign',delta*dt>0),('opposite_sign',delta*dt<0)]:
                    summary[k+'_count']=int((m&mask).sum())
                    summary[k+'_fraction']=float(m[mask].mean()) if n else np.nan
                summary['active_pair_mae_before']=float(before[mask].mean()) if n else np.nan
                summary['active_pair_mae_after']=float(after[mask].mean()) if n else np.nan
                summary['all_pair_mae_before']=float(before[subset].mean()) if subset.any() else np.nan
                summary['all_pair_mae_after']=float(after[subset].mean()) if subset.any() else np.nan
                for amp,am in [('underestimated',under),('overestimated',over),('equal_amplitude',~(under|over))]:
                    for sign,sm in [('same_sign',delta*dt>0),('opposite_sign',delta*dt<0),('zero_sign',delta*dt==0)]:
                        cm=mask&am&sm;nn=int(cm.sum())
                        summary[f'{amp}_{sign}_count']=nn
                        summary[f'{amp}_{sign}_improved_fraction']=float(improved[cm].mean()) if nn else np.nan
                rows.append(summary)
                for b,name in enumerate(('0_to_.25','.25_to_.50','.50_to_.75','.75_to_1','ge_1')):
                    bm=subset&(category==b);bn=int(bm.sum())
                    bins.append({**identity,'edge_scope':scope,'bin':name,'edge_count':bn,
                        'mean_abs_predicted_delta':float(np.abs(delta[bm]).mean()) if bn else np.nan,
                        'mean_abs_true_delta':float(np.abs(dt[bm]).mean()) if bn else np.nan,
                        'pair_mae_before':float(before[bm].mean()) if bn else np.nan,
                        'underestimated_fraction':float(under[bm].mean()) if bn else np.nan,
                        'overestimated_fraction':float(over[bm].mean()) if bn else np.nan,
                        'same_nonzero_sign_fraction':float((delta[bm]*dt[bm]>0).mean()) if bn else np.nan})
        if wi%30==0:print('mechanism',wi+1,'/',len(work),flush=True)
    raw=pd.DataFrame(rows);bf=pd.DataFrame(bins)
    raw.to_csv(dest/'mechanism_run_metrics.csv',index=False);bf.to_csv(dest/'predicted_delta_bins_run.csv',index=False)
    skip={'dataset','model','edge_scope','target','fold','seed'}
    cols=[c for c in raw if c not in skip]
    target,summary=hierarchical(raw,['dataset','model','edge_scope'],cols)
    target.to_csv(dest/'mechanism_target_metrics.csv',index=False)
    extra=[]
    for (ds,m,scope),g in target.groupby(['dataset','model','edge_scope']):
        eligible=g.dropna(subset=['active_pair_mae_before'])
        sufficient=eligible.underestimated_fraction>.5
        improved=eligible.improved_fraction>.5
        extra.append({'dataset':ds,'model':m,'edge_scope':scope,'tasks':len(g),'active_eligible_tasks':len(eligible),
            'underestimation_majority_tasks':int(sufficient.sum()),'improvement_majority_tasks':int(improved.sum()),
            'active_pair_relative_change_macro':float(eligible.active_pair_mae_after.mean()/eligible.active_pair_mae_before.mean()-1),
            'all_pair_relative_change_macro':float(g.all_pair_mae_after.mean()/g.all_pair_mae_before.mean()-1),
            'under_majority_pair_relative_change':float(eligible[sufficient].active_pair_mae_after.mean()/eligible[sufficient].active_pair_mae_before.mean()-1) if sufficient.any() else np.nan})
    summary=summary.merge(pd.DataFrame(extra),on=['dataset','model','edge_scope'])
    summary.to_csv(dest/'mechanism_summary.csv',index=False)
    c=[v for v in bf if v not in skip|{'bin'}]
    bt,bs=hierarchical(bf,['dataset','model','edge_scope','bin'],c)
    bt.to_csv(dest/'predicted_delta_bins_target.csv',index=False);bs.to_csv(dest/'predicted_delta_bins_summary.csv',index=False)
    return {'runs':len(work),'model_runs':len(work)*3,'datasets':sorted(raw.dataset.unique())}


def structure(args,dest):
    payload=joblib.load(record(ROOT/'experiments/eswa_response_free_risk/model/risk_model.joblib'))
    threshold=float(payload['thresholds']['0.8'])
    rows=[];cr=[];pool={};cpool={}
    for p in aq_runs(1 if args.smoke else None):
        r,base,pred,left,right,e,scores=load_aq(p)
        y=base.to_numpy();full=pred['bps'].to_numpy();g=scores.to_numpy()<=threshold
        hybrid=np.where(g,full,y)
        active=np.abs(y[right]-y[left])>=.5
        lo,hi=left[active],right[active]
        groups={'all_active':np.ones(len(lo),bool),'both_accepted':g[lo]&g[hi],
                'one_accepted':g[lo]^g[hi],'both_rejected':~g[lo]&~g[hi]}
        identity={'fold':r['fold'],'seed':r['seed']}
        for method,output in [('full_BPS',full),('hybrid_candidate',hybrid)]:
            deviation=np.abs(output[hi]-output[lo]-1.5*(y[hi]-y[lo]))
            if method=='full_BPS':assert np.max(deviation,initial=0)<TOL
            for group,mask in groups.items():
                if group=='both_accepted':assert np.max(deviation[mask],initial=0)<TOL
                a=deviation[mask];pool.setdefault((method,group),[]).append(a)
                rows.append({**identity,'method':method,'group':group,'edge_fraction':float(mask.mean()) if len(mask) else np.nan,**distributions(a)})
            count,label=components(len(y),lo,hi);size=np.bincount(label)
            drift=np.bincount(label,weights=output-y)/size
            if method=='full_BPS':assert np.max(np.abs(drift),initial=0)<TOL
            for scope,mask in [('all_components',np.ones(count,bool)),('nontrivial_components',size>1)]:
                a=np.abs(drift[mask]);cpool.setdefault((method,scope),[]).append(a)
                cr.append({**identity,'method':method,'scope':scope,'nonzero_fraction':float((a>TOL).mean()) if len(a) else np.nan,
                           'mean_signed_shift':float(drift[mask].mean()) if len(a) else np.nan,**distributions(a)})
    pd.DataFrame(rows).to_csv(dest/'constraint_run_metrics.csv',index=False)
    pd.DataFrame(cr).to_csv(dest/'component_mean_run_metrics.csv',index=False)
    summary=[]
    for (m,g),parts in pool.items():
        a=np.concatenate(parts)
        summary.append({'method':m,'group':g,**distributions(a)})
    pd.DataFrame(summary).to_csv(dest/'constraint_pooled_descriptive.csv',index=False)
    cs=[]
    for (m,g),parts in cpool.items():
        a=np.concatenate(parts);cs.append({'method':m,'scope':g,'nonzero_fraction':float((a>TOL).mean()),**distributions(a)})
    pd.DataFrame(cs).to_csv(dest/'component_mean_pooled_descriptive.csv',index=False)
    # Coverage reproduced with original fold-then-seed scientific unit.
    return {'external_runs':len(aq_runs(1 if args.smoke else None)), 'threshold':threshold,
            'interpretation':'Hybrid is an audited candidate; original point-risk authorization remains NO-GO.'}


def signatures(label):
    # Exact member-set hash, invariant to arbitrary component numbering.
    values={}
    for k in np.unique(label):
        values[int(k)]=hashlib.sha256(np.flatnonzero(label==k).astype('<i8').tobytes()).hexdigest()
    return np.array([values[int(k)] for k in label])


def perturb(args,dest):
    payload=joblib.load(record(ROOT/'experiments/eswa_response_free_risk/model/risk_model.joblib'))
    threshold=float(payload['thresholds']['0.8']);model=payload['model'];names=payload['features']
    rows=[];near=[]
    for k,p in enumerate(aq_runs(1 if args.smoke else None)):
        r,base,pred,left,right,e,scores=load_aq(p)
        y=base.to_numpy();full=pred['bps'].to_numpy();svm=pred['svm'].to_numpy()
        fixed=.5*(pred['rf'].to_numpy()+pred['gine'].to_numpy())
        x=feature_array(y,full,svm,fixed,left,right)
        expected=original_features(base,pred['bps'],pred['svm'],pd.Series(fixed,index=base.index),e)[names].to_numpy()
        assert np.max(np.abs(x-expected),initial=0)<1e-9
        score=model.predict(pd.DataFrame(x,columns=names))
        assert np.max(np.abs(score-scores.to_numpy()),initial=0)<1e-8
        g=score<=threshold;hybrid=np.where(g,full,y)
        active=np.abs(y[right]-y[left])>=.5
        c0,l0=components(len(y),left[active],right[active]);sig0=signatures(l0)
        q=np.abs(np.abs(y[right]-y[left])-.5)
        for eps in (.01,.05):
            for seed in (range(1) if args.smoke else range(5)):
                # Stable per-run noise; identical eps-scaled draw at both magnitudes.
                token=f'290820:{r["fold"]}:{r["seed"]}:{seed}'.encode()
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'))
                noise=rng.uniform(-eps,eps,len(y));yn=y+noise
                an=np.abs(yn[right]-yn[left])>=.5
                new,ln=project(yn,left[an],right[an]);frozen,_=project(yn,left[active],right[active])
                xn=feature_array(yn,new,svm+noise,fixed+noise,left,right)
                sn=model.predict(pd.DataFrame(xn,columns=names));gn=sn<=threshold
                hn=np.where(gn,new,yn)
                flip=active^an
                assert not np.any(flip&(q>2*eps+TOL))
                input_l2=float(np.linalg.norm(noise));fixed_l2=float(np.linalg.norm(frozen-full))
                assert fixed_l2<=1.5*input_l2+1e-9
                row={'fold':r['fold'],'seed':r['seed'],'noise_seed':seed,'epsilon':eps,
                     'nodes':len(y),'mmp_edges':len(left),'active_edges_before':int(active.sum()),'active_edges_after':int(an.sum()),
                     'added_edges':int((an&~active).sum()),'deleted_edges':int((active&~an).sum()),
                     'added_fraction_all_MMP':float((an&~active).mean()),'deleted_fraction_all_MMP':float((active&~an).mean()),
                     'flip_fraction_all_MMP':float(flip.mean()),'components_before':c0,'components_after':len(np.unique(ln)),
                     'component_count_change':len(np.unique(ln))-c0,'component_membership_changed_fraction':float((signatures(ln)!=sig0).mean()),
                     'input_l2':input_l2,'fixed_graph_output_l2':fixed_l2,'fixed_graph_lipschitz_ratio':fixed_l2/input_l2,
                     'reselected_output_l2':float(np.linalg.norm(new-full)),
                     'reselection_excess_l2':float(np.linalg.norm(new-frozen)),
                     'selector_flip_fraction':float((gn!=g).mean()),'coverage_before':float(g.mean()),'coverage_after':float(gn.mean()),
                     'full_output_change_mae':float(np.abs(new-full).mean()),'full_output_change_max':float(np.abs(new-full).max()),
                     'hybrid_output_change_mae':float(np.abs(hn-hybrid).mean()),'hybrid_output_change_max':float(np.abs(hn-hybrid).max()),
                     'fixed_graph_change_mae':float(np.abs(frozen-full).mean()),'fixed_graph_change_max':float(np.abs(frozen-full).max())}
                rows.append(row)
                for window in (.01,.05):
                    mask=q<=window
                    near.append({'fold':r['fold'],'seed':r['seed'],'noise_seed':seed,'epsilon':eps,'window':window,
                        'near_edges':int(mask.sum()),'near_fraction':float(mask.mean()),'near_flipped':int((mask&flip).sum()),
                        'near_flip_fraction':float(flip[mask].mean()) if mask.any() else np.nan,
                        'share_of_flips_in_window':float((mask&flip).sum()/flip.sum()) if flip.any() else np.nan})
        print('perturbation run',k+1,flush=True)
    f=pd.DataFrame(rows);n=pd.DataFrame(near)
    f.to_csv(dest/'perturbation_run_metrics.csv',index=False);n.to_csv(dest/'threshold_neighborhood_run_metrics.csv',index=False)
    cols=[c for c in f.select_dtypes(include='number') if c not in ('fold','seed','noise_seed','epsilon')]
    fold=f.groupby(['epsilon','fold'],as_index=False)[cols].mean()
    mean=fold.groupby('epsilon',as_index=False)[cols].mean()
    maxima=f.groupby('epsilon',as_index=False)[['full_output_change_max','hybrid_output_change_max','fixed_graph_lipschitz_ratio']].max()
    maxima=maxima.rename(columns={c:c+'_worst_run' for c in maxima if c!='epsilon'})
    mean.merge(maxima,on='epsilon').to_csv(dest/'perturbation_summary.csv',index=False)
    n.groupby(['epsilon','window'],as_index=False).mean(numeric_only=True).to_csv(dest/'threshold_neighborhood_summary.csv',index=False)
    return {'external_runs':len(aq_runs(1 if args.smoke else None)),'perturbations':len(f),'selector_refitted':False}


def selector_audit(args,dest):
    framepath=record(ROOT/'results/is_revision2_selector_coverage_curve/development_selector_frame.parquet')
    f=pd.read_parquet(framepath)
    payload=joblib.load(record(ROOT/'experiments/eswa_response_free_risk/model/risk_model.joblib'))
    score=payload['model'].predict(f[payload['features']])
    rowid=f.target.astype(str)+':'+f.fold.astype(str)+':'+f.seed.astype(str)+':'+f.mol_id.astype(str)
    assert not rowid.duplicated().any()
    tables=[]
    for target,g in f.groupby('target'):
        structure=pd.read_parquet(record(ROOT/f'data/processed/moleculeace/{target}/structures.parquet'))
        mapping=structure.set_index('mol_id').canonical_isomeric_smiles
        smiles=g.mol_id.map(mapping)
        assert smiles.notna().all()
        t=g[['target','fold','seed','mol_id']].copy();t['canonical_smiles']=smiles
        tables.append(t)
    identities=pd.concat(tables,ignore_index=True)
    grouped=identities.groupby('canonical_smiles').agg(records=('mol_id','size'),
                tasks=('target','nunique'),outer_folds=('fold','nunique'),seeds=('seed','nunique'))
    # Do not distribute molecule identities in summary output; preserve only counts.
    grouped.reset_index(drop=True).to_csv(dest/'molecule_repeat_counts.csv',index=False)
    q=[]
    for coverage in (.5,.7,.8,.9,1.):
        computed=float(np.quantile(score,coverage));frozen=float(payload['thresholds'][str(coverage)])
        q.append({'nominal_coverage':coverage,'frozen_threshold':frozen,'recomputed_training_score_quantile':computed,
                  'absolute_difference':abs(computed-frozen),'realized_training_fraction':float((score<=frozen).mean())})
        assert abs(computed-frozen)<1e-8
    pd.DataFrame(q).to_csv(dest/'threshold_quantile_reproduction.csv',index=False)
    corr=pd.DataFrame({'score':score,'harm':f.normalized_point_harm,'target':f.target,'fold':f.fold,'seed':f.seed})
    run=corr.groupby(['target','fold','seed']).apply(lambda v:v.score.corr(v.harm),include_groups=False).rename('training_internal_correlation').reset_index()
    run.to_csv(dest/'training_internal_run_correlations.csv',index=False)
    old=pd.read_csv(record(ROOT/'experiments/eswa_response_free_risk/model/development_run_audit.csv'))
    merged=run.merge(old,on=['target','fold','seed'])
    diff=float(np.abs(merged.training_internal_correlation-merged.risk_harm_correlation).max())
    assert diff<1e-8
    result={'development_rows':len(f),'development_runs':len(run),'tasks':int(f.target.nunique()),
            'unique_standardized_molecules':len(grouped),'molecules_repeated':int((grouped.records>1).sum()),
            'molecules_across_tasks':int((grouped.tasks>1).sum()),
            'molecules_across_outer_fold_numbers':int((grouped.outer_folds>1).sum()),
            'molecules_across_seeds':int((grouped.seeds>1).sum()),
            'training_internal_harm_AUROC':float(roc_auc_score(f.harmed,score)),
            'global_training_internal_pearson':float(np.corrcoef(score,f.normalized_point_harm)[0,1]),
            'reproduced_original_run_correlation_max_absolute_error':diff,
            'selector_level_OOF_found':False,'original_selector_scores':'fitted on all development rows; predicted on same rows',
            'base_model_OOF_is_not_selector_OOF':True,'coverage_quantiles_source':'training-internal MoleculeACE development scores',
            'external_AqSolDB_used_for_fit_or_quantiles':False,
            'new_selector_fit_performed':False,
            'interpretation':'Fig2(c) is training-internal mechanism diagnosis. Existing AqSolDB is separate external evaluation.'}
    print(json.dumps(result,indent=2),flush=True)
    return result


def fair_operators(y,left,right,target,seed,scales,thresholds):
    yield 'identity','identity',1.,0.,y.copy(),0.
    # Exact same deterministic random-edge rule and naive/Laplacian formulas as S7.
    for tau in thresholds:
        active=np.abs(y[right]-y[left])>=tau
        al,ar=left[active],right[active]
        token=f'2917:earliest:{target}:{seed}:{tau:.3f}'.encode()
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'))
        chosen=np.sort(rng.choice(len(left),size=int(active.sum()),replace=False)) if active.any() else np.array([],dtype=int)
        degree=np.bincount(np.r_[al,ar],minlength=len(y))
        neighbor_sum=np.bincount(np.r_[al,ar],weights=np.r_[y[ar],y[al]],minlength=len(y))
        unsharp=np.zeros(len(y));mask=degree>0
        unsharp[mask]=y[mask]-neighbor_sum[mask]/degree[mask]
        delta=y[ar]-y[al]
        higher=np.where(delta>=0,ar,al)
        count=np.bincount(higher,minlength=len(y));sums=np.bincount(higher,weights=np.abs(delta),minlength=len(y))
        naive=np.divide(sums,count,out=np.zeros(len(y)),where=count>0)
        centered=project(y,al,ar,2.)[0]-y
        random_centered=project(y,left[chosen],right[chosen],2.)[0]-y
        for scale in scales:
            for family,direction in [('BPS',centered),('Laplacian_unsharp',unsharp),('random_edge_scaling',random_centered),('naive_active_edge_push',naive)]:
                z=y+(scale-1.)*direction
                yield family,f'{family}_s{scale:.2f}_t{tau:.2f}',scale,tau,z,float(np.linalg.norm(z-y))


def fair(args,dest):
    dataroot=ROOT/'data/processed/chembl36_temporal_earliest'
    predroot=ROOT/'experiments/chembl36_temporal_earliest/predictions'
    targets=sorted(p.name for p in predroot.iterdir() if p.is_dir())
    if args.smoke:targets=targets[:1]
    scales=[1.10,1.25,1.50,1.75];thresholds=[.25,.5,.75]
    rows=[];choices=[];gridrows=[]
    for ti,target in enumerate(targets):
        train,valid,test=temporal_split_ids(dataroot,target)
        assert not (set(valid)&set(test))
        all_edges=pd.read_parquet(record(ROOT/f'graphs/chembl36_temporal_mmp/{target}/eligible_mmp_edges.parquet'))
        val_svm=readpred(ROOT/f'experiments/chembl36_temporal_earliest/models/svm/{target}/seed_0/validation_predictions.parquet')
        for seed in ([0] if args.smoke else [0,1,2]):
            rf=readpred(ROOT/f'experiments/chembl36_temporal_earliest/models/rf/{target}/seed_{seed}/validation_predictions.parquet')
            gine=readpred(OUT/f'validation_predictions/{target}/seed_{seed}/predictions.parquet')
            assert set(val_svm.index)==set(valid)==set(rf.index)==set(gine.index)
            index=val_svm.index
            y=.5*val_svm.to_numpy()+.25*rf.reindex(index).to_numpy()+.25*gine.reindex(index).to_numpy()
            left,right,_=graph(index,all_edges)
            # Only validation responses are read before all method parameters are written.
            vtruth=readtruth(dataroot,target,index).set_index('mol_id').y_pactivity.reindex(index).to_numpy()
            ref=metrics(y,vtruth,left,right)
            candidates={}
            for family,name,scale,tau,z,norm in fair_operators(y,left,right,target,seed,scales,thresholds):
                m=metrics(z,vtruth,left,right)
                recordrow={'target':target,'seed':seed,'family':family,'candidate':name,'scale':scale,'threshold':tau,
                    'correction_l2':norm,'identity':family=='identity',**m,
                    'validation_point_cost':float(m['overall_mae']/ref['overall_mae']-1.),
                    'prediction_hash':arrhash(z)}
                candidates[name]=recordrow;gridrows.append(recordrow)
            selected=[]
            for family in ('BPS','Laplacian_unsharp','random_edge_scaling','naive_active_edge_push'):
                available=[v for v in candidates.values() if v['family'] in (family,'identity') and v['validation_point_cost']<=.01+1e-12]
                if ref['n_cliff_edges']==0:
                    best=candidates['identity'];reason='no_validation_true_cliff_pairs_identity'
                else:
                    best=min(available,key=lambda v:(v['pair_delta_mae'],v['overall_mae'],v['correction_l2'],not v['identity'],v['candidate']))
                    reason='validation_pair_minimum_under_point_budget'
                selected.append({'target':target,'seed':seed,'method':'calibrated_'+family,
                    'chosen_candidate':best['candidate'],'scale':best['scale'],'threshold':best['threshold'],
                    'validation_point_cost':best['validation_point_cost'],'validation_pair_mae':best['pair_delta_mae'],
                    'validation_cliff_pairs':ref['n_cliff_edges'],'validation_candidates_per_family':13,
                    'identity_selected':best['identity'],'selection_reason':reason,'test_response_used_for_selection':False})
            # Concrete, flushed per-run selection record BEFORE test-label materialization.
            selected_path=dest/'selection_records'/target/f'seed_{seed}.json'
            save_json(selected_path,{'selected':selected,'test_labels_loaded':False,'time_unix':time.time()})
            choices.extend(selected)
            base=readpred(predroot/target/f'seed_{seed}/unscaled_reference/predictions.parquet')
            assert set(base.index)==set(test)
            ty=base.to_numpy();tl,tr,_=graph(base.index,all_edges)
            options={name:z for family,name,s,t,z,norm in fair_operators(ty,tl,tr,target,seed,scales,thresholds)}
            predictions={'identity':ty,'canonical_BPS':options['BPS_s1.50_t0.50']}
            existing=readpred(predroot/target/f'seed_{seed}/bps/predictions.parquet').reindex(base.index).to_numpy()
            assert np.max(np.abs(existing-predictions['canonical_BPS']),initial=0)<1e-9
            for rec in selected:predictions[rec['method']]=options[rec['chosen_candidate']]
            for name,z in predictions.items():
                op=dest/'predictions'/target/f'seed_{seed}'/name/'predictions.parquet'
                freeze_prediction_file(pd.DataFrame({'mol_id':base.index,'prediction':z}),op,
                    {'target':target,'seed':seed,'method':name,'test_labels_materialized':False,
                     'selection_record_sha256':sha256_file(selected_path),'protocol':'reviewer_20261020_validation_only'})
            # Test labels enter only after rule and all candidate outputs are frozen.
            truth=readtruth(dataroot,target,base.index).set_index('mol_id').y_pactivity.reindex(base.index).to_numpy()
            for name,z in predictions.items():
                rows.append({'target':target,'fold':0,'seed':seed,'method':name,**metrics(z,truth,tl,tr)})
        print('fair target',ti+1,'/',len(targets),target,flush=True)
    raw=pd.DataFrame(rows);sel=pd.DataFrame(choices)
    raw.to_csv(dest/'test_run_metrics.csv',index=False);sel.to_csv(dest/'validation_selected_parameters.csv',index=False)
    pd.DataFrame(gridrows).to_csv(dest/'validation_candidate_grid_metrics.csv',index=False)
    target=raw.groupby(['method','target'],as_index=False)[METRICS+['all_mmp_pair_mae']].mean()
    target.to_csv(dest/'test_target_metrics.csv',index=False)
    summary=[];rng=np.random.default_rng(291020)
    ref=target[target.method=='identity'].set_index('target')
    for method,g in target.groupby('method'):
        g=g.set_index('target')
        for metric in METRICS+['all_mmp_pair_mae']:
            a=g[metric];b=ref[metric];eligible=a.notna()&b.notna()&(b>0)
            av=a[eligible].to_numpy();bv=b[eligible].to_numpy()
            draws=rng.integers(0,len(av),size=(10000,len(av)))
            boot=av[draws].mean(axis=1)/bv[draws].mean(axis=1)-1.
            summary.append({'method':method,'metric':metric,'eligible_targets':len(av),
                'raw_macro':float(av.mean()),'identity_macro':float(bv.mean()),
                'relative_change_ratio_of_macro':float(av.mean()/bv.mean()-1.),
                'relative_change_mean_of_target_ratios':float(np.mean(av/bv-1.)),
                'bootstrap95_lower':float(np.quantile(boot,.025)),'bootstrap95_upper':float(np.quantile(boot,.975)),
                'favorable_targets':int((av<bv).sum()),'bootstrap_unit':'target_after_seed_average'})
    pd.DataFrame(summary).to_csv(dest/'fair_comparison_summary.csv',index=False)
    contrasts=[]
    bps=target[target.method=='calibrated_BPS'].set_index('target')
    for reference_method in ('canonical_BPS','calibrated_Laplacian_unsharp','calibrated_random_edge_scaling','calibrated_naive_active_edge_push'):
        baseline=target[target.method==reference_method].set_index('target')
        for metric in METRICS:
            pair=pd.concat([bps[metric].rename('candidate'),baseline[metric].rename('reference')],axis=1).dropna()
            av=pair.candidate.to_numpy();bv=pair.reference.to_numpy()
            draw=rng.integers(0,len(av),size=(10000,len(av)))
            boot=av[draw].mean(axis=1)/bv[draw].mean(axis=1)-1.
            contrasts.append({'candidate':'calibrated_BPS','reference':reference_method,'metric':metric,
                'eligible_targets':len(av),'relative_change':float(av.mean()/bv.mean()-1),
                'bootstrap95_lower':float(np.quantile(boot,.025)),'bootstrap95_upper':float(np.quantile(boot,.975)),
                'favorable_targets':int((av<bv).sum()),'tied_targets':int(np.isclose(av,bv,atol=1e-12,rtol=0).sum())})
    pd.DataFrame(contrasts).to_csv(dest/'paired_method_contrasts.csv',index=False)
    sel.groupby('method').agg(runs=('identity_selected','size'),identity_count=('identity_selected','sum'),identity_fraction=('identity_selected','mean')).to_csv(dest/'identity_selection_summary.csv')
    return {'targets':len(targets),'seeds_per_target':1 if args.smoke else 3,'test_runs':len(raw)//6,
            'new_training':False,'test_used_for_parameter_choice':False,
            'data_version':'earliest accepted publication year only','matched_posthoc_controls_reused_as_deployable':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['mechanism','structure','perturb','selector','fair'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    if hasattr(os, 'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cpus[:max(1, min(32, len(cpus) - 30))])
    dest=OUT/('smoke' if args.smoke else 'full')/args.stage
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():
        print('EXISTING_COMPLETE',dest);return
    started=time.time()
    try:
        result={'mechanism':mechanism,'structure':structure,'perturb':perturb,
                'selector':selector_audit,'fair':fair}[args.stage](args,dest)
    except Exception:
        save_json(dest/'source_hashes_partial.json',SOURCES)
        raise
    save_json(dest/'source_hashes.json',SOURCES)
    save_json(dest/'DONE.json',{'stage':args.stage,'smoke':args.smoke,'duration_seconds':time.time()-started,
        'script_sha256':sha256_file(Path(__file__)),'protocol_sha256':sha256_file(OUT/'PROTOCOL_FROZEN.md'),**result})
    print('DONE',args.stage,round(time.time()-started,2),flush=True)


if __name__=='__main__':main()
