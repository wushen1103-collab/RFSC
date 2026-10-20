"""Executor reproducibility checks using the project's original operators/evaluator."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(os.environ.get('RFSC_WORKSPACE_ROOT',Path(__file__).resolve().parents[1])).resolve()
OUT=Path(os.environ.get('RFSC_AUDIT_OUTPUT',ROOT/'reproduced/structural_audits')).resolve()
sys.path.insert(0,str(ROOT))
from bps_gnn.io import read_frozen_prediction,read_labels_for_ids,sha256_file
from bps_gnn.projection import boundary_projection,select_active_edges
from scripts.evaluate_chembl36_temporal import metric_row


def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')


def main():
    stages=('mechanism','fair','structure','perturb','selector')
    for stage in stages:
        done=json.loads((OUT/f'full/{stage}/DONE.json').read_text())
        assert done['smoke'] is False
    raw=pd.read_csv(OUT/'full/fair/test_run_metrics.csv')
    selected=pd.read_csv(OUT/'full/fair/validation_selected_parameters.csv')
    assert len(raw)==648 and len(selected)==432
    assert not selected.test_response_used_for_selection.any()
    assert (selected.validation_point_cost<=.01+1e-12).all()
    assert (selected.validation_candidates_per_family==13).all()
    max_metric_error=0.;max_operator_error=0.
    # Reuse the exact S7 implementations as a numerical conformance check.
    spec=importlib.util.spec_from_file_location('s7',Path(__file__).resolve().parent/'evaluate_audit_matched_controls.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for (target,seed),group in raw.groupby(['target','seed']):
        original=ROOT/f'experiments/chembl36_temporal_earliest/predictions/{target}/seed_{seed}'
        base=read_frozen_prediction(original/'unscaled_reference/predictions.parquet')
        all_edges=pd.read_parquet(ROOT/f'graphs/chembl36_temporal_mmp/{target}/eligible_mmp_edges.parquet')
        edges=all_edges[all_edges.mol_id_lo.isin(base.index)&all_edges.mol_id_hi.isin(base.index)]
        chosen=selected[(selected.target==target)&(selected.seed==seed)]
        for row in chosen.itertuples():
            actual=read_frozen_prediction(OUT/f'full/fair/predictions/{target}/seed_{seed}/{row.method}/predictions.parquet')
            if row.identity_selected:
                expected=base
            else:
                active=select_active_edges(base,edges,row.threshold)
                if row.method=='calibrated_BPS':expected=boundary_projection(base,active,row.scale)
                elif row.method=='calibrated_Laplacian_unsharp':expected=old.laplacian_unsharp(base,active,row.scale-1.)
                elif row.method=='calibrated_naive_active_edge_push':expected=old.naive_active_edge_push(base,active,row.scale)
                else:
                    rng=old.stable_rng(2917,'earliest',target,int(seed),row.threshold)
                    picks=np.sort(rng.choice(len(edges),size=len(active),replace=False)) if len(active) else np.array([],dtype=int)
                    expected=boundary_projection(base,edges.iloc[picks],row.scale)
            err=float(np.max(np.abs(actual.reindex(base.index).to_numpy()-expected.reindex(base.index).to_numpy()),initial=0))
            max_operator_error=max(max_operator_error,err)
        truth=read_labels_for_ids(ROOT/'data/processed/chembl36_temporal_earliest',target,base.index).set_index('mol_id').y_pactivity
        for row in group.itertuples():
            prediction=read_frozen_prediction(OUT/f'full/fair/predictions/{target}/seed_{seed}/{row.method}/predictions.parquet')
            original_metrics=metric_row(prediction,truth,edges)
            for metric in ('overall_mae','cliff_endpoint_mae','pair_delta_mae'):
                a,b=original_metrics[metric],getattr(row,metric)
                if np.isnan(a):assert np.isnan(b)
                else:max_metric_error=max(max_metric_error,abs(a-b))
    assert max_operator_error<1e-9 and max_metric_error<1e-10
    canonical=raw[raw.method=='canonical_BPS'].groupby('target')[['overall_mae','cliff_endpoint_mae','pair_delta_mae']].mean()
    previous=pd.read_csv(ROOT/'experiments/chembl36_temporal_earliest/evaluation/primary_statistics.csv').set_index('metric')
    canonical_errors={metric:abs(float(canonical[metric].mean())-float(previous.loc[metric,'candidate_macro'])) for metric in canonical.columns}
    assert max(canonical_errors.values())<1e-10
    # Provenance detail: sklearn's automatic early stopping uses a row-level monitor split.
    frame=pd.read_parquet(ROOT/'results/is_revision2_selector_coverage_curve/development_selector_frame.parquet')
    payload=joblib.load(ROOT/'experiments/eswa_response_free_risk/model/risk_model.joblib')
    model=payload['model']
    train,val=train_test_split(np.arange(len(frame)),test_size=model.validation_fraction,random_state=model._random_seed)
    training_molecules=set(frame.iloc[train].mol_id)
    monitor=frame.iloc[val]
    early={'fit_input_rows':len(frame),'actual_gradient_fit_rows':len(train),'internal_early_stopping_monitor_rows':len(val),
        'early_stopping_parameter':model.early_stopping,'early_stopping_active':bool(model.do_early_stopping_),
        'actual_iterations':model.n_iter_,'validation_fraction':model.validation_fraction,
        'split_random_seed':int(model._random_seed),
        'monitor_rows_sharing_molecule_id_with_training':int(monitor.mol_id.isin(training_molecules).sum()),
        'monitor_fraction_sharing_molecule_with_training':float(monitor.mol_id.isin(training_molecules).mean()),
        'molecule_group_split':False,'selector_level_independent_OOF':False,
        'correct_description':'Development diagnostics combine gradient-fit rows and an automatic row-level early-stopping monitor; they are not molecule-disjoint held-out selector validation.',
        'original_frozen_model_changed':False,'external_responses_used_in_fit_or_thresholds':False}
    save('selector_fit_provenance.json',early)
    hashes={}
    for stage in stages:
        for name,data in json.loads((OUT/f'full/{stage}/source_hashes.json').read_text()).items():
            if name in hashes:assert hashes[name]==data
            hashes[name]=data
    for name,data in hashes.items():assert sha256_file(ROOT/name)==data['sha256']
    original_model_hash=hashes['experiments/eswa_response_free_risk/model/risk_model.joblib']['sha256']
    result={'verification_kind':'executor reproducibility checks, not independent integrity certification',
        'independent_review_status':'REVIEW_UNAVAILABLE','completed_full_stages':list(stages),
        'fair_test_runs':108,'fair_test_rows':len(raw),'validation_selected_rows':len(selected),
        'maximum_original_operator_conformance_error':max_operator_error,
        'maximum_original_evaluator_error':max_metric_error,'canonical_ChEMBL_reproduction_errors':canonical_errors,
        'source_hashes_reverified':len(hashes),'original_risk_model_sha256':original_model_hash,
        'stage_execution_complete':True}
    save('EXECUTOR_VERIFICATION.json',result)
    print(json.dumps(result,indent=2))
    print('EARLY_STOPPING',json.dumps(early,indent=2))
    print('FAIR_SUMMARY')
    print(pd.read_csv(OUT/'full/fair/fair_comparison_summary.csv').to_csv(index=False))


if __name__=='__main__':main()
