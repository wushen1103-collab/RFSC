"""Additional derived summaries and final resource/artifact checks."""
from pathlib import Path
import json
import os
import hashlib
import sys
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('RFSC_WORKSPACE_ROOT',Path(__file__).resolve().parents[1])).resolve()
OUT=Path(os.environ.get('RFSC_AUDIT_OUTPUT',ROOT/'reproduced/structural_audits')).resolve()
sys.path.insert(0,str(ROOT))
from bps_gnn.io import read_frozen_prediction,sha256_file


def main():
    cohort=sorted(p.name for p in (ROOT/'experiments/svm_bps_selected/phase4_secondary').iterdir() if p.is_dir())
    target=pd.read_csv(OUT/'full/mechanism/mechanism_target_metrics.csv')
    rows=[]
    for label,frame in [('all_eligible_tasks',target),('manuscript_23_MoleculeACE_targets',target[(target.dataset=='MoleculeACE')&target.target.isin(cohort)])]:
        for keys,g in frame.groupby(['dataset','model','edge_scope']):
            elig=g.dropna(subset=['active_pair_mae_before'])
            rows.append({'cohort':label,'dataset':keys[0],'model':keys[1],'edge_scope':keys[2],
                'tasks':len(g),'active_eligible_tasks':len(elig),
                'underestimated_fraction':float(elig.underestimated_fraction.mean()),
                'improved_fraction':float(elig.improved_fraction.mean()),
                'all_pair_relative_change_macro':float(g.all_pair_mae_after.mean()/g.all_pair_mae_before.mean()-1),
                'all_pair_relative_change_mean_of_target_ratios':float((g.all_pair_mae_after/g.all_pair_mae_before-1).mean()),
                'active_pair_relative_change_macro':float(elig.active_pair_mae_after.mean()/elig.active_pair_mae_before.mean()-1),
                'underestimation_majority_tasks':int((elig.underestimated_fraction>.5).sum()),
                'improvement_majority_tasks':int((elig.improved_fraction>.5).sum())})
    pd.DataFrame(rows).to_csv(OUT/'mechanism_cohort_summaries.csv',index=False)
    checked=0
    for p in (OUT/'validation_predictions').glob('*/seed_*/predictions.parquet'):
        pred=read_frozen_prediction(p)
        md=json.loads(p.with_suffix('.parquet.sha256.json').read_text())['metadata']
        assert md['partition']=='validation' and md['model_retrained'] is False
        assert md['validation_labels_materialized'] is False and md['test_labels_materialized'] is False
        assert sha256_file(ROOT/md['source_checkpoint'])==md['source_checkpoint_sha256']
        checked+=1
    assert checked==108
    ci=pd.read_csv(OUT/'full/fair/fair_comparison_summary.csv')
    narrow=ci[(ci.method=='calibrated_BPS')&ci.metric.isin(['overall_mae','cliff_endpoint_mae'])]
    fullgate={r.metric:bool(r.bootstrap95_upper<=.01) for r in narrow.itertuples()}
    primary=target[(target.dataset=='MoleculeACE')&(target.model=='mixture')&(target.edge_scope=='true_cliff')&target.target.isin(cohort)]
    effect=float((primary.all_pair_mae_after/primary.all_pair_mae_before-1).mean())
    old=pd.read_csv(ROOT/'paper/tables/svm_bps/table_svm_bps_phase4_paired_statistics.csv')
    original=float(old[(old.baseline=='SVM/RF/GINE equal-shape mix')&(old.metric=='pair_delta_mae')].relative_change_pct.iloc[0])/100.
    assert abs(effect-original)<1e-10
    integrity={'regenerated_validation_predictions':checked,'checkpoint_hash_checks':checked,
               'MoleculeACE_23_task_original_effect_reproduced':effect,
               'MoleculeACE_original_effect_absolute_error':abs(effect-original),
               'checkpoint_retraining_performed':False,'calibrated_BPS_test_noninferiority_CI_within_1pct':fullgate,
               'full_point_and_cliff_gate_passed':all(fullgate.values()),'independent_review_status':'REVIEW_UNAVAILABLE'}
    (OUT/'FINAL_CHECKS.json').write_text(json.dumps(integrity,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(integrity,indent=2))
    print(pd.DataFrame(rows).query("cohort=='manuscript_23_MoleculeACE_targets'").to_csv(index=False))


if __name__=='__main__':main()
