#!/usr/bin/env python3
"""Verify the lightweight frozen structural-audit archive without raw data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/structural_audits'
MANIFEST = ARCHIVE / 'release_manifest.json'
SOURCE_FILES = [
    'scripts/run_structural_audits.py', 'scripts/infer_audit_validation.py',
    'scripts/verify_structural_audits.py', 'scripts/finalize_structural_audits.py',
    'scripts/evaluate_audit_matched_controls.py', 'scripts/build_structural_audit_tables.py',
    'scripts/audit_structural_release.py', 'protocol/structural_audits.md',
    'reproduction/structural_audits.md',
]
PRIVATE_PATTERNS = [
    r'/home/[^/\s]+/', r'[A-Za-z]:[\\/](?:Users|我的论文|test)[\\/]',
    r'192\.168\.\d+\.\d+', r'ghp_[A-Za-z0-9]{20,}',
    r'-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) KEY-----',
]


def files():
    return sorted([p for p in ARCHIVE.rglob('*') if p.is_file() and p != MANIFEST] +
                  [ROOT / name for name in SOURCE_FILES])


def rows(name):
    with (ARCHIVE / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, tolerance=1e-10):
    assert abs(float(actual) - expected) <= tolerance, (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Freeze hashes after checking a prepared release.')
    args = parser.parse_args()
    inventory = {}
    for path in files():
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
        for pattern in ([] if path.resolve() == Path(__file__).resolve() else PRIVATE_PATTERNS):
            if re.search(pattern, text, flags=re.I):
                raise RuntimeError(f'Private/local information pattern found in {path.relative_to(ROOT)}')
        inventory[path.relative_to(ROOT).as_posix()] = {
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding='utf-8')
    assert json.loads(MANIFEST.read_text(encoding='utf-8')) == inventory, 'Frozen release hash mismatch'

    selected = rows('full/fair/validation_selected_parameters.csv')
    test = rows('full/fair/test_run_metrics.csv')
    assert len(selected) == 432 and len(test) == 648
    assert all(row['test_response_used_for_selection'].lower() == 'false' for row in selected)
    assert all(float(row['validation_point_cost']) <= .01+1e-12 for row in selected)
    assert all(int(row['validation_candidates_per_family']) == 13 for row in selected)
    assert len(list((ARCHIVE/'full/fair/selection_records').glob('*/seed_*.json'))) == 108

    fair = {(r['method'], r['metric']): r for r in rows('full/fair/fair_comparison_summary.csv')}
    close(fair['calibrated_BPS', 'pair_delta_mae']['relative_change_ratio_of_macro'], -.03386431617066776)
    close(fair['calibrated_BPS', 'overall_mae']['relative_change_ratio_of_macro'], .00554934667368423)
    assert float(fair['calibrated_BPS', 'cliff_endpoint_mae']['bootstrap95_upper']) > .01

    cohort = rows('mechanism_cohort_summaries.csv')
    for dataset, scope, expected in [
        ('MoleculeACE', 'manuscript_23_MoleculeACE_targets', .3084962980885009),
        ('corrected_ChEMBL', 'all_eligible_tasks', .26743167093424486),
    ]:
        row = next(r for r in cohort if r['dataset'] == dataset and r['cohort'] == scope
                   and r['model'] == 'mixture' and r['edge_scope'] == 'all_MMP')
        close(row['improved_fraction'], expected)

    perturb = rows('full/perturb/perturbation_run_metrics.csv')
    assert len(perturb) == 150
    assert max(float(r['fixed_graph_lipschitz_ratio']) for r in perturb) <= 1.5+1e-9
    fit = json.loads((ARCHIVE/'selector_fit_provenance.json').read_text(encoding='utf-8'))
    assert fit['fit_input_rows'] == 144297 and fit['actual_iterations'] == 152
    assert fit['selector_level_independent_OOF'] is False
    decision = json.loads((ARCHIVE/'interpretation.json').read_text(encoding='utf-8'))
    assert decision['AqSolDB_subsequent_deployment_authorization'] == 'NO-GO'
    assert decision['authorization_did_not_modify_evaluated_candidate_predictions'] is True
    assert decision['pair_dominance_over_Laplacian_established'] is False
    report = {'release_files_verified': len(inventory),
              'released_bytes': sum(v['bytes'] for v in inventory.values()),
              'validation_selections': len(selected), 'fair_metric_rows': len(test),
              'perturbation_runs': len(perturb), 'privacy_scan': 'PASS',
              'raw_data_experiments_rerun': False,
              'verification_role': 'Release consistency, not independent scientific replication'}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
