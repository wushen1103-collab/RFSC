"""Build Tables S8--S11 exclusively from the frozen reviewer-audit CSV files.

This is table assembly, not a new experiment, estimator fit, or parameter search.
Run from any directory: python scripts/build_structural_audit_tables.py
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'reproduced/tables'
HERE.mkdir(parents=True, exist_ok=True)
EVIDENCE = ROOT / 'results/structural_audits'
TARGETS23 = set('''CHEMBL1862_Ki CHEMBL1871_Ki CHEMBL2034_Ki CHEMBL2047_EC50
CHEMBL204_Ki CHEMBL2147_Ki CHEMBL214_Ki CHEMBL218_EC50 CHEMBL219_Ki
CHEMBL228_Ki CHEMBL231_Ki CHEMBL233_Ki CHEMBL235_EC50 CHEMBL236_Ki
CHEMBL237_EC50 CHEMBL237_Ki CHEMBL238_Ki CHEMBL239_EC50 CHEMBL244_Ki
CHEMBL262_Ki CHEMBL264_Ki CHEMBL3979_EC50 CHEMBL4792_Ki'''.split())
assert len(TARGETS23) == 23


def read(name):
    if name.startswith(('fair/', 'mechanism/', 'structure/', 'perturb/')):
        name = 'full/' + name
    with (EVIDENCE / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def pct(value, digits=3, signed=False):
    return format(100 * float(value), ('+' if signed else '') + f'.{digits}f') + r'\%'


def num(value, digits=5):
    return f'{float(value):.{digits}f}'


def sci(value):
    mantissa, exponent = f'{float(value):.2e}'.split('e')
    return rf'\({mantissa}\times10^{{{int(exponent)}}}\)'


def ci(row):
    return (pct(row['relative_change_ratio_of_macro'], signed=True) + ' [' +
            pct(row['bootstrap95_lower'], signed=True) + ', ' +
            pct(row['bootstrap95_upper'], signed=True) + ']')


def tabular(spec, headers, rows):
    return '\n'.join([rf'\begin{{tabularx}}{{\textwidth}}{{@{{}}{spec}@{{}}}}',
                      r'\toprule', ' & '.join(headers) + r' \\', r'\midrule',
                      *(' & '.join(row) + r' \\' for row in rows),
                      r'\bottomrule', r'\end{tabularx}'])


def table(caption, label, body):
    return '\n'.join([r'\begin{table}[H]', r'\begin{revisionblock}', r'\centering',
                      rf'\caption{{{caption}}}', rf'\label{{{label}}}',
                      r'\scriptsize', r'\renewcommand{\arraystretch}{1.02}',
                      r'\setlength{\tabcolsep}{3pt}', body,
                      r'\end{revisionblock}', r'\end{table}', ''])


fair = {(r['method'], r['metric']): r for r in read('fair/fair_comparison_summary.csv')}
identities = {r['method']: r for r in read('fair/identity_selection_summary.csv')}
methods = [('identity', 'Identity'), ('canonical_BPS', 'Canonical BPS'),
           ('calibrated_BPS', 'Calibrated BPS'),
           ('calibrated_Laplacian_unsharp', 'Calibrated Laplacian'),
           ('calibrated_random_edge_scaling', 'Calibrated random-edge'),
           ('calibrated_naive_active_edge_push', 'Calibrated naive push')]
s8rows = []
for method, name in methods:
    point = fair[method, 'overall_mae']
    cliff = fair[method, 'cliff_endpoint_mae']
    pair = fair[method, 'pair_delta_mae']
    s8rows.append([name, pct(point['relative_change_ratio_of_macro'], signed=True),
                   pct(cliff['relative_change_ratio_of_macro'], signed=True),
                   ci(pair), '--' if method == 'identity' else pair['favorable_targets'] + '/35',
                   (identities[method]['identity_count'] + '/108') if method in identities else '--'])
contrasts = read('fair/paired_method_contrasts.csv')
s8contrast = []
for method, name in methods[3:]:
    row = next(r for r in contrasts if r['reference'] == method and r['metric'] == 'pair_delta_mae')
    s8contrast.append(['BPS versus ' + name.removeprefix('Calibrated '),
                       pct(row['relative_change'], signed=True),
                       '[' + pct(row['bootstrap95_lower'], signed=True) + ', ' + pct(row['bootstrap95_upper'], signed=True) + ']'])
s8 = table(
    'Validation-only postprocessor comparison on corrected earliest-year ChEMBL. '
    r'All methods use the same mixture, splits, validation budget, and identity option. '
    r'Changes are ratios of target-macro MAEs relative to identity; negative values improve error. '
    r'Ordinary MAE has 36 targets and cliff/pair MAE has 35. Intervals use 10,000 paired target bootstrap resamples after seed averaging. '
    r'The canonical confidence interval is recomputed for this audit; its unchanged point estimate matches Table 4.',
    'tab:supp_fair_validation',
    tabular('Yrrlrr', ['Method', r'\shortstack{\(\Delta\) ordinary}', r'\shortstack{\(\Delta\) cliff}',
                      r'\shortstack{\(\Delta\) pair [95\% CI]}', r'\shortstack{Pair\\favorable}', r'\shortstack{Identity\\selected}'], s8rows)
    + '\n\\par\\vspace{6pt}\n'
    + tabular('Yrr', ['Calibrated paired contrast', r'\(\Delta\) pair', r'95\% CI'], s8contrast))

cohort = read('mechanism_cohort_summaries.csv')
targets = read('mechanism/mechanism_target_metrics.csv')


def subset(rows, dataset):
    return [r for r in rows if r['dataset'] == dataset and
            (dataset != 'MoleculeACE' or r['target'] in TARGETS23)]


def mean(rows, key):
    values = [float(r[key]) for r in rows if r[key] and r[key].lower() != 'nan']
    return statistics.mean(values)


s9rows = []
for dataset, label, coh in [('MoleculeACE', 'MoleculeACE (23)', 'manuscript_23_MoleculeACE_targets'),
                             ('corrected_ChEMBL', 'Corrected ChEMBL', 'all_eligible_tasks')]:
    for model in ['mixture', 'SVM', 'GINE']:
        allrows = [r for r in subset(targets, dataset) if r['model'] == model and r['edge_scope'] == 'all_MMP']
        cliffrows = [r for r in subset(targets, dataset) if r['model'] == model and r['edge_scope'] == 'true_cliff']
        summary = next(r for r in cohort if r['dataset'] == dataset and r['model'] == model and
                       r['cohort'] == coh and r['edge_scope'] == 'true_cliff')
        active_n = summary['active_eligible_tasks']
        s9rows.append([label + ': ' + model, active_n,
                       pct(mean(allrows, 'underestimated_fraction'), 2) + '/' + pct(mean(allrows, 'overestimated_fraction'), 2),
                       pct(mean(allrows, 'improved_fraction'), 2) + '/' + pct(mean(allrows, 'worsened_fraction'), 2),
                       pct(mean(cliffrows, 'underestimated_fraction'), 2) + '/' + pct(mean(cliffrows, 'overestimated_fraction'), 2),
                       pct(mean(cliffrows, 'improved_fraction'), 2) + '/' + pct(mean(cliffrows, 'worsened_fraction'), 2),
                       summary['underestimation_majority_tasks'] + '/' + summary['improvement_majority_tasks']])
bins = read('mechanism/predicted_delta_bins_target.csv')
bin_labels = [('0_to_.25', r'\([0,.25)\)'), ('.25_to_.50', r'\([.25,.50)\)'),
              ('.50_to_.75', r'\([.50,.75)\)'), ('.75_to_1', r'\([.75,1)\)'), ('ge_1', r'\(\geq1\)')]
s9bins = []
for dataset, label in [('MoleculeACE', 'MoleculeACE (23)'), ('corrected_ChEMBL', 'Corrected ChEMBL')]:
    for scope, scope_label in [('all_MMP', 'all MMPs'), ('true_cliff', 'true cliffs')]:
        for key, value in bin_labels:
            rows = [r for r in subset(bins, dataset) if r['model'] == 'mixture' and r['edge_scope'] == scope and r['bin'] == key]
            s9bins.append([label + ', ' + scope_label, value,
                           num(mean(rows, 'mean_abs_predicted_delta'), 3),
                           num(mean(rows, 'mean_abs_true_delta'), 3),
                           num(mean(rows, 'pair_mae_before'), 3), pct(mean(rows, 'underestimated_fraction'), 2)])
s9 = table(
    r'Underestimation mechanism and predicted-amplitude bins. Panel A summarizes active edges at \(\tau=0.5,s=1.5\); '
    r'all-MMP and true-cliff under/over and improve/worsen fractions are reported separately. Improvement/worsening uses signed-difference absolute error, '
    r'not amplitude alone. Fractions are equal-weight task summaries after run averaging, not pooled-edge inference. '
    r'The final column counts true-cliff tasks with a majority of underestimated/improved edges. '
    r'Panel B uses the mixture, includes inactive bins, and averages available task-bin values; '
    r'\(\hat\Delta=\hat y_j-\hat y_i\) and \(\Delta^\star=y_j-y_i\). True cliffs are evaluator-only.',
    'tab:supp_underestimation',
    r'\noindent Panel A: active-edge mechanism\\[3pt]' + '\n' +
    tabular('Yrrrrrr', ['Cohort and predictor', r'\shortstack{Cliff\\tasks}', r'\shortstack{All MMP\\under/over}',
                      r'\shortstack{All active MMP\\improve/worsen}',
                      r'\shortstack{True cliff\\under/over}', r'\shortstack{True cliff\\improve/worsen}', r'\shortstack{Majority\\under/improve}'], s9rows)
    + '\n\\par\\vspace{6pt}\\noindent Panel B: mixture amplitude bins\\\\[3pt]\n'
    + tabular('Ylrrrr', ['Cohort and evaluation edges', r'Predicted \(|\Delta|\)', r'Mean \(|\hat\Delta|\)',
                        r'Mean \(|\Delta^\star|\)', 'Pair MAE', 'Under'], s9bins))

constraints = read('structure/constraint_pooled_descriptive.csv')
s10edge = []
for method, group, label in [('full_BPS', 'all_active', 'Full BPS: all active'),
                             ('hybrid_candidate', 'both_accepted', 'Hybrid: both accepted'),
                             ('hybrid_candidate', 'one_accepted', 'Hybrid: one accepted'),
                             ('hybrid_candidate', 'both_rejected', 'Hybrid: both rejected')]:
    row = next(r for r in constraints if r['method'] == method and r['group'] == group)
    tiny = method == 'full_BPS' or group == 'both_accepted'
    fmt = sci if tiny else num
    s10edge.append([label, f"{int(row['count']):,}", fmt(row['mean']), fmt(row['median']), fmt(row['p95']), fmt(row['maximum'])])
s10component = []
for row in read('structure/component_mean_pooled_descriptive.csv'):
    if row['scope'] != 'nontrivial_components':
        continue
    tiny = row['method'] == 'full_BPS'
    fmt = sci if tiny else num
    s10component.append(['Full BPS' if tiny else 'Candidate hybrid', row['count'], fmt(row['mean']), fmt(row['p95']),
                         fmt(row['maximum']), pct(row['nonzero_fraction'], 2)])
s10 = table(
    r'Structural-contract audit of selective execution with the original AqSolDB nominal 80\% candidate selector across 15 fold--seed runs. '
    r'Panel A reports \(|(z_j-z_i)-s(\hat y_j-\hat y_i)|\) on original active edges. '
    r'Both-rejected edges return identity, so their nonzero amplification residual is expected. '
    r'The 23,047 records include repeated seeds and are descriptive; their denominator differs from true-cliff-pair coverage. '
    r'Panel B reports absolute component-mean drift on 1,662 nontrivial original components, avoiding dilution by isolated nodes. '
    r'Nonzero drift means greater than \(10^{-10}\); magnitudes use the original property units.',
    'tab:supp_hybrid_constraints',
    r'\noindent Panel A: active-edge amplification residuals\\[3pt]' + '\n' +
    tabular('Yrrrrr', ['Output and edge status', 'Records', 'Mean', 'Median', '95th pct.', 'Maximum'], s10edge)
    + '\n\\par\\vspace{6pt}\\noindent Panel B: nontrivial-component mean drift\\\\[3pt]\n'
    + tabular('Yrrrrr', ['Output', 'Components', 'Mean', '95th pct.', 'Maximum', 'Nonzero'], s10component))

perturb = read('perturb/perturbation_summary.csv')
perturb.sort(key=lambda r: float(r['epsilon']))
s11rows = []
for field, label, percent in [
    ('added_fraction_all_MMP', 'Added active edges / all MMPs', True),
    ('deleted_fraction_all_MMP', 'Deleted active edges / all MMPs', True),
    ('flip_fraction_all_MMP', 'Activation flips / all MMPs', True),
    ('component_membership_changed_fraction', 'Node component-membership changes', True),
    ('selector_flip_fraction', 'Node selector-acceptance flips', True),
    ('full_output_change_mae', 'Full BPS mean absolute output change', False),
    ('full_output_change_max_worst_run', 'Full BPS worst-node output change', False),
    ('hybrid_output_change_mae', 'Candidate hybrid mean absolute output change', False),
    ('hybrid_output_change_max_worst_run', 'Candidate hybrid worst-node output change', False),
    ('fixed_graph_change_mae', 'Fixed-graph mean absolute output change', False),
    ('fixed_graph_lipschitz_ratio_worst_run', r'Maximum fixed-graph \(L_2\) change ratio', False)]:
    s11rows.append([label, *((pct(r[field]) if percent else num(r[field])) for r in perturb)])
s11 = table(
    r'Prediction-perturbation stability on the original AqSolDB selector: 15 fold--seed runs, two noise amplitudes, '
    r'and five noise seeds, totaling 150 perturbations. The same nodewise uniform noise is added to all three predictors; '
    r'active graphs and selector features are rebuilt, while \(\tau=0.5,s=1.5\) and the selector remain frozen. '
    r'Means first average within folds over noise and model seeds and then across folds; worst-node values are maxima over all runs. '
    r'The fixed-graph ratio is output-to-input Euclidean change and must not exceed 1.5. No thresholds are selected from this audit.',
    'tab:supp_perturbation',
    tabular('Yrr', ['Diagnostic', r'\(\epsilon=0.01\)', r'\(\epsilon=0.05\)'], s11rows))

output = r'''% Generated by build_revision_tables.py from frozen CSV files; do not hand-edit numbers.
\begin{revisionblock}
\noindent\textbf{Additional audit protocols.}
The validation-only controls share the scale/threshold grid and identity fallback in the main text.
Each family has 12 candidates, \(s\in\{1.10,1.25,1.50,1.75\}\) and
\(\tau\in\{0.25,0.50,0.75\}\), plus identity.
Validation selects minimum cliff-pair MAE under an ordinary-MAE cost of at most +1\%;
ties favor lower ordinary MAE, then smaller correction norm, then identity.
Without validation cliff pairs, selection returns identity. Chosen parameters and prediction hashes
are recorded before reading test responses.
On the active MMP graph, Laplacian unsharp uses the base prediction minus its neighbor mean
(zero for isolated nodes); naive active-edge push adds each node's mean absolute active difference
only when that node is the higher-prediction endpoint. Both directions are multiplied by \(s-1\).
Random-edge scaling samples, without replacement, the same number of MMP edges as the active set,
then applies component-centered scaling on that sampled graph using a deterministic seed.
These rules match the earlier diagnostic implementations; only validation evidence selects their parameters here.
Calibrated BPS's ordinary-cost interval is [+0.296\%, +0.860\%], while its cliff-endpoint interval
is [\(-0.148\%\), +1.470\%]; satisfying the validation budget alone does not establish test noninferiority.
Tables~\ref{tab:supp_fair_validation}--\ref{tab:supp_perturbation} contain the four new audits.
\end{revisionblock}

'''
output += '\n'.join([s8, s9, s10, s11])
output = output.replace('\\begin{revisionblock}\n', '').replace('\\end{revisionblock}\n', '')
(HERE / 'structural_audits.tex').write_text(output, encoding='utf-8')
manifest = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(EVIDENCE.rglob('*')) if p.is_file()}
manifest['reproduced/tables/structural_audits.tex'] = hashlib.sha256(output.encode('utf-8')).hexdigest()
(HERE / 'structural_audit_evidence_hashes.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f'Generated 4 tables; {len(manifest)-1} evidence files hashed; no experiment or fitting performed.')
