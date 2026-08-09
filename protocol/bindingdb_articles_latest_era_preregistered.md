# BindingDB articles target-specific latest-era confirmation freeze

The globally fixed 2022+ split produced zero count-eligible tasks and is closed
without relaxing a threshold. This replacement is frozen after that
structure/year/count audit but before any validation/test response metric,
prediction, model fit, or method comparison.

All source, human/single-chain, Swiss-Prot, exact Ki/Kd, standardization,
earliest-year-only response, duplicate-QC, model, graph, projection, seed,
endpoint, statistical, and decision rules remain identical to
`bindingdb_articles_temporal_preregistered.md`.

The inherited SVM, RF, and GINE hyperparameters are recorded in the dedicated
`*_bindingdb_articles_latest_era.yaml` files. Their only change from the
fixed-calendar configurations is the new data/cache root; no model setting was
selected from this dataset.

The sole design change is the time partition:

1. within each Swiss-Prot/measurement task, count standardized molecules by
   earliest accepted publication year without reading response values;
2. starting from the latest complete year block, assign whole years to test
   until at least 20% of task molecules are assigned;
3. continue backward, assigning whole years to validation until at least 10%
   are assigned;
4. assign every earlier year to train; a publication year can never cross
   partitions;
5. retain the unchanged >=100/20/30 train/validation/test molecule and >=8
   internal-test MMP-edge gates, requiring at least eight eligible tasks.

This is a target-specific latest-era extrapolation, not a universal 2022+ claim.
Because the redesign used no response statistic, its later response evaluation
remains fresh. Nevertheless, both the failed global-calendar eligibility audit
and this adaptive feasibility step must be disclosed in the paper.
