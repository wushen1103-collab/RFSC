# DeepDelta cycle projection implementation audit

Date: 2026-07-19

The first benchmark-response evaluation was rejected by its preregistered sanity
check: the Pythagorean identity failed, which is impossible for the exact
orthogonal projection evaluated against a gradient truth flow.

Root cause: `freeze_prediction_file` intentionally sorts generic molecule IDs
lexicographically. The response-free DeepDelta artifact used IDs such as
`position_2` and `position_10`, while the evaluator consumed the sorted series as
an unnamed array. Consequently, every fold with at least 11 nodes reconstructed
the projected matrix in the wrong node order.

Corrective action:

- keep all 100 frozen potential files and their hashes unchanged;
- restore the numeric order explicitly by the frozen `position_<integer>` IDs;
- add a regression test covering the `position_2`/`position_10` case;
- preserve the rejected output under
  `evaluation_rejected_lexicographic_order/`;
- rerun the unchanged preregistered metrics and decision gates once.

No benchmark response, metric value, dataset-specific choice, hyperparameter,
candidate value, or decision threshold is used by this repair. Because the bug
was discovered after responses were first materialized, the corrected result is
reported as a protocol-conformant implementation repair rather than a second
independent external confirmation.
