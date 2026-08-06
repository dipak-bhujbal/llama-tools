# Fixture reproduction record — 2026-08-04

This is the committed artifact behind the fixture set. Under Ground Rule 3 a
number is only reportable if a run artifact stands behind it; this file is that
artifact for the fixture self-test, and records the chain of custody from
untrusted intake to promoted `tests/fixtures/` asset.

**What this record does not cover.** HANDOFF.md §2.2 claims a verifier gate of
"1,600/1,600 pass, 0 false positives, 0 misses". That is a claim about
`mining/mine_pairs.py`'s verifier, not about these files, and it is **not**
established here. What is established below is that the 1,600 generated pairs
are structurally well-formed against their own tool schemas, and that the
committed bytes are exactly what the pinned seed produces.

## 1. Intake (untrusted)

Four files were supplied as chat attachments and copied unmodified to
`intake/quarantine/2026-08-04-chat-attachments/` in commit `ce4f7db`; a fifth,
`mine_pairs.py`, arrived separately and was quarantined in this commit. Receipt
and static-inspection results are in that directory's `README.md`.

As-delivered SHA-256:

| File | SHA-256 |
|---|---|
| `scale_pairs_fixed.py` | `53f3b7b9b1df420781bf1274cf45231bd35e935447e6d1381c5a8fbf39052810` |
| `fixture_pairs_train.jsonl` | `6b0215aae4bf18c806954916bbd889df0d8de1019292af2169cc86f35f0b8d04` |
| `fixture_pairs_eval.jsonl` | `ae862c34c216c01f99a263400684fcbb1229e8ece7cf17e8c297c050c8232807` |
| `fixture_audit_sample_50.jsonl` | `88e56ef65f18a7be7506de08f89d219ff9075e7f118cc9f27a774f764665265f` |

## 2. The delivered JSONLs reproduce from the delivered generator

Before any edit, the supplied generator was run in an isolated directory outside
the repository, with only its three hardcoded `/home/claude/dpo_review/...`
output paths redirected. All three delivered files regenerated **byte-identical**
to the hashes in §1.

This is what made promotion defensible: the attachments are the deterministic
output of the attached generator, not unexplained blobs.

## 3. Promotion (one content change, deliberate)

`scale_pairs_fixed.py` was promoted to `tests/fixtures/` with these edits:

1. **Output paths.** The three absolute `/home/claude/dpo_review/...` constants
   became `--out-dir`, defaulting to the script's own directory. Adds `--check`
   (regenerate to a temp dir and byte-compare) and `--print-digests`.
2. **Provenance corrected.** Every row's `meta.provenance` named `scale_pairs.py`
   — a file that does not exist here, and whose name is on the purge list in
   `scripts/check_artifact_boundaries.py`. It now names
   `tests/fixtures/scale_pairs_fixed.py`, the file that actually produced it.
3. **Docstring.** A literal `\n` escape in the header was rendered, and the
   fixture-self-test / verifier-gate distinction stated explicitly.
4. **Structure.** Writing and summary moved under `main()`; generation and its
   assertions stay at module scope, with RNG call order untouched so the seed
   still determines the output.

Edit 2 changes row bytes, so the promoted files hash differently from §1. That
difference was verified to be **entirely** attributable to the rename:

```
diff <(sed 's#tests/fixtures/scale_pairs_fixed\.py#scale_pairs.py#g' tests/fixtures/FILE) \
     intake/quarantine/2026-08-04-chat-attachments/FILE
```

is empty for all three files. No pair, prompt, completion, split assignment, or
error-type label changed.

Promoted SHA-256:

| File | SHA-256 |
|---|---|
| `fixture_pairs_train.jsonl` | `83e12a3e2f374d3e440b1e61c7d3b71cb50c7f2379bf2ef51b0b2be1c9e1ac0c` |
| `fixture_pairs_eval.jsonl` | `f4228bacce224dff52be7e0bb53524dcf7019b94c09ce0fc3ef0cad0a9b10b87` |
| `fixture_audit_sample_50.jsonl` | `4801cd616f93093633eeb6991c03a685d84a869208aeceb77ad88c4390bbbb32` |

## 4. Reproduction check (run this)

```
python tests/fixtures/scale_pairs_fixed.py --check
```

Output:

```
total: 1600  train: 1440  eval: 160 (10% held out, stratified)  audit sample: 50
unique prompts: 1600   max length gap: 40% (floor 40%)
syntax-error pairs: 80/1600 (5%, cap 5%)

error-type mix:
  wrong_param_value             400  (25%)
  wrong_function_selection      400  (25%)
  missing_required_parameter    240  (15%)
  spurious_tool_call            240  (15%)
  missed_tool_call              160  (10%)
  hallucinated_parameter         80  (5%)
  malformed_syntax               80  (5%)

all structural checks passed (1600/1600 pairs well-formed).
NOTE: this is the fixture structural self-test, not the verifier gate.
reproduction: OK — committed fixtures are byte-identical to a fresh run.
```

The same check runs under `pytest tests/test_fixture_pairs.py`, which also
enforces Ground Rules 1-2 (every row `synthetic: true`, honest provenance, no
fabricated verifier metadata) so a future edit cannot quietly launder these
pairs into evidence.

## 5. What is now citable

- "The 1,600-pair fixture set is deterministic and reproduces byte-for-byte from
  a committed generator at seed 20260804." — yes, this record.
- "1,600/1,600 pairs are structurally well-formed against their tool schemas." —
  yes, with "structurally well-formed" stated, not dropped.
- "Fixture self-test gate: 1,600/1,600 pass, 0 false positives, 0 misses." —
  **no.** That is a verifier result and requires a reviewed
  `mining/mine_pairs.py` run against these fixtures, committed as its own
  artifact.
