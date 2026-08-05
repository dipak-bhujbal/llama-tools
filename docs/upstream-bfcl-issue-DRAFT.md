# Upstream issue — BFCL v4 `simple_python_363` answer key

**STATUS: FILED 2026-08-05 as
[ShishirPatil/gorilla#1354](https://github.com/ShishirPatil/gorilla/issues/1354).**

Authorized by the owner in #general msg 2063 ("still file upstream, and the
filing just got stronger"), released by review sign-off of `5d0e32b` in msg
2070, filed under the owner's GitHub account. Searched
`simple_python_363`, `find_closest`, `restaurant_search`, `possible_answer
simple_python` across open and closed issues and PRs first: no duplicate.

Filed against upstream's `bfcl.md` issue template rather than as the free-form
text below, so it carries the datapoint permalink, commit id, and
previous/updated datapoint block their triage expects. The `bfcl` label could
not be applied — it is not assignable by an outside contributor — so the title
carries the `[BFCL]` prefix their template specifies.

One claim was added after this draft was written, having been verified against
the upstream API rather than inferred: `9d8416a9` is **32 commits ahead** of the
release commit (2025-12-12 vs 2025-08-25) and its message reads "Add model
Nanbeige and Fix some data bugs" (#1257). The draft said the fix "looks like" it
already landed; upstream's own history says so.

The text below is the pre-filing draft, kept as the record of what was prepared
and reviewed.

The argument got stronger after the scorer check. The original framing was
"qualified naming is arguably correct" — a matter of taste. The evidence says
something sharper: at the release commit, the item grades against a function
name the benchmark never offers the model, and upstream has already fixed it in
a later revision. That is a reproducible defect report, not a preference.

---

**Title:** `simple_python_363`: release-commit answer key expects `find_closest`, but the item only offers `restaurant_search.find_closest`

**Body:**

At release commit `58f57e9124ea981403792dd51e00a6577e621fae`, the answer key
for `BFCL_v4_simple_python.json` item `simple_python_363` expects the function
name `find_closest`.

For reference, the file this report is against:
`berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json`
at that commit — sha256
`38b1bc7469d1de73a812ffce9e2b10a1d8812425fd090ed314066ccec76d0ceb`, Git blob
`455e9c4df114782ca187ded9665819e7ef912846`, 400 rows.

The question file for that same item presents exactly one tool, named
`restaurant_search.find_closest`:

```
BFCL_v4_simple_python.json  simple_python_363
  function: [ { "name": "restaurant_search.find_closest", ... } ]

possible_answer/BFCL_v4_simple_python.json  simple_python_363
  ground_truth: [ { "find_closest": { ... } } ]      # release commit
```

Since BFCL matches function names by exact string comparison, a model that
calls the tool by the only name it was given is scored incorrect. There is no
prediction that satisfies both the prompt and the key.

This appears to be an isolated inconsistency rather than a convention:

- Across all 400 `simple_python` items at the later revision
  `9d8416a96d1d69975493f1b6d60ff07d12a1726a`, the key name is among the names
  presented in that item's tool list — 0 exceptions. The same holds for all 200
  `multiple` items and all 258 `live_simple` items (858 rows, 0 exceptions).
- Module-qualified names are common, not exceptional: 42% of `simple_python`
  keys, 62% of `multiple` keys, and 30% of `live_simple` keys use them.
- At revision `9d8416a9…`, `simple_python_363` expects
  `restaurant_search.find_closest`, matching the presented tool. It looks like
  this was already corrected there; this issue is to confirm that reading and
  flag that the release commit still carries the earlier value.

Stripping the module prefix before comparison would not be a safe general fix:
29 of the 200 `multiple` items offer two or more tools that differ *only* by
module prefix (e.g. `triangle_properties.get` and `circle_properties.get`), so
tail-matching would make those items unscoreable.

No patch is being proposed: the fix already exists at `9d8416a9…`, and the
release commit is immutable in any case. The ask is narrower — confirm that
`9d8416a9…` is the intended value, and if so consider noting it somewhere
release-pinned consumers will see, since anyone who froze the benchmark at the
release commit is still scoring this item against a name it does not offer.

---

## Notes for the owner before filing

- The release-commit values in this draft are hash-verified, not restated. That
  file is pinned in `eval/manifests/bfcl_v4_study2.json` under role
  `answer_key_release_commit` and re-verified on every run of the comparison;
  its sha256 and Git blob id are quoted in the body above and recorded in
  `eval/results/answer_key_comparison.json`.
- The comparison also measures the internal-consistency claim directly: the
  release key fails the answer-key preflight at `simple_python_363`, and the
  data-fix key passes on all 400 rows.
- That measurement is what selected our canonical key, under a rule fixed in
  advance of the outcome — *canonical = pinned AND valid; when two pinned keys
  disagree, the preflight decides*. Both candidates are hash-pinned, so the
  choice turned on validity alone; had the preflight failed on `9d8416a9…`
  instead, the same rule would have selected the release key and our headline
  would be lower. The rule, both keys' hashes, and the measured outcomes are
  recorded in `eval/results/answer_key_comparison.json`, which fails to build if
  the recorded choice is not the one the rule produces.
- The 858-row and 29-item figures are reproducible from pinned data:
  `python eval/fetch_pinned_bfcl.py && python eval/answer_key_comparison.py`.
  The first command now also runs the answer-name preflight over every pinned
  category, so the "0 exceptions in 858 rows" claim above is re-checked on every
  verification rather than measured once.
- Filing this is what converts the disputed item from a private annoyance into
  a visible contribution, and gives the point somewhere to come back from if
  upstream amends the key.
