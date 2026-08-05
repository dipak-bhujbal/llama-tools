# Upstream issue draft — BFCL v4 `simple_python_363` answer key

**STATUS: DRAFT for the owner to file. Not filed by any agent.**

Ground Rule 7 puts anything public in human hands. This is text to paste, not
an action taken. Target: `ShishirPatil/gorilla`, Issues.

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

Happy to send a PR against the release-commit key if that is useful.

---

## Notes for the owner before filing

- Verify the release-commit values yourself before posting. This repo does not
  commit the release-commit file, so the `find_closest` value is reproduced
  from an independent check rather than hash-verified here — see the
  `provenance` note in `eval/results/answer_key_comparison.json`.
- The 858-row and 29-item figures are reproducible from committed data:
  `python eval/answer_key_comparison.py`.
- Filing this is what converts the disputed item from a private annoyance into
  a visible contribution, and gives the point somewhere to come back from if
  upstream amends the key.
