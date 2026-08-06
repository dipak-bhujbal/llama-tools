# Working Agreement — llama-tools

Operating rules for agent-assisted work in this repository. Owner: **Dipak Bhujbal** (human).
Agents execute; the owner signs off at every decision, spend, and publication gate.

Ratified 2026-08-04. Supersedes nothing; adds to the ground rules in `docs/HANDOFF.md`
when that file lands.

> **Why this file lives here.** The equivalent document for another project sits in a
> repository that is currently off limits. Rather than write outside the approved scope,
> the agreement is recorded here, where the work actually happens.

---

## 1. Scope

- Active scope is **`llama-tools` only**.
- `boundary-validity` and `boundary-crossing` are **off limits**: not read, not modified,
  not run, not referenced as working context.
- Work outside this repository — including files under `~/Downloads` such as the resume —
  is **flagged to the owner, never edited by an agent**.

## 2. Roles

| Role | Holder | Duty |
|---|---|---|
| Implementer | **Claude** | Writes code, docs, and manifests. Single writer. |
| Reviewer | **Codex** | Independently verifies claims and artifacts. Does not write. |
| Owner | **Dipak** | Approves every decision, spend, and publication gate. |

- **Role reversal:** after **3 review → fix cycles** without sign-off, roles reverse until
  sign-off is reached. The cycle counter resets on sign-off.
- **Single-writer discipline.** Only the current implementer writes. Handoff is explicit:
  the outgoing writer commits or stashes, states the commit, and confirms it has stopped.
  This exists because a real edit collision occurred when both agents patched one shared
  file concurrently.

## 3. Spend

- **No model API or paid compute spend without both halves:** the agents agree a written
  estimate *and* the owner explicitly approves it. Approval of one stage is not approval
  of the next.
- Every paid stage carries a **hard cap**. Exceeding the cap stops the run.
- Engineering, verification, and documentation are `$0` and need no spend approval.

## 4. Evidence and claims

- **Results are outputs, not targets.** Portfolio figures the owner supplies are
  **planning / go-no-go targets only**. They are never written into the README, ADRs,
  model cards, commit messages, HF artifacts, or the resume as measured results.
- **Only artifact-backed numbers may be reported** — every figure must trace to a run
  artifact (ledger, `trainer_state.json`, eval JSON) committed in this repo.
- **No fabricated provenance.** Every metadata field must describe something that actually
  happened. Synthetic fixtures are permitted only when labelled `"synthetic": true` with
  honest provenance.
- **A negative or null result is a publishable outcome**, not a failure to be hidden or
  reframed.

## 5. Data integrity

- **Eval sets are read-only and pinned.** Every scored file is fixed in
  `eval/manifests/` by **per-file** git blob SHA-1, content SHA-256, row count, unique-id
  count, and sorted-id digest. Commit SHAs are provenance annotation only — two files in
  one upstream repository have independent histories, and pinning both to one commit has
  already produced a silent, score-changing error once.
- **Decontamination removes items from training pools only**, never from eval sets.
- **Ledgers are append-only.** Never hand-edited.
- **Fixtures never train anything.** Trainer dry-runs use `learning_rate = 0.0` with a
  throwaway output directory, and prove it by asserting parameter-hash equality before and
  after. No dry-run artifact enters selection or evidence.

## 6. Pre-registration

- Any threshold, sweep value, or kill line must be committed to the pre-registration
  document **before** the corresponding run starts. If it is not in that file beforehand,
  it cannot be used to select results afterwards.
- **Checkpoint selection is by held-out task metric**, never by preference metrics and
  never by training loss.
- **Final held-out sets never select a checkpoint.** Selection uses a development set that
  is disjoint from every final scoring set.

## 7. Publication

- Pushing to GitHub or Hugging Face, and changing artifact visibility, are **owner actions**.
  Agents stage; the owner publishes.
- **`main` requires review before merge** (owner, #general msg 2117). No merge or direct push
  to `main` happens until codex has signed off on the exact commit being merged — owner
  authorization alone is not sufficient. This was added after a merge landed on owner
  authorization while a review was still in flight, so `main` briefly carried a version the
  reviewer had already asked to change. The rule costs one round trip; the alternative is a
  public default branch that is ahead of its own review.
- The reviewed-and-merged state of `main` is what a stranger reads. Work in progress belongs
  on the branch, clearly labelled, until it has cleared the same bar.

## 8. Intake

- Externally supplied files are **untrusted until inspected and hashed**. Claims about
  their contents are not carried forward until reproduced locally.

## 9. Coordination

- When a decision may be in flight, read the **backing message log** rather than a cursored
  feed. Cursor-based reads have silently skipped decision messages more than once, causing
  agents to re-request approvals already granted.
- Corrections are stated plainly and moved past. Being wrong in review is the mechanism
  working, not a failure.
