#!/usr/bin/env python3
"""Invariant test for the counter state machine in trainer_token_accounting.patch.

The patch splits token accounting across three sites in _train_step (buffer,
drop-branch, apply-branch) plus two gate predicates. That is a state machine, and
a diff review cannot show that it is correct. This replicates it exactly and
checks the invariants that the Phase 1 pre-registration depends on:

  I1  applied + dropped + pending == total, always
  I2  applied == total when the spike guard is off (loss_spike_ratio 0, PROTOCOL 1.3)
  I3  applied only ever advances at accumulation boundaries
  I4  the token gate stops within one effective batch of the budget
  I5  the eval grid is hit at the same token multiples regardless of batch sizes
      (this is what makes two arms comparable; PROTOCOL 2.4)
  I6  max_steps CANNOT pre-empt the token budget  <-- regression test

I6 exists because the first version of this file constructed every Sim with
max_steps=10**9, so the step gate could never bind and the test was structurally
blind to it. An audit found that the patch consulted max_steps unconditionally:
since max_steps counts MICRO-batches, two arms with accumulate_steps 12 and 17
and the same max_steps get token budgets differing by 17/12 = 1.42x -- the exact
incommensurability the token gate exists to remove. A test that cannot fail on
the bug it is meant to guard is worse than no test, because it reads as proof.

Run: python phase0/test_token_counter.py
"""
from __future__ import annotations

import sys


class Sim:
    """Exactly the patched state machine, nothing else."""

    def __init__(self, accumulate_steps, max_steps, max_target_tokens=0,
                 eval_every_tokens=0, backstop=0):
        self.global_step = 0
        self.accumulate_steps = accumulate_steps
        self.max_steps = max_steps
        self.max_target_tokens = max_target_tokens
        self.backstop = backstop
        self.stopped_on_backstop = False
        self.eval_every_tokens = eval_every_tokens
        self._next_eval_tokens = eval_every_tokens
        self.total_train_tokens = 0
        self.applied_target_tokens = 0
        self._pending_tokens = 0
        self._dropped_tokens = 0
        self._current_batch_spike = False
        self.evals_at = []
        self.applied_advanced_at = []

    def _should_continue(self):
        # Mirrors the patched trainer exactly: in a token-keyed run max_steps is
        # NOT consulted, because it counts micro-batches and is therefore
        # arm-dependent. Only an explicit micro-step backstop can pre-empt.
        if self.max_target_tokens:
            if self.applied_target_tokens >= self.max_target_tokens:
                return False
            if self.backstop and self.global_step >= self.backstop:
                self.stopped_on_backstop = True
                return False
            return True
        return self.global_step < self.max_steps

    def _eval_due(self):
        if self.eval_every_tokens:
            return self.applied_target_tokens >= self._next_eval_tokens
        return self.global_step >= self.max_steps + 1  # step path not under test

    def train_step(self, n_tokens, micro_spike=False):
        self.global_step += 1
        self.total_train_tokens += n_tokens
        self._pending_tokens += n_tokens
        if micro_spike:
            self._current_batch_spike = True
        if self.global_step % self.accumulate_steps == 0:
            if self._current_batch_spike:
                self._dropped_tokens += self._pending_tokens
            else:
                before = self.applied_target_tokens
                self.applied_target_tokens += self._pending_tokens
                if self.applied_target_tokens != before:
                    self.applied_advanced_at.append(self.global_step)
            self._pending_tokens = 0
            self._current_batch_spike = False

    def after_eval(self):
        while self._next_eval_tokens <= self.applied_target_tokens:
            self._next_eval_tokens += self.eval_every_tokens


def run(accum, tok_per_micro, n_micro, spike_every=0, max_target=0, eval_every=0,
        max_steps=10 ** 9, backstop=0):
    s = Sim(accum, max_steps=max_steps, max_target_tokens=max_target,
            eval_every_tokens=eval_every, backstop=backstop)
    i = 0
    while s._should_continue() and i < n_micro:
        spike = bool(spike_every) and (i % spike_every == 0)
        s.train_step(tok_per_micro, spike)
        if s.eval_every_tokens and s._eval_due():
            s.evals_at.append(s.applied_target_tokens)
            s.after_eval()
        i += 1
    return s


def check(name, cond, detail=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    return cond


def main() -> int:
    ok = True
    print("I1/I2 — conservation, and applied == total with the guard off")
    s = run(accum=4, tok_per_micro=2500, n_micro=1000)
    ok &= check("conservation", s.applied_target_tokens + s._dropped_tokens
                + s._pending_tokens == s.total_train_tokens,
                f"{s.applied_target_tokens:,}+{s._dropped_tokens:,}+{s._pending_tokens:,}"
                f" == {s.total_train_tokens:,}")
    ok &= check("applied == total (guard off, 1000 micro / accum 4)",
                s.applied_target_tokens == s.total_train_tokens,
                f"{s.applied_target_tokens:,}")

    print("\nI1 — conservation still holds when the guard drops batches")
    s = run(accum=4, tok_per_micro=2500, n_micro=1000, spike_every=37)
    ok &= check("conservation with spikes",
                s.applied_target_tokens + s._dropped_tokens + s._pending_tokens
                == s.total_train_tokens)
    ok &= check("applied < total when spikes occur",
                s.applied_target_tokens < s.total_train_tokens,
                f"applied {s.applied_target_tokens:,} dropped {s._dropped_tokens:,}")

    print("\nI3 — applied advances ONLY at accumulation boundaries")
    s = run(accum=4, tok_per_micro=2500, n_micro=200)
    ok &= check("every advance is at a multiple of accumulate_steps",
                all(g % 4 == 0 for g in s.applied_advanced_at),
                f"{len(s.applied_advanced_at)} advances")

    print("\nI4 — the token gate stops within ONE effective batch of the budget")
    budget = 10_000_000
    for accum, tpm in ((4, 2500), (8, 1200), (1, 9000)):
        s = run(accum=accum, tok_per_micro=tpm, n_micro=10 ** 6, max_target=budget)
        eff = accum * tpm
        over = s.applied_target_tokens - budget
        ok &= check(f"accum={accum} tok/micro={tpm}", 0 <= over < eff,
                    f"stopped at {s.applied_target_tokens:,} (+{over:,}, eff batch {eff:,})")

    print("\nI5 — the eval grid lands on the same multiples across DIFFERENT arms")
    grid = 5_000_000
    arms = {}
    for label, (accum, tpm) in {"Base-like": (4, 2500), "Big-like": (12, 830)}.items():
        s = run(accum=accum, tok_per_micro=tpm, n_micro=10 ** 6,
                max_target=50_000_000, eval_every=grid)
        arms[label] = [e // grid for e in s.evals_at]
        ok &= check(f"{label}: {len(s.evals_at)} evals, all past their multiple",
                    all(e >= (i + 1) * grid for i, e in enumerate(s.evals_at)))
    ok &= check("both arms evaluate at the SAME grid indices",
                arms["Base-like"] == arms["Big-like"],
                f"{arms['Base-like'][:6]}...")

    print("\nI5b — a step-keyed grid would NOT be comparable (the thing this fixes)")
    a = run(accum=4, tok_per_micro=2500, n_micro=4000)
    b = run(accum=12, tok_per_micro=830, n_micro=4000)
    ok &= check("same micro-step count gives DIFFERENT token counts across arms",
                a.applied_target_tokens != b.applied_target_tokens,
                f"{a.applied_target_tokens:,} vs {b.applied_target_tokens:,}")

    print("\nI6 — REGRESSION: max_steps must not be able to pre-empt the token budget")
    # PROTOCOL 3.2 calibration: both arms at 98,304 applied tokens/optimizer step,
    # Base accumulate_steps 12, Big 17. An inherited max_steps of 800,000
    # micro-batches (the largest in any existing config) would stop Base at 6.55B
    # and Big at 4.63B applied tokens -- a 29% deficit for Big, invisible.
    BUDGET = 20_000_000_000
    arms = {}
    for name, accum in (("Base", 12), ("Big", 17)):
        tpm = 98_304 // accum
        s = run(accum=accum, tok_per_micro=tpm, n_micro=10 ** 7,
                max_target=BUDGET, max_steps=800_000)
        arms[name] = s.applied_target_tokens
        ok &= check(f"{name} (accum {accum}) reaches its token budget despite "
                    "max_steps=800,000",
                    s.applied_target_tokens >= BUDGET,
                    f"{s.applied_target_tokens/1e9:.2f}B / {BUDGET/1e9:.0f}B")
    ok &= check("both arms get the SAME budget",
                abs(arms["Base"] - arms["Big"]) / BUDGET < 0.01,
                f"Base {arms['Base']/1e9:.2f}B vs Big {arms['Big']/1e9:.2f}B")

    print("\nI6b — the OLD behaviour, shown to be arm-dependent (why the fix matters)")
    old = {}
    for name, accum in (("Base", 12), ("Big", 17)):
        tpm = 98_304 // accum
        s = Sim(accum, max_steps=800_000, max_target_tokens=0)
        while s.global_step < s.max_steps:
            s.train_step(tpm)
        old[name] = s.applied_target_tokens
    ratio = old["Base"] / old["Big"]
    ok &= check("equal max_steps gives the two arms DIFFERENT token budgets",
                abs(ratio - 17 / 12) < 0.01,
                f"Base {old['Base']/1e9:.2f}B / Big {old['Big']/1e9:.2f}B = {ratio:.3f}x "
                f"(= 17/12 = {17/12:.3f})")

    print("\nI7 — an explicit backstop fires and is flagged, not silent")
    s = run(accum=12, tok_per_micro=8192, n_micro=10 ** 7,
            max_target=20_000_000_000, backstop=100_000)
    ok &= check("backstop stopped the run", s.stopped_on_backstop)
    ok &= check("and the token budget was NOT met",
                s.applied_target_tokens < 20_000_000_000,
                f"{s.applied_target_tokens/1e9:.2f}B of 20B")

    print("\n" + ("ALL INVARIANTS HOLD" if ok else "SOME INVARIANTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
