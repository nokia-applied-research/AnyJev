"""End-to-end through the real prompt path with a synthetic biased model."""
import numpy as np
import pytest

from anyjev import Decider, LevelError, Question
from anyjev.backends.fake import FakeBackend

OPTIONS = ["billing", "technical", "sales", "other"]
TRUTH = {"card declined": "billing", "app crashes": "technical", "bulk discount": "sales"}


def content(state, option):
    return 3.0 if TRUTH.get(state) == option else 0.0


def test_raw_is_fooled_by_position_bias_l0_is_not():
    be = FakeBackend(content, position_bias=[4.0, 0, 0, 0])   # loves position A
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be)
    raw = d.decide("app crashes", [q], level="raw")["route"]
    l0 = d.decide("app crashes", [q], level="L0")["route"]
    assert raw.level == "raw" and raw.argmax == "billing"       # wrong, position A
    assert l0.level == "L0" and l0.argmax == "technical"
    assert l0.diagnostics["order_flip_raw"] > 0
    assert l0.diagnostics["prior_method"] == "none"             # one item: batch prior not available yet
    assert l0.diagnostics["order_flip_l0"] == l0.diagnostics["order_flip_raw"]


def test_content_free_prior_removes_label_prior_exactly():
    be = FakeBackend(content, label_prior={"Yes": 2.0})
    q = Question.noul("Is this about billing?", name="bill")
    d = Decider(be, prior="content_free", adaptive_shifts=False)
    raw = d.decide("nothing", [q], level="raw")["bill"]
    l0 = d.decide("nothing", [q], level="L0")["bill"]
    assert raw.p_true > 0.85
    assert abs(l0.p_true - 0.5) < 1e-6
    assert l0.diagnostics["prior_method"] == "content_free"


def test_batch_prior_removes_label_prior_on_a_balanced_batch():
    be = FakeBackend(lambda s, o: 1.5 if (o == "Yes") == s.startswith("yes") else 0.0,
                     label_prior={"Yes": 2.0})
    q = Question.noul("Is it a yes?", name="y")
    states = [f"yes {i}" for i in range(10)] + [f"no {i}" for i in range(10)]
    d = Decider(be, prior="batch", min_prior_n=8)
    raw = d.decide_batch(states, q, level="raw")
    l0 = d.decide_batch(states, q, level="L0")
    raw_acc = np.mean([(r.p_true >= 0.5) == s.startswith("yes") for r, s in zip(raw, states)])
    l0_acc = np.mean([(r.p_true >= 0.5) == s.startswith("yes") for r, s in zip(l0, states)])
    assert raw_acc == 0.5 and l0_acc == 1.0          # prior swamps content raw; batch prior removes it
    assert l0[0].diagnostics["prior_method"] == "batch"
    # single-item call afterwards uses the running prior
    one = d.decide("no 99", [q])["y"]
    assert one.diagnostics["prior_method"] == "batch" and one.p_true < 0.5


def test_score_expected_value():
    be = FakeBackend(lambda s, o: 5.0 if o.startswith("0.75") else 0.0)
    q = Question.score("How complete?", bins=4, name="done")
    r = Decider(be).decide("x", [q])["done"]
    assert 0.8 < r.value < 0.9
    assert r.level == "L0"


def test_probes_are_shared_across_states():
    be = FakeBackend(content, position_bias=[1, 0, 0, 0])
    q = Question.choice("Which handler?", OPTIONS)
    d = Decider(be, prior="content_free", adaptive_shifts=False)
    d.decide_batch(list(TRUTH), q)
    # 3 states x 4 perms real + 4 perms x 3 probes shared = 24, not 3 x (4 + 12) = 48;
    # two forward calls: the real prompts and the probes never share a batch
    assert be.prompts_seen == 24 and be.calls == 2
    d.decide("card declined", [q])
    assert be.prompts_seen == 28                        # cf prior cached: 4 new prompts, no probes
    d2 = Decider(FakeBackend(content), adaptive_shifts=False)
    d2.decide_batch(list(TRUTH), q)
    assert d2.backend.prompts_seen == 12                # default batch prior: no probes at all


def test_l1_requires_artifact_and_reports_level():
    be = FakeBackend(content, temperature=0.3)   # overconfident
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be)
    with pytest.raises(ValueError):
        d.decide("card declined", [q], level="L1")
    states = list(TRUTH) * 20
    labels = [OPTIONS.index(TRUTH[s]) for s in states]
    art = d.calibrate(q, states, labels)
    assert art["model"] == "fake" and art["temperature"] > 0
    r = d.decide("card declined", [q], level="L1")["route"]
    assert r.level == "L1" and r.argmax == "billing"
    d2 = Decider(FakeBackend(content), adaptive_shifts=False)
    d2.load_artifact(q, art)
    with pytest.raises(ValueError):
        d2.load_artifact(q, {**art, "model": "some-other-model"})


def test_max_permutations_cap():
    be = FakeBackend(content)
    q = Question.choice("Which handler?", OPTIONS)
    r = Decider(be, max_permutations=2).decide("x", [q])[0]
    assert r.diagnostics["permutations"] == 2


def test_ablation_data_present():
    be = FakeBackend(content, position_bias=[4.0, 0, 0, 0])
    q = Question.choice("Which handler?", OPTIONS, name="route")
    r = Decider(be, record_content_free=True).decide("app crashes", [q])["route"]
    assert r.diagnostics["p_pos_raw"].shape == (4, 4)
    assert r.diagnostics["cf_prior"].shape == (4, 4)
    assert r.diagnostics["batch_prior"] is None         # one item: below min_prior_n
    assert len(r.diagnostics["perms"]) == 4


def test_noul_phrasing_swap_keeps_yes_attached_to_yes():
    # no biases at all: "Yes or No" and "No or Yes" must give the same distribution
    be = FakeBackend(lambda s, o: 2.0 if o == "Yes" else 0.0)
    q = Question.noul("Is it?", name="it")
    r = Decider(be).decide("x", [q])["it"]
    P = r.diagnostics["p_pos_raw"]            # [2 phrasings, 2 positions]
    perms = r.diagnostics["perms"]
    a = {perms[0][j]: P[0, j] for j in range(2)}
    b = {perms[1][j]: P[1, j] for j in range(2)}
    assert abs(a[0] - b[0]) < 1e-9 and a[0] > 0.8   # option 0 = Yes in both
    assert r.diagnostics["order_flip_raw"] == 0.0
    assert r.p_true > 0.8


def test_artifacts_round_trip(tmp_path):
    be = FakeBackend(content, temperature=0.3)
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be, prior="none")            # same prior on both sides, so only the artifact differs
    states = list(TRUTH) * 20
    d.calibrate(q, states, [OPTIONS.index(TRUTH[s]) for s in states])
    path = tmp_path / "art.json"
    d.save_artifacts(str(path))
    d2 = Decider(FakeBackend(content, temperature=0.3), prior="none")
    assert d2.load_artifacts(str(path)) == 1
    a = d.decide("card declined", [q], level="L1")["route"]
    b = d2.decide("card declined", [q], level="L1")["route"]
    np.testing.assert_allclose(a.probs, b.probs, atol=1e-12)
    assert a.diagnostics["temperature"] == b.diagnostics["temperature"]
    with pytest.raises(ValueError):
        Decider(FakeBackend(content)).load_artifacts({"model": "other", "artifacts": {}})


def test_prior_strength_interpolates_between_none_and_full():
    def be():
        return FakeBackend(lambda s, o: 1.5 if (o == "Yes") == s.startswith("yes") else 0.0, label_prior={"Yes": 2.0})

    q = Question.noul("Is it a yes?", name="y")
    states = [f"yes {i}" for i in range(10)] + [f"no {i}" for i in range(10)]
    p_none = Decider(be(), prior="none").decide_batch(states, q)[-1].p_true
    p_half = Decider(be(), prior="batch", prior_strength=0.5).decide_batch(states, q)[-1].p_true
    p_full = Decider(be(), prior="batch", prior_strength=1.0).decide_batch(states, q)[-1].p_true
    d = Decider(be())                                    # default strength for the batch prior
    p_def = d.decide_batch(states, q)[-1].p_true
    assert d.strength() == 0.75 and Decider(be(), prior="content_free").strength() == 1.0
    assert p_full < p_def < p_half < p_none               # a "no" item: more correction, less Yes
    assert d.decide_batch(states, q)[0].diagnostics["prior_strength"] == 0.75
    with pytest.raises(ValueError):
        Decider(be(), prior_strength=1.5)


def test_l1_artifact_is_a_pure_function_of_the_calibration_set():
    """The repro check found L1 drifting with the decider's history (a running batch prior).
    The artifact now freezes the prior it was fit with."""
    q = Question.choice("Which handler?", OPTIONS, name="route")
    calib = list(TRUTH) * 20
    labels = [OPTIONS.index(TRUTH[s]) for s in calib]
    fresh = Decider(FakeBackend(content, temperature=0.3, position_bias=[1.0, 0, 0, 0]),
                    adaptive_shifts=False)
    busy = Decider(FakeBackend(content, temperature=0.3, position_bias=[1.0, 0, 0, 0]),
                   adaptive_shifts=False)
    busy.decide_batch(["unrelated ticket"] * 40 + ["app crashes"] * 25, q)      # history before calibrate
    a = fresh.calibrate(q, calib, labels)
    b = busy.calibrate(q, calib, labels)
    assert a["temperature"] == b["temperature"] and a["prior"] == b["prior"] and a["n_calib"] == 60
    assert a["prior_method"] == "batch" and a["prior_strength"] == 0.75
    # and L1 decisions no longer depend on what else the decider has seen
    busy.decide_batch(["bulk discount"] * 30, q)
    x = fresh.decide("card declined", [q], level="L1")["route"]
    y = busy.decide("card declined", [q], level="L1")["route"]
    np.testing.assert_allclose(x.probs, y.probs, atol=1e-12)
    assert x.diagnostics["prior_method"] == "frozen:batch"
    # a third decider that only loads the artifact agrees too
    other = Decider(FakeBackend(content, temperature=0.3, position_bias=[1.0, 0, 0, 0]))
    other.load_artifact(q, a)
    z = other.decide("card declined", [q], level="L1")["route"]
    np.testing.assert_allclose(x.probs, z.probs, atol=1e-12)


def test_old_artifacts_without_a_prior_still_load():
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content))
    d.load_artifact(q, {"model": "fake", "question": q.key, "method": "temperature", "temperature": 2.0})
    r = d.decide("card declined", [q], level="L1")["route"]
    assert r.level == "L1" and r.diagnostics["temperature"] == 2.0


def test_content_free_probes_never_share_a_forward_call_with_real_states():
    """The batch composition of the real prompts (and so their bf16 logits on a GPU) must not
    depend on whether the content-free probes are cached yet, nor on `record_content_free`."""
    q = Question.choice("Which handler?", OPTIONS)
    states = list(TRUTH) + ["card declined", "reset my password"]

    def spy(dec):
        calls = []
        orig = dec._score

        def _score(prompts, prompt_ids, prompt_parts):
            calls.append(list(prompts))
            return orig(prompts, prompt_ids, prompt_parts)

        dec._score = _score
        return calls

    plain = Decider(FakeBackend(content), prior="batch", adaptive_shifts=False)
    diag = Decider(FakeBackend(content), prior="batch", record_content_free=True,
                   adaptive_shifts=False)
    calls_plain, calls_diag = spy(plain), spy(diag)
    a = plain.decide_batch(states, q, level="L0")
    b = diag.decide_batch(states, q, level="L0")
    # the real-state call is identical prompt for prompt; the probes are a separate call
    assert calls_plain[0] == calls_diag[0]
    assert len(calls_plain) == 1 and len(calls_diag) == 2
    assert not set(calls_diag[1]) & set(calls_diag[0])
    assert all(np.allclose(x.probs, y.probs) for x, y in zip(a, b))
    # a second call on the used decider (probes cached) scores exactly the same real prompts
    diag.decide_batch(states, q, level="L0")
    assert calls_diag[2] == calls_diag[0]


def test_an_artifact_with_a_prior_refuses_another_question_layout():
    """The frozen prior is indexed by (permutation, position), so it belongs to the option order
    it was fit on. A 0.0.2 artifact (temperature only) still loads anywhere."""
    from anyjev.question import Question as Q
    be = FakeBackend(content, temperature=0.3)
    q = Question.choice("Which handler?", OPTIONS)
    qr = Q(q.kind, q.text, tuple(reversed(q.options)), q.name, q.scale, q.ordered)
    d = Decider(be)
    states = list(TRUTH) * 20
    art = d.calibrate(q, states, [OPTIONS.index(TRUTH[s]) for s in states])
    assert art["prior"] is not None
    with pytest.raises(ValueError):
        d.load_artifact(qr, art)
    d.load_artifact(q, art)                                        # same layout: fine
    old = {k: v for k, v in art.items() if k not in ("prior", "prior_method", "prior_strength", "n_calib")}
    d.load_artifact(qr, old)                                       # temperature-only: fine
    assert d.decide("card declined", [qr], level="L1")[0].level == "L1"


# ---- L2: closed-form per-question head on hidden states ----------------------------------
def _labelled_routing(n: int = 60):
    states = [f"ticket {i}" for i in range(n)]
    labels = [i % 4 for i in range(n)]
    truth = {s: OPTIONS[y] for s, y in zip(states, labels)}
    return states, labels, truth


def test_l2_head_is_fit_in_closed_form_and_read_from_a_truncated_forward():
    states, labels, truth = _labelled_routing()
    be = FakeBackend(lambda s, o: 3.0 if truth.get(s) == o else 0.0, position_bias=[4.0, 0, 0, 0])
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be)
    art = d.fit_head(q, states[:48], labels[:48])
    assert art["method"] in ("head:lda", "head:ridge")
    assert art["layer_abs"] in (2, 3, 4)                    # candidates: 50..100% of the fake's 4 blocks
    assert art["n_calib"] == 48 and art["W"]["shape"] == [be.hidden_size, 4]   # compact float32 artifact
    decs = d.decide_batch(states[48:], q, level="L2")
    assert all(x.level == "L2" for x in decs)
    acc = np.mean([x.argmax == truth[s] for x, s in zip(decs, states[48:])])
    assert acc >= 0.9                                        # the planted best-option code is separable
    raw = d.decide_batch(states[48:], q, level="raw")
    assert np.mean([x.argmax == truth[s] for x, s in zip(raw, states[48:])]) < acc   # raw is fooled by position A
    diag = decs[0].diagnostics
    assert diag["readout"] == art["method"] and diag["blocks_executed"] == art["layer_abs"]
    assert diag["n_blocks"] == be.n_layers and diag["permutations"] == 1
    assert decs[0].require("L2") is decs[0]
    with pytest.raises(LevelError):
        d.decide(states[48], [q], level="L0")["route"].require("L2")


def test_l2_artifacts_round_trip_and_refuse_other_layouts():
    states, labels, truth = _labelled_routing()
    content = lambda s, o: 3.0 if truth.get(s) == o else 0.0   # noqa: E731
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content))
    art = d.fit_head(q, states[:48], labels[:48], layers=[4])
    assert art["layer_abs"] == 4
    saved = d.export_artifacts()
    assert list(saved["heads"]) == [q.key] and saved["artifacts"] == {}
    d2 = Decider(FakeBackend(content), adaptive_shifts=False)
    assert d2.load_artifacts(saved) == 1
    p1 = np.stack([x.probs for x in d.decide_batch(states[48:], q, level="L2")])
    p2 = np.stack([x.probs for x in d2.decide_batch(states[48:], q, level="L2")])
    assert np.allclose(p1, p2)
    d3 = Decider(FakeBackend(content))
    d3.load_artifact(q, art)
    assert np.allclose(np.stack([x.probs for x in d3.decide_batch(states[48:], q, level="L2")]), p1)
    other = Question.choice("Which handler?", OPTIONS[::-1], name="route")
    with pytest.raises(ValueError):
        d3.load_artifact(other, art)                         # an artifact is bound to its exact layout
    unrelated = Question.choice("Which handler?", OPTIONS[:3] + ["legal"], name="route")
    assert d3.route(unrelated) is None
    with pytest.raises(ValueError):
        d3.decide(states[0], [unrelated], level="L2")        # no head for this option set
    with pytest.raises(ValueError):
        d3.fit_head(q, states[:5], labels[:5])              # fewer than max(8, 2K) labels


# ---- L2 routing, test-time adaptation, listing orders, auto level -------------------------
def test_recentring_undoes_a_shift_and_rescale_of_the_features_exactly():
    import dataclasses

    from anyjev.heads import _standardise, fit_head
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, 90)
    X = rng.randn(90, 16) + np.eye(3)[y] @ (2.0 * rng.randn(3, 16))
    head = fit_head(X, y, 3, kind="lda")
    Xs = X * 1.7 + rng.randn(16) * 3.0                     # a new "wording": every feature shifted and rescaled
    assert np.mean(head.probs(Xs).argmax(1) == head.probs(X).argmax(1)) < 1.0
    m, s = _standardise(Xs)
    adapted = dataclasses.replace(head, mean=m, scale=s)
    assert np.allclose(adapted.probs(Xs), head.probs(X), atol=1e-6)


def test_l2_routes_a_reworded_question_and_adapts_without_labels():
    states, labels, truth = _labelled_routing()
    content = lambda s, o: 3.0 if truth.get(s) == o else 0.0   # noqa: E731
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content), adapt_min_n=8)           # adapt="routed" by default
    d.fit_head(q, states[:48], labels[:48], layers=[4])
    assert not d.decide_batch(states[48:], q, level="L2")[0].diagnostics["adapted"]   # own layout: untouched
    reworded = Question.choice("Who should take this ticket?", OPTIONS, name="route2")
    assert d.route(reworded) is not None and d.route(reworded)[1] is None
    decs = d.decide_batch(states[48:], reworded, level="L2")
    assert np.mean([x.argmax == truth[s] for x, s in zip(decs, states[48:])]) >= 0.9
    diag = decs[0].diagnostics
    assert diag["routed_from"] == "route" and diag["adapted"] and diag["adapt_n"] == 12
    assert decs[0].question is reworded
    cold = Decider(FakeBackend(content), adapt_min_n=100)
    cold.load_artifacts(d.export_artifacts())
    assert not cold.decide_batch(states[48:], reworded, level="L2")[0].diagnostics["adapted"]


def test_l2_head_fit_on_random_listings_serves_the_options_in_another_order():
    states, labels, truth = _labelled_routing()
    content = lambda s, o: 3.0 if truth.get(s) == o else 0.0   # noqa: E731
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content), adapt=False)
    art = d.fit_head(q, states[:48], labels[:48], layers=[4])          # listing="auto": random for K=4
    assert art["params"]["listing"] == "random"
    wide = Question.choice("Which of twelve?", [f"opt{i}" for i in range(12)], name="wide")
    wide_truth = {s: wide.options[i % 12] for i, s in enumerate(states)}
    d_wide = Decider(FakeBackend(lambda s, o: 3.0 if wide_truth.get(s) == o else 0.0))
    art_wide = d_wide.fit_head(wide, states[:48], [i % 12 for i in range(48)], layers=[4])
    assert art_wide["params"]["listing"] == "canonical"                # auto: canonical above 8 options
    reordered = Question.choice("Which handler?", OPTIONS[::-1], name="route_rev")
    entry, index_map = d.route(reordered)
    assert index_map == [3, 2, 1, 0]
    decs = d.decide_batch(states[48:], reordered, level="L2")
    assert np.mean([x.argmax == truth[s] for x, s in zip(decs, states[48:])]) >= 0.9
    assert decs[0].diagnostics["reordered"] and decs[0].diagnostics["listing"] == "random"
    d_canon = Decider(FakeBackend(content))
    d_canon.fit_head(q, states[:48], labels[:48], layers=[4], listing="canonical")
    assert d_canon.route(reordered) is None                    # a one-order head is not offered to another order
    with pytest.raises(ValueError):
        d_canon.decide(states[0], [reordered], level="L2")


def test_auto_level_uses_the_best_available_level_per_question():
    states, labels, truth = _labelled_routing()
    content = lambda s, o: 3.0 if truth.get(s) == o else 0.0   # noqa: E731
    q = Question.choice("Which handler?", OPTIONS, name="route")
    other = Question.noul("Is this ticket about billing?", name="billing")
    d = Decider(FakeBackend(content))
    d.fit_head(q, states[:48], labels[:48], layers=[4])
    r = d.decide(states[50], [q, other], level="auto")
    assert r["route"].level == "L2" and r["billing"].level == "L0" and r.level == "L0"
    assert Decider(FakeBackend(content), level="auto").decide(states[50], [other])["billing"].level == "L0"


def test_a_wording_shift_of_the_hidden_state_costs_the_routed_head_and_recentring_recovers_it():
    """FakeBackend(wording_shift=a) moves the last-position state by a per-feature affine map that depends on the
    question line; the head fit on one wording, served as is under another, loses accuracy, and the label-free
    recentring (adapt="routed") brings it back to the exact head's."""
    states, labels, truth = _labelled_routing(120)
    content = lambda s, o: 3.0 if truth.get(s) == o else 0.0   # noqa: E731
    be = FakeBackend(content, wording_shift=3.0)
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(be)
    d.fit_head(q, states[:60], labels[:60], layers=[4])
    export = d.export_artifacts()
    exact = np.mean([x.argmax == truth[s] for x, s in zip(d.decide_batch(states[60:], q, level="L2"), states[60:])])
    reworded = Question.choice("Who should take this ticket?", OPTIONS, name="route2")
    as_is = Decider(FakeBackend(content, wording_shift=3.0), adapt=False)
    as_is.load_artifacts(export)
    acc_as_is = np.mean([x.argmax == truth[s]
                         for x, s in zip(as_is.decide_batch(states[60:], reworded, level="L2"), states[60:])])
    rec = Decider(FakeBackend(content, wording_shift=3.0), adapt="routed", adapt_min_n=30)
    rec.load_artifacts(export)
    decs = rec.decide_batch(states[60:], reworded, level="L2")
    acc_rec = np.mean([x.argmax == truth[s] for x, s in zip(decs, states[60:])])
    assert exact >= 0.9 and acc_as_is < acc_rec and acc_rec == exact
    assert decs[0].diagnostics["routed_from"] == "route" and decs[0].diagnostics["adapted"]
    # the knobs default to off: a backend without them gives the same prompts and vectors as before
    plain = FakeBackend(content)
    assert plain.logit_noise == 0.0 and plain.wording_shift == 0.0 and plain.layer_noise == 0.4


# ---- observe(): the head grows out of the traffic; adaptation statistics survive a restart ------
def test_observe_solves_the_head_when_enough_labels_arrive_and_refits_as_they_double():
    states, labels, truth = _labelled_routing(160)
    content = lambda s, o: 3.0 if truth.get(s) == o else 0.0   # noqa: E731
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content), level="auto")
    assert d.decide(states[150], [q])["route"].level == "L0"     # day 0: no head, auto answers at L0
    fits = []
    for i, (s, y) in enumerate(zip(states[:150], labels[:150]), 1):
        art = d.observe(q, s, y, fit_at=30, layers=[4])
        if art is not None:
            fits.append((i, art["n_calib"]))
    assert [i for i, _ in fits] == [30, 60, 120] and [n for _, n in fits] == [30, 60, 120]
    decs = d.decide_batch(states[150:], q)                         # level auto -> L2 now
    assert all(x.level == "L2" for x in decs)
    assert np.mean([x.argmax == truth[s] for x, s in zip(decs, states[150:])]) >= 0.9
    assert d.observations(q)[1] == labels[:150]
    with pytest.raises(ValueError):
        d.observe(q, states[0], 7)


def test_adaptation_statistics_and_observations_survive_export_and_load():
    states, labels, truth = _labelled_routing()
    content = lambda s, o: 3.0 if truth.get(s) == o else 0.0   # noqa: E731
    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content), adapt_min_n=8)
    d.fit_head(q, states[:48], labels[:48], layers=[4])
    reworded = Question.choice("Who should take this ticket?", OPTIONS, name="route2")
    before = d.decide_batch(states[48:], reworded, level="L2")
    assert before[0].diagnostics["adapted"] and before[0].diagnostics["adapt_n"] == 12
    for s, y in zip(states[:5], labels[:5]):
        d.observe(reworded, s, y)
    saved = d.export_artifacts(include_observations=True)
    assert set(saved["adaptation"]) == {reworded.key} and saved["observations"][reworded.key]["labels"] == labels[:5]
    d2 = Decider(FakeBackend(content), adapt_min_n=8)
    d2.load_artifacts(saved)
    after = d2.decide_batch(states[48:], reworded, level="L2")   # the restart starts adapted: 12 seen + 12 now
    assert after[0].diagnostics["adapted"] and after[0].diagnostics["adapt_n"] == 24
    assert np.allclose(np.stack([x.probs for x in after]), np.stack([x.probs for x in before]), atol=1e-6)
    assert d2.observations(reworded)[1] == labels[:5]


def test_an_unknown_calibrator_method_raises_instead_of_being_read_as_a_temperature():
    """Before 0.2.0 the dispatch fell through: an artifact with any method other than `head:` reached
    `TemperatureScaler.from_dict` and died on a missing key. A registry makes adding an L1 calibrator
    one line and makes a typo an error."""
    from anyjev.decider import CALIBRATORS

    q = Question.choice("Which handler?", OPTIONS, name="route")
    d = Decider(FakeBackend(content))
    with pytest.raises(ValueError, match="no loader"):
        d.load_artifact(q, {"method": "binning", "bins": [0.1, 0.9]})
    assert "temperature" in CALIBRATORS
    d.load_artifact(q, {"method": "temperature", "temperature": 2.0})   # and the known one still loads
    assert d.decide("card declined", [q], level="L1")["route"].level == "L1"


def test_reset_prior_forgets_the_history_of_one_question_or_all():
    """A long-lived Decider (a server) keeps the batch prior running across every call. Two
    runs through the same Decider then differ only by what it saw before; reset_prior() puts a
    question back where a fresh Decider starts, without dropping anything else it holds."""
    be = lambda: FakeBackend(lambda s, o: 1.5 if (o == "Yes") == s.startswith("yes") else 0.0,  # noqa: E731
                             label_prior={"Yes": 2.0})
    q = Question.noul("Is it a yes?", name="y")
    other = Question.noul("Is it urgent?", name="u")
    history = [f"yes {i}" for i in range(12)]
    d = Decider(be())
    d.decide_batch(history, q)
    d.decide_batch(history, other)
    assert d.running_prior(q) is not None and d.running_prior(other) is not None

    d.reset_prior(q)
    assert d.running_prior(q) is None and d.running_prior(other) is not None
    after = d.decide("no 1", [q])["y"]
    fresh = Decider(be()).decide("no 1", [q])["y"]
    assert after.diagnostics["prior_method"] == "none"
    np.testing.assert_allclose(after.probs, fresh.probs, atol=1e-12)

    d.reset_prior()
    assert d.running_prior(other) is None
