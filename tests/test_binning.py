import json

import numpy as np
import pytest

from anyjev.calibrate.binning import HistogramBinning
from bench.metrics import ece


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def test_bin_value_is_the_measured_accuracy():
    p = np.tile(softmax([np.log(18.0), 0.0, 0.0]), (100, 1))
    y = np.array([0] * 60 + [1] * 40)
    cal = HistogramBinning.fit(p, y)        
    assert len(cal.values) == 1
    out = cal.apply(p[:1])[0]
    np.testing.assert_allclose(out, [0.6, 0.2, 0.2], atol=1e-12)


def test_two_bins_each_return_their_own_accuracy():
    low = np.tile([0.6, 0.2, 0.2], (50, 1))    
    high = np.tile([0.95, 0.03, 0.02], (50, 1))   
    p = np.vstack([low, high])
    y = np.array([0] * 20 + [1] * 30 + [0] * 40 + [1] * 10)
    cal = HistogramBinning.fit(p, y, n_bins=2, min_count=20)
    np.testing.assert_allclose(cal.values, [0.4, 0.8], atol=1e-12)
    out = cal.apply(np.array([[0.6, 0.2, 0.2], [0.95, 0.03, 0.02]]))
    np.testing.assert_allclose(out[0], [0.4, 0.3, 0.3], atol=1e-12)
    np.testing.assert_allclose(out[1], [0.8, 0.12, 0.08], atol=1e-12)
    edge = cal.apply(np.array([[0.34, 0.33, 0.33], [0.999, 0.0005, 0.0005]]))
    np.testing.assert_allclose(edge[:, 0], [0.4, 0.8], atol=1e-12)


@pytest.mark.parametrize("k", [2, 4, 20])
def test_argmax_is_kept_and_rows_sum_to_one(k):
    rng = np.random.default_rng(k)
    p = rng.dirichlet(np.full(k, 0.5), size=2000)
    y = rng.integers(0, k, size=2000)            
    cal = HistogramBinning.fit(p, y)
    out = cal.apply(p)
    np.testing.assert_array_equal(out.argmax(1), p.argmax(1))
    np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-12)
    assert (out >= 0).all()


def test_runner_up_cannot_overtake_the_top_option():
    p = np.tile([0.50, 0.45, 0.05], (10, 1))
    y = np.array([0] * 3 + [1] * 7)
    cal = HistogramBinning.fit(p, y, min_count=5)
    np.testing.assert_allclose(cal.values, [0.3])
    out = cal.apply(p[:1])[0]
    assert out.argmax() == 0 and out[0] > out[1]
    np.testing.assert_allclose(out[0], 0.45 / 0.95, atol=1e-8)  
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)


def test_all_mass_on_one_option():
    cal = HistogramBinning(edges=np.zeros(0), values=np.array([0.7]), counts=np.array([30]))
    out = cal.apply(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(out, [0.7, 0.15, 0.15], atol=1e-12)


def test_unfitted_calibrator_cannot_be_built():
    with pytest.raises(TypeError):
        HistogramBinning()


def test_outputs_are_never_exactly_zero_or_one():
    p = np.tile([0.9, 0.08, 0.02], (20, 1))
    cal = HistogramBinning.fit(p, [0] * 20)
    np.testing.assert_allclose(cal.values, [1.0])
    out = cal.apply(np.vstack([p[:1], [[1.0, 0.0, 0.0]]]))
    assert (out > 0).all() and (out < 1).all()
    assert np.isfinite(np.log(out)).all()
    np.testing.assert_array_equal(out.argmax(1), [0, 0])
    np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-12)


def test_fewer_labels_than_min_count_give_one_bin():
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 0, 1, 0, 1])
    cal = HistogramBinning.fit(p, y, min_count=20)
    np.testing.assert_allclose(cal.values, [0.6])


def test_overconfident_model_gets_calibrated_on_held_out_data():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6000, 4)) * 2
    y = np.array([rng.choice(4, p=softmax(z)) for z in logits])
    overconfident = softmax(logits * 3.0)
    cal = HistogramBinning.fit(overconfident[:3000], y[:3000])
    out = cal.apply(overconfident[3000:])
    assert ece(out, y[3000:]) < ece(overconfident[3000:], y[3000:]) / 2
    np.testing.assert_array_equal(out.argmax(1), overconfident[3000:].argmax(1))


def test_round_trip_is_exact():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(5), size=500)
    y = rng.integers(0, 5, size=500)
    cal = HistogramBinning.fit(p, y)
    back = HistogramBinning.from_dict(json.loads(json.dumps(cal.to_dict())))
    np.testing.assert_array_equal(back.apply(p), cal.apply(p))


def test_bad_input_is_refused():
    p = np.array([[0.7, 0.3], [0.4, 0.6]])
    with pytest.raises(ValueError):
        HistogramBinning.fit(p, [0])              
    with pytest.raises(ValueError):
        HistogramBinning.fit(p, [0, 2])                
    with pytest.raises(ValueError):
        HistogramBinning.fit(np.array([[1.0], [1.0]]), [0, 0])   # K = 1
    with pytest.raises(ValueError):
        HistogramBinning.from_dict({"method": "temperature", "temperature": 1.0})