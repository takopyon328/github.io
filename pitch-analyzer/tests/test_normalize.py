import numpy as np
import pytest

from pitchan.normalize import (
    speaker_reference,
    time_normalized_contour,
    to_log_z,
    to_semitone,
)


def test_speaker_reference_geometric_mean():
    f0 = np.array([100.0, 0.0, 400.0])
    ref, mu, sigma = speaker_reference([f0])
    assert ref == pytest.approx(200.0)  # 幾何平均、無声(0)は除外
    assert mu == pytest.approx(np.log(200.0))


def test_to_semitone():
    st = to_semitone(np.array([200.0, 400.0, 0.0]), ref_hz=200.0)
    assert st[0] == pytest.approx(0.0)
    assert st[1] == pytest.approx(12.0)  # 1 オクターブ = 12 半音
    assert np.isnan(st[2])


def test_to_log_z():
    f0 = np.array([100.0, 200.0, 0.0])
    z = to_log_z(f0, mu=np.log(100.0), sigma=np.log(2.0))
    assert z[0] == pytest.approx(0.0)
    assert z[1] == pytest.approx(1.0)
    assert np.isnan(z[2])


def test_contour_resampling():
    times = np.arange(0, 1.0, 0.005)
    values = times * 10.0  # 線形に上昇する輪郭
    c = time_normalized_contour(times, values, 0.0, 1.0, n_points=11)
    assert c[0] == pytest.approx(0.0, abs=0.1)
    assert c[-1] == pytest.approx(10.0, abs=0.1)
    assert c[5] == pytest.approx(5.0, abs=0.1)


def test_contour_too_few_voiced():
    times = np.arange(0, 1.0, 0.005)
    values = np.full_like(times, np.nan)
    values[10] = 5.0
    c = time_normalized_contour(times, values, 0.0, 1.0, n_points=10)
    assert np.isnan(c).all()


def test_range_from_f0():
    from pitchan.f0 import range_from_f0

    rng = np.random.default_rng(0)
    f0 = np.zeros(1000)
    f0[:500] = rng.normal(120, 15, 500).clip(80, 200)  # 男性話者相当の分布
    floor, ceil = range_from_f0(f0)
    # Q25*0.75 ~ 80台, Q75*1.5 ~ 190台 になるはず
    assert 50 <= floor < 110
    assert 150 < ceil <= 600
    assert ceil >= floor * 2
    # 有声フレームが少なければ既定値
    assert range_from_f0(np.zeros(100)) == (60.0, 500.0)


def test_extract_f0_zeroes_out_of_range():
    from pitchan.f0 import extract_f0

    sr = 16000
    x = np.zeros(sr)
    period = int(sr / 100)  # 100 Hz のパルス列
    x[::period] = 0.5
    _, f0 = extract_f0(x, sr, f0_floor=150.0, f0_ceil=400.0)
    # 100 Hz は範囲外なので有声フレームに 150 未満の値が残らない
    assert not ((f0 > 0) & (f0 < 150.0)).any()
