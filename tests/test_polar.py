import numpy as np
import pytest

from espdr import dsp, polar, txmodes as tm
from helpers import firmware_steps, m17_like_frequency, rrc_taps


def tone(freq, n, rate=80000, amp=1.0):
    return amp * np.exp(2j * np.pi * freq * np.arange(n) / rate)


def fm_signal(freq_track, rate):
    return np.exp(2j * np.pi * np.cumsum(freq_track) / rate)


def test_a_tone_gives_a_constant_frequency():
    f = polar.IqToPolar(80000).process(tone(1234.5, 8000))
    assert len(f) == 4000
    assert np.allclose(f[2:], 1234.5, atol=1e-6)


def test_a_negative_frequency_comes_out_negative():
    f = polar.IqToPolar(80000).process(tone(-6000, 8000))
    assert np.allclose(f[2:], -6000, atol=1e-6)


@pytest.mark.parametrize("sizes", [[160], [4096], [333], [1, 7, 1000, 3]])
def test_the_result_does_not_depend_on_the_block_size(sizes):
    t = np.arange(24000) / 80000.0
    x = fm_signal(2500 * np.sin(2 * np.pi * 700 * t), 80000) * (1 + 0.2 * np.sin(2 * np.pi * 50 * t))
    whole = polar.IqToPolar(80000).process(x)
    c = polar.IqToPolar(80000)
    parts, i, k = [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        parts.append(c.process(x[i:i + n]))
        i, k = i + n, k + 1
    assert np.allclose(np.concatenate(parts), whole, atol=1e-9)


def test_fm_gives_its_modulation_back():
    t = np.arange(16000) / 40000.0
    audio = 0.8 * np.sin(2 * np.pi * 1000 * t)                      # at the update rate
    x = fm_signal(np.repeat(2500 * audio, 2), 80000)               # 80 kHz FM, deviation 2500 Hz
    f = polar.IqToPolar(80000).process(x)
    n = np.arange(100, len(f) - 100)
    assert np.sqrt(np.mean((f[n] - 2500 * audio[n]) ** 2)) < 0.01 * 2000


def test_silence_holds_the_last_frequency_and_does_not_make_garbage():
    c = polar.IqToPolar(80000)
    a = c.process(tone(3000, 800))
    b = c.process(np.zeros(1600, dtype=complex))
    d = c.process(tone(3000, 800))
    assert np.allclose(b, 3000, atol=1e-6)                         # held, not 0 and not noise
    assert np.allclose(d[2:], 3000, atol=1e-6) and abs(a[-1] - 3000) < 1e-6


def test_the_first_block_starts_at_the_idle_frequency():
    f = polar.IqToPolar(80000).process(np.zeros(800, dtype=complex))
    assert not f.any()


def test_a_rate_that_is_not_a_multiple_resamples():
    c = polar.IqToPolar(48000)
    f = c.process(tone(2000, 48000, rate=48000))
    assert abs(len(f) - 40000) < 40
    assert np.allclose(f[100:-100], 2000, atol=1.0)


def test_the_iq_rate_cannot_be_below_the_update_rate():
    with pytest.raises(ValueError):
        polar.IqToPolar(30000)


def test_the_modulator_makes_records_inside_the_firmware_limits():
    m = polar.IqModulator(80000, power_db=-6, static_hz=1000.0)
    rec = m.process(tone(2500, 16000))
    q4 = (rec & 0xFFFF).astype(np.uint16).view(np.int16)
    assert rec.dtype == np.uint32 and len(rec) == 8000
    assert abs(q4[10:] * tm.Q4_HZ - 3500).max() < tm.Q4_HZ
    assert {int(c) for c in (rec >> 16) & 0xFF} == {tm.peak_code(-6)}
    assert not ((rec >> 24) & 0xFF).any() and m.clipped == 0


def test_a_frequency_outside_the_range_is_counted_and_clipped():
    m = polar.IqModulator(80000)
    rec = m.process(tone(30000, 800))
    q4 = (rec & 0xFFFF).astype(np.uint16).view(np.int16)
    assert m.clipped > 300 and abs(q4).max() == tm.MAX_Q4


def test_the_power_can_change_while_running():
    m = polar.IqModulator(80000, power_db=0.0)
    a = m.process(tone(1000, 800))
    m.set_power(-12.0)
    b = m.process(tone(1000, 800))
    assert (a >> 16 & 0xFF).max() == tm.peak_code(0.0) and (b >> 16 & 0xFF).min() == tm.peak_code(-12.0)


def test_empty_blocks_are_fine():
    m = polar.IqModulator(80000)
    assert len(m.process(np.zeros(0, dtype=complex))) == 0


def band_error_db(error, signal, limit_hz, rate=40000):
    """Power of `error` inside |f| < limit_hz against the power of `signal` there, in dB."""
    w = np.hanning(len(error))
    e, s = np.abs(np.fft.rfft(error * w)) ** 2, np.abs(np.fft.rfft(signal * w)) ** 2
    m = np.fft.rfftfreq(len(error), 1 / rate) < limit_hz
    return 10 * np.log10(e[m].sum() / s[m].sum())


def m17_pipeline(n_sym=1200):
    symbols, f48 = m17_like_frequency(n_sym)
    x80 = dsp.Resampler(48000, 80000).process(fm_signal(f48, 48000))
    return symbols, f48, polar.IqToPolar(80000).process(x80), polar.IqModulator(80000).process(x80)


def test_m17_like_4fsk_quantisation_stays_far_below_the_signal():
    _, f48, f_exact, rec = m17_pipeline()
    played = firmware_steps(rec)                                  # Hz at 40 kHz, with the chip's own quantisation noise
    seg = slice(2000, -2000)
    assert np.abs(f_exact).max() > 2500                           # the test signal has M17's deviation
    host = np.rint(f_exact / tm.Q4_HZ) * tm.Q4_HZ - f_exact
    assert band_error_db(host[seg], f_exact[seg], 4000) < -50      # records: 28.6 Hz units
    assert band_error_db((played - f_exact)[seg], f_exact[seg], 4000) < -40   # plus the firmware's noise-shaped PLL steps


def test_m17_like_4fsk_has_no_symbol_errors_after_the_pipeline():
    symbols, f48, _, rec = m17_pipeline()
    back = dsp.Resampler(40000, 48000).process(firmware_steps(rec))
    seg = slice(1500, 10000)                                      # align the chain's delay (whole samples), then the receiver's matched filter
    lags = list(range(-200, 201))
    lag = lags[int(np.argmax([np.dot(back[seg.start + l:seg.stop + l], f48[seg]) for l in lags]))]
    back = back[lag:] if lag > 0 else np.concatenate([np.zeros(-lag), back])
    h = rrc_taps()
    k = np.arange(20, len(symbols) - 20)
    y_ideal, y_got = np.convolve(f48, h)[80 + 10 * k], np.convolve(back[:len(f48)], h)[80 + 10 * k]
    unit = np.abs(y_ideal[np.abs(symbols[k]) == 1]).mean()        # the matched filter's output for symbol 1
    sym = np.array([-3, -1, 1, 3])
    thresholds = np.array([-2, 0, 2]) * unit
    assert (sym[np.searchsorted(thresholds, y_ideal)] == symbols[k]).all()
    assert (sym[np.searchsorted(thresholds, y_got)] == symbols[k]).all()
