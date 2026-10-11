"""Helpers shared by the tests of the modulators."""
import numpy as np

from espdr import txmodes as tm


def ideal_polar(rec, rate=40000):
    """The signal an ideal transmitter makes of the records: the gain path responds one update before the frequency path."""
    q4 = (rec & 0xFFFF).astype(np.uint16).view(np.int16).astype(float)
    ph = 2 * np.pi * np.cumsum(q4 * tm.Q4_HZ) / rate
    amp = 10 ** (-(tm.RANGE_DB - tm.db_of_code(((rec >> 16) & 0xFF).astype(float))) / 20)
    return np.roll(amp, -1) * np.exp(1j * ph)


def run(mod, x):
    return np.concatenate([mod.process(x[i:i + 160]) for i in range(0, len(x), 160)])


def spectrum(s, rate=40000):
    s = s[8000:-2000]
    sp = np.abs(np.fft.fftshift(np.fft.fft(s * np.hanning(len(s)))))
    f = np.fft.fftshift(np.fft.fftfreq(len(s), 1 / rate))
    return lambda hz: sp[np.abs(f - hz) < 40].max()


def instantaneous_frequency(sig, rate=40000):
    """The frequency of a complex signal in Hz, one value less than samples."""
    return np.angle(sig[1:] * np.conj(sig[:-1])) * rate / (2 * np.pi)


def firmware_steps(rec):
    """The frequency in Hz the chip plays for the records, one value per update: the firmware's second-order error feedback
    (noise transfer function (1 - z^-1)^2, firmware/src/radio.c) turns the frequency field (1/16 PLL steps) into whole steps, limited to
    +-44 steps. The result still carries the shaped quantisation noise, which sits far above the voice band."""
    q4 = (rec & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64)
    out = np.empty(len(q4))
    e1 = e2 = 0
    for i, q in enumerate(q4.tolist()):
        u = q * 4096 - 2 * e1 + e2
        w = (u + 0x8000) >> 16
        e2, e1 = e1, w * 65536 - u
        out[i] = max(-44, min(44, w))
    return out * tm.STEP_HZ


def rrc_taps(alpha=0.5, ntaps=81, sps=10):
    """Root-raised-cosine taps with the sum of squares 1 (so that two of them in a row form a raised cosine with the centre value 1)."""
    t = (np.arange(ntaps) - (ntaps - 1) / 2) / sps
    h = np.empty(ntaps)
    for i, x in enumerate(t):
        if abs(x) < 1e-9:
            h[i] = 1 - alpha + 4 * alpha / np.pi
        elif abs(abs(x) - 1 / (4 * alpha)) < 1e-9:
            h[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - alpha)) + 4 * alpha * x * np.cos(np.pi * x * (1 + alpha))) / (np.pi * x * (1 - (4 * alpha * x) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def m17_like_frequency(n_symbols, seed=1):
    """The frequency track (Hz, at 48 kHz) of M17-like 4FSK: random symbols +-1 and +-3, 4800 a second, RRC alpha 0.5, 800 Hz per symbol
    unit at the symbol instants (peaks between them reach about +-3 kHz). Returns (symbols, frequency track); symbol k sits at index
    40 + 10 k of the track, 80 + 10 k after a matched filter."""
    rng = np.random.default_rng(seed)
    symbols = rng.choice([-3, -1, 1, 3], n_symbols)
    pulses = np.zeros(n_symbols * 10)
    pulses[::10] = symbols
    h = rrc_taps()
    return symbols, 800.0 * np.convolve(pulses, h / h[len(h) // 2])
