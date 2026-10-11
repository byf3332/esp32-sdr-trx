import io

import numpy as np
import pytest

from espdr import iq


def cf32(x):
    v = np.empty(2 * len(x), dtype="<f4")
    v[0::2], v[1::2] = x.real, x.imag
    return v.tobytes()


class Dribble(io.RawIOBase):
    """A pipe that hands out a few bytes at a time."""

    def __init__(self, data, chunk=7):
        self.data, self.chunk = data, chunk

    def read(self, n=-1):
        out, self.data = self.data[:min(n, self.chunk)], self.data[min(n, self.chunk):]
        return out


def test_cf32_blocks_give_back_the_samples():
    x = (np.arange(5000) + 1j * np.arange(5000)[::-1]) / 5000
    src = iq.IqSource(io.BytesIO(cf32(x)), 80000, "cf32")
    blocks = list(src.blocks())
    assert [len(b) for b in blocks[:3]] == [1600, 1600, 1600] and sum(len(b) for b in blocks) == 5000
    assert np.allclose(np.concatenate(blocks), x, atol=1e-6)


@pytest.mark.parametrize("fmt,dtype,full", [("cs16", "<i2", 32768), ("cs8", "i1", 128)])
def test_the_integer_formats_are_scaled_to_full_scale_one(fmt, dtype, full):
    raw = np.array([full // 2, -full // 2, 0, -full], dtype=dtype)
    out = np.concatenate(list(iq.IqSource(io.BytesIO(raw.tobytes()), 1000, fmt).blocks()))
    assert np.allclose(out, [0.5 - 0.5j, 0 - 1j])


def test_a_pipe_that_hands_out_a_few_bytes_at_a_time_gives_the_same_blocks():
    x = np.exp(2j * np.pi * 0.01 * np.arange(3000))
    whole = np.concatenate(list(iq.IqSource(io.BytesIO(cf32(x)), 80000).blocks()))
    dribble = np.concatenate(list(iq.IqSource(Dribble(cf32(x)), 80000).blocks()))
    assert np.allclose(whole, dribble) and len(dribble) == 3000


def test_a_half_sample_at_the_end_is_dropped():
    raw = cf32(np.ones(10, dtype=complex)) + b"\x00\x00\x80"
    assert sum(len(b) for b in iq.IqSource(io.BytesIO(raw), 1000).blocks()) == 10


def test_a_file_reports_how_far_it_has_been_read(tmp_path):
    f = tmp_path / "a.cf32"
    f.write_bytes(cf32(np.ones(8000, dtype=complex)))
    src = iq.open_iq(str(f), 80000)
    it = src.blocks()
    next(it)
    assert 0.15 < src.fraction() < 0.25 and src.name == "a.cf32"
    list(it)
    assert src.fraction() == 1.0
    src.close()


def test_errors_are_clear(tmp_path):
    with pytest.raises(iq.IqError, match="cannot open"):
        iq.open_iq(str(tmp_path / "missing"), 80000)
    with pytest.raises(iq.IqError, match="unknown IQ format"):
        iq.IqSource(io.BytesIO(b""), 80000, "cf64")
