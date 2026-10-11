#!/usr/bin/env python3
"""Decodes the M17 signal in an IQ recording with the receive chain of pluto-tx's receiver (FM demodulator, DC removal, RRC matched filter,
symbol synchroniser, gr-m17 decoder), and prints the link setup frames it finds (source and destination call signs) and how much voice
it decoded. Nothing is transmitted.

    python3 scripts/m17_iq_check.py recording.cf32 --rate 2400000            # finds the signal by itself
    python3 scripts/m17_iq_check.py m17.cf32 --rate 80000 --offset 0         # the file made by m17_iq_make.py

The recording is complex float32 ('cf32'). The signal may sit anywhere inside it: the script looks for the strongest narrow signal
(or use --offset, its frequency against the centre of the recording in Hz)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plutotx


def find_offset(x, rate):
    """The centre of the strongest signal about 10 kHz wide, in Hz against the recording's centre."""
    import numpy as np
    n = 1 << 18
    seg = x[len(x) // 2 - n // 2:len(x) // 2 + n // 2] if len(x) > n else x
    sp = np.abs(np.fft.fftshift(np.fft.fft(seg * np.hanning(len(seg))))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(seg), 1 / rate))
    width = max(3, int(10_000 / (rate / len(seg))))
    smooth = np.convolve(sp, np.ones(width), mode="same")
    smooth[np.abs(f) < 15_000] = 0                                    # the LO's DC spike and its skirts
    k = int(np.argmax(smooth))
    w = slice(max(0, k - width), k + width)
    return float(np.sum(f[w] * sp[w]) / np.sum(sp[w]))


def to_48k(x, rate, offset):
    import numpy as np
    from math import gcd
    t0 = 0
    out = []
    from espdr import dsp
    step = 1_000_000
    res = None
    # a mixer back to baseband, in pieces; then the rational conversion to 48 kHz
    try:
        from scipy.signal import resample_poly
        g = gcd(int(rate), 48000)
        n = np.arange(len(x))
        y = x * np.exp(-2j * np.pi * offset * n / rate)
        return resample_poly(y, 48000 // g, int(rate) // g, window=("kaiser", 8.0)).astype(np.complex64)
    except ImportError:
        res = dsp.Resampler(rate, 48000)
        for i in range(0, len(x), step):
            seg = x[i:i + step] * np.exp(-2j * np.pi * offset * (np.arange(i, min(i + step, len(x)))) / rate)
            out.append(res.process(seg))
        return np.concatenate(out).astype(np.complex64)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="the IQ recording (cf32)")
    p.add_argument("--rate", type=float, required=True, help="its sample rate in Hz")
    p.add_argument("--offset", type=float, help="frequency of the signal against the centre of the recording in Hz (default: find it)")
    p.add_argument("--pluto-tx", default=str(_plutotx.DEFAULT_DIR), help="the pluto-tx source directory")
    args = p.parse_args(argv)
    _plutotx.prepare(args.pluto_tx)

    import math
    import numpy as np
    from gnuradio import analog, blocks, digital, gr
    from gnuradio.filter import firdes, fir_filter_fff
    from pluto_advanced_rx import flowgraph as rxf
    from pluto_advanced_rx.m17_deframer import M17FieldsDeframer
    from pluto_tx import config
    if not rxf.M17_AVAILABLE:
        raise SystemExit("m17_iq_check: gr-m17 is not available to pluto-tx")
    from gnuradio import m17 as _m17

    x = np.fromfile(args.file, dtype=np.complex64)
    offset = args.offset if args.offset is not None else find_offset(x.astype(np.complex128), args.rate)
    print(f"recording: {len(x)} samples at {args.rate:g} Hz ({len(x) / args.rate:.2f} s); signal at {offset:+.0f} Hz from the centre")
    sig = to_48k(x, args.rate, offset)

    found = []
    tb = gr.top_block()
    src = blocks.vector_source_c(sig.tolist(), False)
    demod = analog.quadrature_demod_cf(config.M17_BASEBAND_RATE / (2 * math.pi * config.M17_DEVIATION_HZ))
    n_avg = int(0.1 * config.M17_BASEBAND_RATE)
    avg = blocks.moving_average_ff(n_avg, 1.0 / n_avg, 4000)
    sub = blocks.sub_ff()
    mf = fir_filter_fff(1, firdes.root_raised_cosine(1.0, config.M17_BASEBAND_RATE, config.M17_SYMBOL_RATE, config.M17_RRC_ALPHA, config.M17_RRC_NTAPS))
    sync = digital.symbol_sync_ff(digital.TED_GARDNER, config.M17_RRC_SPS, 2 * math.pi * 0.0015, 1.0, 1.0, 0.05, 1,
                                  digital.constellation_bpsk().base(), digital.IR_MMSE_8TAP, 128, [])
    dec = _m17.m17_decoder(False, False, 2.0, 30.0, True, False, 0, "", "")
    deframer = M17FieldsDeframer(lambda fields: found.append(fields))
    c2 = _m17.codec2_decoder()
    audio = blocks.vector_sink_s()
    tb.connect(src, demod, sub)
    tb.connect(demod, avg)
    tb.connect(avg, (sub, 1))
    tb.connect(sub, mf, sync, dec)
    tb.msg_connect(dec, "fields", deframer, "fields")
    tb.connect(dec, c2, audio)
    tb.run()
    n_audio = len(audio.data())
    print(f"link setup frames decoded: {len(found)}")
    for fields in found[:3]:
        print(f"  source {fields.get('src')!r}  destination {fields.get('dst')!r}")
    print(f"decoded voice: {n_audio / 8000:.2f} s")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
