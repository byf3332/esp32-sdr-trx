#!/usr/bin/env python3
"""Makes an IQ file of an M17 voice transmission with the transmit chain of pluto-tx (gr-m17, codec2, RRC, FM modulator), to be played by
`espdr-tx -m iq` (the ESP32-S3 sends its frequency; the envelope of M17 is constant).

    python3 scripts/m17_iq_make.py --wav speech.wav --src DL1ABC --out m17.cf32
    espdr-tx -f 2350 -m iq -i m17.cf32 --iq-rate 80000 --power -12

pluto-tx (GPL-3) is used as it is installed on this machine (--pluto-tx DIR, default ~/Dokumente/plutosdr); nothing of it is copied into this
repository. It needs gnuradio and gr-m17; the script starts itself again with the library path that pluto-tx's own launcher uses. The chain
runs at a PlutoSDR's rate (--chain-rate, 2.5 MHz: some of pluto-tx's branches cannot be built for a device that runs at 80 kHz), and the narrow
M17 signal is then brought down to --rate (default 80000, the rate of the pluto-tx device of the ESP32). The file holds float32 I/Q pairs
('cf32'). Nothing is transmitted."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plutotx

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", required=True, help="the voice to send (a WAV file, mono)")
    p.add_argument("--src", required=True, help="your call sign (goes into the M17 link setup frame; up to 9 characters)")
    p.add_argument("--dst", default="", help="destination (default: the pluto-tx default, ALL)")
    p.add_argument("--out", required=True, help="the IQ file to write (float32 I/Q pairs)")
    p.add_argument("--rate", type=int, default=80000, help="sample rate of the file (default 80000)")
    p.add_argument("--chain-rate", type=int, default=2_500_000, help="the rate pluto-tx's chain runs at (default 2500000, a PlutoSDR's)")
    p.add_argument("--pluto-tx", default=str(_plutotx.DEFAULT_DIR), help="the pluto-tx source directory")
    args = p.parse_args(argv)
    _plutotx.prepare(args.pluto_tx)

    import numpy as np
    import wave
    from gnuradio import blocks
    from pluto_tx import config, devices
    from pluto_tx import flowgraph as txf
    from pluto_tx.devices.base import PowerStage, TxDevice

    if not txf.M17_AVAILABLE:
        raise SystemExit("m17_iq_make: gr-m17 is not available to pluto-tx (see install-m17.sh in the pluto-tx directory)")
    with wave.open(args.wav) as w:
        seconds = w.getnframes() / w.getframerate()
        if w.getnchannels() != 1:
            raise SystemExit("m17_iq_make: the WAV file must be mono")
    rate = args.chain_rate

    class FileIqDevice(TxDevice):
        """pluto-tx's own transmit chain with a sink that collects the samples."""
        device_type = "espdr-iq-file"
        display_name = "IQ file"
        connection_kind = "uri"
        DEFAULT_CONNECTION = "x"
        frequency_range_hz = (1e6, 6e9)
        sample_rate_range_hz = (rate, rate)
        default_sample_rate_hz = rate
        default_bandwidth_hz = None
        power_stages = (PowerStage("gain", "Gain", "continuous_db", -17.9, 0.0, "dB", True, -17.9),)
        default_power_ceiling = 0.0

        def prepare_for_start(self): pass
        def build_sink(self):
            self.sink = blocks.vector_sink_c()
            return self.sink
        def set_frequency(self, hz): pass
        def set_power(self, name, value): pass
        def pre_key(self): pass
        def post_unkey(self): pass
        def force_safe_state(self): pass
        def read_hw_state(self): return {}
        @staticmethod
        def probe_with_timeout(connection, timeout_s=5): return None
        @staticmethod
        def scan_devices_with_timeout(timeout_s=5): return {}, None

    devices.DEVICE_REGISTRY["espdr-iq-file"] = FileIqDevice
    fg = txf.PlutoTxFlowgraph(device_type="espdr-iq-file", frequency=2.35e9, mode=txf.PlutoTxFlowgraph.MODE_M17,
                              source=txf.PlutoTxFlowgraph.SRC_FILE, wav_path=args.wav, m17_src_callsign=args.src,
                              m17_dst_callsign=args.dst)
    fg.start()
    fg.key_ptt()
    time.sleep(seconds + 1.0)
    fg.unkey_ptt()
    time.sleep(config.M17_EOT_HOLD_S + 0.3)
    fg.finish_unkey_m17()
    fg.stop()
    fg.wait()
    iq = np.array(fg.device.sink.data(), dtype=np.complex64)
    on = np.flatnonzero(np.abs(iq) > 0.01)
    if not len(on):
        raise SystemExit("m17_iq_make: the transmit chain produced no signal")
    iq = iq[on[0]:on[-1] + 1]
    if args.chain_rate != args.rate:
        from espdr import dsp
        res = dsp.Resampler(args.chain_rate, args.rate)
        step = 100_000                                           # the resampler's kernel is long: work in pieces
        iq = np.concatenate([res.process(iq[i:i + step]) for i in range(0, len(iq), step)]).astype(np.complex64)
    iq.tofile(args.out)
    print(f"wrote {args.out}: {len(iq)} samples at {args.rate} Hz, {len(iq) / args.rate:.2f} s (the voice is {seconds:.2f} s)")


if __name__ == "__main__":
    main()
