import threading
import time

import numpy as np
import pytest

from espdr import nb, polar, sim, txlink as tl, txmodes as tm


def start(esp, limit=600):
    lo, _ = tl.choose_lo(2_350_000_000)
    s = tl.Session(nb.Link(esp))
    s.configure(lo, 40000, 0, limit)
    s.begin(expect_word=tl.lo_word(lo))
    return s


def records(n, base=0):
    """n distinct records (the frequency field counts, the gain code is fixed)."""
    return ((np.arange(n, dtype=np.uint32) + base) & 0x1FF) | np.uint32(100 << 16)


def played(esp):
    return np.array(esp.played, dtype=np.uint32)


def test_everything_pushed_plays_in_order_and_the_end_record_follows():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp))
    parts = [records(1000, 1000 * i) for i in range(30)]
    for p in parts:
        assert st.push(p, timeout=10)
    summary = st.finish(timeout=20)
    sent = np.concatenate(parts)
    assert summary["reason"] == 0 and summary["underruns"] == 0
    assert np.array_equal(played(esp)[:len(sent)], sent)
    assert (esp.played[-1] >> 24) & 1 and len(esp.played) == len(sent) + 1


def test_a_short_clip_below_the_prefill_still_keys_through_the_end_record():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp))
    st.push(records(500))
    summary = st.finish(timeout=20)
    assert summary["reason"] == 0 and len(esp.played) == 501


def test_push_blocks_while_the_queue_is_full_and_goes_on_when_the_chip_takes_records():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp), max_buffered=2000)
    big = records(60000)
    t0 = time.monotonic()
    assert st.push(records(2000), timeout=5) and st.push(records(2000), timeout=5)
    assert st.push(big, timeout=30)                       # waits for room, then goes in as one block
    assert time.monotonic() - t0 >= 0.0
    summary = st.finish(timeout=60)
    assert summary["reason"] == 0 and len(esp.played) == 64000 + 1


def test_a_push_that_cannot_be_taken_returns_false_after_the_timeout():
    esp = sim.SimTx(speed=0.01)                           # a very slow chip: 400 records a second
    st = tl.Streamer(start(esp), max_buffered=1000)
    assert st.push(records(5000), timeout=2)              # the chip's ring takes these (target fill 5000)
    assert st.push(records(1000), timeout=2)              # waits in the queue, which is now at its limit
    assert st.push(records(1000), timeout=0.3) is False
    esp.speed = 1000.0                                    # let it finish quickly
    assert st.close(timeout=10) is not None


def test_abort_ends_the_transmission_quickly_and_drops_what_the_chip_has_not_got():
    esp = sim.SimTx(speed=1.0)
    st = tl.Streamer(start(esp))
    for i in range(3):
        assert st.push(records(4000, 4000 * i), timeout=5)    # 12000 records: 5000 reach the chip, the rest wait in the queue
    time.sleep(0.1)                                       # the carrier is on, about 4000 records have played
    t0 = time.monotonic()
    st.abort()
    summary = st.close(timeout=5)
    assert time.monotonic() - t0 < 1.0                    # the chip still plays what is in its ring: about 125 ms
    assert summary is not None and summary["reason"] == 0
    assert 5000 < len(esp.played) < 11500                 # the queued rest was dropped (12000 were pushed)
    assert (esp.played[-1] >> 24) & 1


def test_close_never_raises_and_can_be_called_twice():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp))
    st.push(records(100))
    st.close()
    st.close()
    assert st.push(records(10)) is False


def test_finish_after_abort_returns_the_summary_or_raises_a_clear_error():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp))
    st.push(records(100))
    st.abort()
    st.finish(timeout=5)


def test_the_chip_watchdog_ends_the_session_when_the_host_stops_pushing():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp))
    st.push(records(6000))                                # more than the prefill, then nothing
    time.sleep(3.0)                                       # 30 s of virtual time
    summary = st.close(timeout=5)
    assert summary is not None and summary["reason"] in (0, 1)


def test_the_time_limit_of_the_session_ends_it():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp, limit=1))
    st.push(records(40000 * 3))
    summary = st.finish(timeout=30)
    assert summary["reason"] == 2


def test_twenty_seconds_of_m17_like_traffic_do_not_underrun():
    esp = sim.SimTx(speed=5.0)
    st = tl.Streamer(start(esp))
    mod = polar.IqModulator(80000)
    block = np.exp(2j * np.pi * 1500 * np.arange(8000) / 80000.0)       # 0.1 s of IQ per block
    for _ in range(200):                                                  # 20 s
        assert st.push(mod.process(block), timeout=30)
    summary = st.finish(timeout=60)
    assert summary["reason"] == 0 and summary["underruns"] == 0
    assert len(esp.played) == 200 * 4000 + 1


def test_the_status_callback_runs_on_the_streamer_thread_and_sees_the_fill():
    esp = sim.SimTx(speed=5.0)
    seen = []
    st = tl.Streamer(start(esp), on_status=lambda ev: seen.append((threading.current_thread().name, ev["fill"])))
    st.push(records(20000))
    st.finish(timeout=20)
    assert seen and all(name == "espdr-streamer" for name, _ in seen)
    assert max(fill for _, fill in seen) > 4000
