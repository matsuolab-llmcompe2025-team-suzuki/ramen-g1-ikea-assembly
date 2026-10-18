import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.desktop.perception.venue_state_guard import (
    DEFAULT_PROFILE, StateAttestor, check_profile, state_fingerprint,
)
from inference.desktop.perception.guarded_state_stream import GuardedStateStream


def message(tick=1, q=0.2):
    motors = [SimpleNamespace(q=q) for _ in range(35)]
    motors[31].q = -0.72
    motors[33].q = 0.0
    return SimpleNamespace(tick=tick, motor_state=motors,
                           imu_state=SimpleNamespace(quaternion=[1., 0., 0., 0.]))


def setup():
    clock = SimpleNamespace(now=10.)
    guard = StateAttestor({"left": [-0.72, 4.64], "right": [0., 5.38]}, clock=lambda: clock.now)
    payload = {"body_q": [0.2] * 29, "base_quat": [1., 0., 0., 0.],
               "gripper_q": {"left": {"q": 0.}, "right": {"q": 0.}}}
    guard.observe(message(1))
    clock.now += .01
    guard.observe(message(2))
    return clock, guard, payload


def wire(payload, proof, nonce=b"x" * 32):
    msgpack = pytest.importorskip("msgpack")
    return [nonce, b"g1_debug" + msgpack.packb(dict(payload, ramen_source=proof))]


def test_stationary_robot_is_fresh_if_real_dds_ticks_advance():
    clock, guard, payload = setup()
    first = guard.attest(payload)
    clock.now += .02
    guard.observe(message(3))
    second = guard.attest(payload)
    assert second["sequence"] > first["sequence"]
    assert second["age_s"] == 0


@pytest.mark.parametrize("repeated_callback", [False, True])
def test_frozen_dds_cannot_be_refreshed_by_bridge_republication(repeated_callback):
    clock, guard, payload = setup()
    clock.now += .3
    if repeated_callback:
        guard.observe(message(2))
    with pytest.raises(ValueError, match="no fresh matching DDS"):
        guard.attest(payload)


def test_stuck_bridge_does_not_match_fresh_moving_dds():
    clock, guard, payload = setup()
    clock.now += .3
    guard.observe(message(3, q=.8))
    with pytest.raises(ValueError, match="no fresh matching DDS"):
        guard.attest(payload)


def test_reboot_tick_regression_is_latched_until_guard_restart():
    clock, guard, payload = setup()
    guard.observe(message(0))
    guard.observe(message(4))
    with pytest.raises(ValueError, match="tick regressed"):
        guard.attest(payload)


def test_uint32_tick_wrap_is_a_valid_new_sample():
    clock, guard, payload = setup()
    guard.tick = 2**32 - 2
    guard.observe(message(1))
    assert guard.attest(payload)["tick"] == 1


@pytest.mark.parametrize("bad", [None, True, -1, 2**32])
def test_invalid_tick_never_produces_proof(bad):
    clock, guard, payload = setup()
    guard.observe(message(bad))
    clock.now += .3  # only the invalid sample is recent
    with pytest.raises(ValueError, match="invalid DDS tick"):
        guard.attest(payload)


def test_one_malformed_dds_sample_is_dropped_not_latched():
    clock, guard, payload = setup()
    broken = message(3)
    broken.motor_state[5].q = float("nan")
    guard.observe(broken)
    clock.now += .01
    guard.observe(message(4))
    assert guard.attest(payload)["tick"] == 4
    assert guard.rejected_count == 1


def test_attest_waits_for_a_dds_callback_that_lags_the_bridge():
    from inference.desktop.perception.venue_state_guard import NoFreshMatch, attest_when_seen
    clock, guard, payload = setup()
    moved = dict(payload, body_q=[.3] * 29)
    arrivals = [(clock.now + .02, message(3, q=.3))]  # the bridge relayed it 20 ms earlier

    def sleep(dt):
        clock.now += dt
        while arrivals and arrivals[0][0] <= clock.now:
            guard.observe(arrivals.pop(0)[1])

    with pytest.raises(NoFreshMatch):
        guard.attest(moved)
    assert attest_when_seen(guard, moved, clock=lambda: clock.now, sleep=sleep)["tick"] == 3
    clock.now += 1.  # a source that really stopped still fails after the bounded wait
    with pytest.raises(NoFreshMatch):
        attest_when_seen(guard, moved, clock=lambda: clock.now, sleep=sleep)


@pytest.mark.parametrize("drop", [.3, .6])
def test_dds_samples_missed_by_the_guard_do_not_starve_the_thor_state(drop):
    """500 Hz lowstate, bridge relays its latest sample at 50 Hz, Thor asks at 50 Hz.

    A sample the guard's own callback missed cannot be attested, so that one request
    fails. Thor treats state older than 0.5 s as stale (MAX_SOURCE_AGE_S); the gap
    between accepted states must stay far below it.
    """
    import random
    from inference.desktop.perception.venue_state_guard import NoFreshMatch
    rng = random.Random(7)
    clock = SimpleNamespace(now=0.)
    guard = StateAttestor({"left": [-0.72, 4.64], "right": [0., 5.38]}, clock=lambda: clock.now)
    bridge, last_ok, worst_gap, successes, requests = None, 0., 0., 0, 0
    for step in range(1, 5001):  # 10 s
        clock.now = step / 500
        sample = message(step, q=.2 + 1e-4 * step)
        if rng.random() >= drop:
            guard.observe(sample)
        if step % 10 == 0:  # the bridge's 50 Hz relay of its newest sample
            bridge = {"body_q": [sample.motor_state[0].q] * 29, "base_quat": [1., 0., 0., 0.],
                      "gripper_q": {"left": {"q": 0.}, "right": {"q": 0.}}}
        if step % 10 == 5 and bridge is not None:  # Thor's request, between relays
            requests += 1
            try:
                guard.attest(bridge)
            except NoFreshMatch:
                continue
            successes += 1
            worst_gap = max(worst_gap, clock.now - last_ok)
            last_ok = clock.now
    assert successes == requests
    assert worst_gap < .05


def test_a_neighbouring_sample_confirms_the_bridge_value_only_within_tolerance():
    from inference.desktop.perception.venue_state_guard import MATCH_TOLERANCE, NoFreshMatch
    clock, guard, payload = setup()
    assert guard.attest(payload)["match"] == "exact"
    clock.now += .002
    guard.observe(message(3, q=.2 + MATCH_TOLERANCE / 2))  # arm moved 5 mrad since the relay
    clock.now += .25  # tick 2 aged out; tick 3 is the only fresh sample
    proof = guard.attest(payload)
    assert (proof["tick"], proof["match"]) == (3, "tolerance")
    assert proof["max_diff"] == pytest.approx(MATCH_TOLERANCE / 2)
    guard.observe(message(4, q=.2 + 2 * MATCH_TOLERANCE))  # the bridge froze while the arm moved
    clock.now += .25
    with pytest.raises(NoFreshMatch):
        guard.attest(payload)


def test_a_guard_refusal_reuses_the_request_socket():
    import threading
    zmq = pytest.importorskip("zmq")
    msgpack = pytest.importorskip("msgpack")
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    stop = threading.Event()

    def refuse():
        while not stop.is_set():
            if server.poll(20):
                nonce = server.recv()
                server.send_multipart([nonce, msgpack.packb({"error": "waiting for advancing DDS ticks"})])

    thread = threading.Thread(target=refuse)
    thread.start()
    reader = GuardedStateStream("127.0.0.1", port)
    try:
        assert reader.read(timeout_ms=500) is None
        first = reader.socket
        assert first is not None
        assert reader.read(timeout_ms=500) is None
        assert reader.socket is first, "a refusal must not reconnect"
        assert "waiting for advancing DDS ticks" in reader.failure
        stop.set()
        thread.join(2)
        assert reader.read(timeout_ms=50) is None  # no reply: the stuck REQ socket is rebuilt
        assert reader.socket is None
    finally:
        stop.set()
        thread.join(2)
        reader.close()
        server.close()
        context.term()


def test_wrong_hand_calibration_is_not_certified():
    _, guard, payload = setup()
    payload["gripper_q"]["left"]["q"] = -.72
    with pytest.raises(ValueError, match="no fresh matching DDS"):
        guard.attest(payload)


@pytest.mark.parametrize("fraction", [0., .1, .5, .9, 1.])
def test_internal_hand_normalization_including_reversed_raw_axis(fraction):
    clock, guard, payload = setup()
    msg = message(3)
    for side, index in (("left", 31), ("right", 33)):
        closed, opened = guard.calibration[side]
        raw = closed + fraction * (opened - closed)
        msg.motor_state[index].q = raw
        payload["gripper_q"][side]["q"] = (raw - closed) * (-5.3 / (opened - closed))
    guard.observe(msg)
    assert guard.attest(payload)["tick"] == 3


def test_guarded_hand_wait_does_not_return_command_echo():
    from inference.desktop.perception.boundary_state_source import BoundaryDex1StateSource
    joint = SimpleNamespace(_guarded=True, latest_gripper=lambda: None)
    synthetic = SimpleNamespace(get=lambda: pytest.fail("synthetic fallback"))
    assert BoundaryDex1StateSource(joint, synthetic).get() is None


def test_the_default_venue_path_reads_5557_without_the_guard(monkeypatch):
    from inference.desktop import entrypoint
    from inference.desktop.perception import boundary_state_source as bss
    monkeypatch.setattr("sys.argv", ["entrypoint", "--action-sink", "boundary"])
    monkeypatch.setenv("IROS_ORIN_HOST", "192.168.123.164")
    args = entrypoint.parse_args()
    assert (args.boundary_state_guard, args.boundary_state_port) == (False, 5557)
    entrypoint.configure_official_endpoints(args)  # no guard requirement without the option
    assert (args.boundary_state_host, args.boundary_state_port) == ("192.168.123.164", 5557)
    legacy = SimpleNamespace(read=lambda timeout_ms: None, close=lambda: None)
    monkeypatch.setattr(bss, "_gripper_aware_stream", lambda host, port: legacy)
    source = bss.BoundaryJointStateSource("127.0.0.1", 5557)
    try:
        assert source._stream is legacy and not source._guarded
    finally:
        source.close()


def test_guard_rejects_legacy_port_and_sdk_route():
    from inference.desktop.entrypoint import configure_official_endpoints
    for sink, port, reason in (("sdk", 5558, "requires"), ("boundary", 5557, "REQ/REP")):
        args = SimpleNamespace(boundary_state_guard=True, action_sink=sink, boundary_state_port=port)
        with pytest.raises(ValueError, match=reason):
            configure_official_endpoints(args)


def test_nonfinite_state_rejected():
    _, _, payload = setup()
    payload["body_q"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        state_fingerprint(payload)


@pytest.mark.requires_boundary
def test_guarded_decoder_preserves_state_and_conservative_age():
    _, guard, payload = setup()
    reader = GuardedStateStream("127.0.0.1", 5558)
    proof = dict(guard.attest(payload), age_s=.1)
    s = reader.decode_reply(b"x"*32, wire(payload, proof), 1_000_000_000, 1_050_000_000)
    assert s.body_q.tolist() == pytest.approx(payload["body_q"])
    assert s.source_received_ns == 900_000_000
    assert s.source_tick == 2
    assert s.gripper_q.tolist() == [0., 0.]


@pytest.mark.requires_boundary
def test_replayed_proof_does_not_refresh_state():
    _, guard, payload = setup()
    reader = GuardedStateStream("127.0.0.1", 5558)
    response = wire(payload, guard.attest(payload))
    assert reader.decode_reply(b"x"*32, response, 1_000_000_000, 1_010_000_000) is not None
    assert reader.decode_reply(b"x"*32, response, 1_100_000_000, 1_110_000_000) is None


@pytest.mark.parametrize(
    "kind",
    ["nonce", "age", "nan_age", "negative_age", "rtt", "missing",
     pytest.param("hand", marks=pytest.mark.requires_boundary), "restart"],
)
def test_invalid_or_restarted_guard_fails_closed(kind):
    _, guard, payload = setup()
    reader = GuardedStateStream("127.0.0.1", 5558)
    proof = guard.attest(payload)
    nonce, end = b"x"*32, 1_010_000_000
    if kind == "nonce": nonce = b"y"*32
    if kind == "age": proof["age_s"] = .51
    if kind == "nan_age": proof["age_s"] = float("nan")
    if kind == "negative_age": proof["age_s"] = -.1
    if kind == "rtt": end = 1_600_000_000
    if kind == "missing": proof = None
    if kind == "hand": payload.pop("gripper_q")
    if kind == "restart": reader.session = "0"*32
    with pytest.raises(ValueError):
        reader.decode_reply(nonce, wire(payload, proof), 1_000_000_000, end)


def profile_fixture(tmp_path):
    profile = json.loads(DEFAULT_PROFILE.read_text())
    for rel in profile["files_sha256"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        profile["files_sha256"][rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    env = {k: v.replace("${HOME}", str(tmp_path)) for k, v in profile["environment"].items()}
    return profile, env


def test_pc2_profile_checks_read_only_files_and_environment(tmp_path):
    profile, env = profile_fixture(tmp_path)
    check_profile(profile, tmp_path, env)
    (tmp_path / "wbc_adapter/deploy/run_wbc_with_dex1.py").write_text("old wrapper")
    with pytest.raises(ValueError, match="missing or changed"):
        check_profile(profile, tmp_path, env)


def test_check_only_never_imports_sdk_or_opens_transport(tmp_path, monkeypatch):
    import builtins
    from inference.desktop.perception import venue_state_guard
    profile, env = profile_fixture(tmp_path)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(venue_state_guard.os, "environ", env)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    original = builtins.__import__

    def audited_import(name, *args, **kwargs):
        assert not name.startswith(("unitree_sdk2py", "zmq")), name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", audited_import)
    assert venue_state_guard.main(["--profile", str(path), "--check-only"]) == 0


@pytest.mark.parametrize("name", ["RMW_IMPLEMENTATION", "ROS_DISTRO", "ROS_DOMAIN_ID", "AMENT_PREFIX_PATH",
                                  "CONDA_PREFIX", "IROS_DEX1_LEFT_OPEN_Q", "LD_PRELOAD"])
def test_missing_or_polluted_environment_rejected(tmp_path, name):
    profile, env = profile_fixture(tmp_path)
    env[name] = "wrong"
    with pytest.raises(ValueError):
        check_profile(profile, tmp_path, env)


def test_guard_contains_no_robot_command_publisher():
    from inference.desktop.perception import venue_state_guard
    source = Path(venue_state_guard.__file__).read_text()
    for forbidden in ("ChannelPublisher", "LowCmd_", "rt/lowcmd", "rt/arm_sdk", "MotionSwitcherClient"):
        assert forbidden not in source


@pytest.mark.requires_boundary
def test_real_loopback_nonce_protocol_and_receiver_thread_cleanup():
    import threading
    import time
    zmq = pytest.importorskip("zmq")
    from inference.desktop.perception.boundary_state_source import BoundaryJointStateSource
    clock, guard, payload = setup()
    ready = threading.Event()
    stopping = threading.Event()
    ports = []
    def serve():
        context = zmq.Context()
        server = context.socket(zmq.REP)
        server.setsockopt(zmq.LINGER, 0)
        ports.append(server.bind_to_random_port("tcp://127.0.0.1"))
        ready.set()
        try:
            while not stopping.is_set():
                if not server.poll(50): continue
                nonce = server.recv()
                clock.now += .02
                guard.observe(message(guard.tick+1))
                server.send_multipart(wire(payload, guard.attest(payload), nonce))
        finally:
            server.close()
            context.term()
    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(2)
    source = BoundaryJointStateSource(host="127.0.0.1", port=ports[0], guarded=True)
    try:
        deadline = time.monotonic()+2
        while source.get() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert source.get() is not None
        assert source.get().tick >= 2
        assert source.latest_gripper() is not None
    finally:
        source.close()
        stopping.set()
        thread.join(2)
    assert not thread.is_alive()
    assert not source._thread.is_alive()


@pytest.mark.requires_boundary
def test_production_relay_rejects_live_bridge_republishing_dead_dds():
    import socket
    import threading
    import time
    msgpack = pytest.importorskip("msgpack")
    zmq = pytest.importorskip("zmq")
    from inference.desktop.perception.venue_state_guard import serve
    from inference.desktop.perception.boundary_state_source import BoundaryJointStateSource

    guard = StateAttestor({"left": [-.72, 4.64], "right": [0., 5.38]})
    _, _, payload = setup()
    stop, freeze, publisher_ready, relay_ready = [threading.Event() for _ in range(4)]
    ports, errors = [], []

    def publish():
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        ports.append(pub.bind_to_random_port("tcp://127.0.0.1"))
        publisher_ready.set()
        tick = 0
        try:
            while not stop.is_set():
                if not freeze.is_set():
                    tick += 1
                guard.observe(message(tick))
                pub.send(b"g1_debug" + msgpack.packb(payload))
                time.sleep(.01)
        finally:
            pub.close(linger=0)
            context.term()

    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    publisher = threading.Thread(target=publish)
    publisher.start()
    assert publisher_ready.wait(2)
    profile = {"upstream_endpoint": f"tcp://127.0.0.1:{ports[0]}",
               "guard_port": port, "source_max_age_s": .25}

    def relay():
        try:
            serve(guard, profile, "127.0.0.1", stop_event=stop, ready=relay_ready)
        except Exception as exc:
            errors.append(exc)

    server = threading.Thread(target=relay)
    server.start()
    source = None
    try:
        assert relay_ready.wait(2)
        source = BoundaryJointStateSource("127.0.0.1", port, guarded=True)
        deadline = time.monotonic() + 3
        while source.get() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert source.get() is not None
        freeze.set()
        time.sleep(.6)
        previous = source.get()
        time.sleep(.2)
        assert source.get() is previous, "republished bridge data renewed dead source"
        assert time.monotonic_ns() - previous.received_monotonic_ns > 500_000_000
    finally:
        if source is not None:
            source.close()
        stop.set()
        publisher.join(2)
        server.join(2)
    assert not errors
    assert not publisher.is_alive() and not server.is_alive()


@pytest.mark.requires_boundary
def test_a_failed_reply_rebuilds_the_guard_socket_instead_of_ending_the_guard(monkeypatch):
    import socket
    import threading
    import time
    msgpack = pytest.importorskip("msgpack")
    zmq = pytest.importorskip("zmq")
    from inference.desktop.perception.venue_state_guard import serve
    from inference.desktop.perception.boundary_state_source import BoundaryJointStateSource

    original = zmq.Socket.send_multipart
    failures = []

    def fail_first_reply(self, frames, *args, **kwargs):
        if not failures:
            failures.append(True)
            raise zmq.Again("simulated reply timeout")
        return original(self, frames, *args, **kwargs)

    monkeypatch.setattr(zmq.Socket, "send_multipart", fail_first_reply)
    guard = StateAttestor({"left": [-.72, 4.64], "right": [0., 5.38]})
    _, _, payload = setup()
    stop, publisher_ready, relay_ready = [threading.Event() for _ in range(3)]
    ports, errors = [], []

    def publish():
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        ports.append(pub.bind_to_random_port("tcp://127.0.0.1"))
        publisher_ready.set()
        tick = 0
        try:
            while not stop.is_set():
                tick += 1
                guard.observe(message(tick))
                pub.send(b"g1_debug" + msgpack.packb(payload))
                time.sleep(.01)
        finally:
            pub.close(linger=0)
            context.term()

    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    publisher = threading.Thread(target=publish)
    publisher.start()
    assert publisher_ready.wait(2)
    profile = {"upstream_endpoint": f"tcp://127.0.0.1:{ports[0]}",
               "guard_port": port, "source_max_age_s": .25}

    def relay():
        try:
            serve(guard, profile, "127.0.0.1", stop_event=stop, ready=relay_ready)
        except Exception as exc:
            errors.append(exc)

    server = threading.Thread(target=relay)
    server.start()
    source = None
    try:
        assert relay_ready.wait(2)
        source = BoundaryJointStateSource("127.0.0.1", port, guarded=True)
        deadline = time.monotonic() + 3
        while source.get() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert failures, "the first reply was not attempted"
        assert source.get() is not None, "the guard stopped serving after one failed reply"
    finally:
        if source is not None:
            source.close()
        stop.set()
        publisher.join(2)
        server.join(2)
    assert not errors
    assert not publisher.is_alive() and not server.is_alive()


@pytest.mark.requires_boundary
def test_a_delayed_reply_does_not_slow_the_measured_velocity():
    """Codex review of #185: the freshness stamp (request time - source age) of a reply
    delayed 300 ms in transit is 300 ms early. Velocity must use the sample spacing."""
    import time
    msgpack = pytest.importorskip("msgpack")
    from inference.desktop.perception.boundary_state_source import BoundaryJointStateSource
    clock = SimpleNamespace(now=100.)
    guard = StateAttestor({"left": [-.72, 4.64], "right": [0., 5.38]}, clock=lambda: clock.now)
    speed, ticks = .15, iter(range(1, 100))

    def sample(t):
        clock.now = t
        q = .5 + speed * (t - 100.)
        guard.observe(message(next(ticks), q=q))
        return {"body_q": [q] * 29, "base_quat": [1., 0., 0., 0.],
                "gripper_q": {"left": {"q": 0.}, "right": {"q": 0.}}}

    sample(99.99)
    reader, nonce, states = GuardedStateStream("127.0.0.1", 5558), b"x" * 32, []
    base = time.monotonic_ns() - 400_000_000
    for sent_ms, taken_s, rtt_ms in ((0, 100.300, 310), (340, 100.340, 1)):  # A delayed, B on time
        payload = sample(taken_s)
        reply = [nonce, b"g1_debug" + msgpack.packb(dict(payload, ramen_source=guard.attest(payload)))]
        started = base + sent_ms * 1_000_000
        states.append(reader.decode_reply(nonce, reply, started, started + rtt_ms * 1_000_000))
    assert states[1].source_received_ns - states[0].source_received_ns == pytest.approx(340e6, abs=2e6)
    replies = iter(states)
    stream = SimpleNamespace(read=lambda timeout_ms=200: next(replies, None) or time.sleep(.01),
                             close=lambda: None)
    source = BoundaryJointStateSource(stream=stream, guarded=True)
    try:
        deadline = time.monotonic() + 2
        while (source.get() is None or source.get().tick != states[1].source_tick) and time.monotonic() < deadline:
            time.sleep(.01)
        latest = source.get()
    finally:
        source.close()
    assert latest.velocity[15] == pytest.approx(speed, rel=.05)
    assert latest.received_monotonic_ns == states[1].source_received_ns  # freshness stays conservative


def test_the_closest_sample_confirms_a_value_not_merely_the_newest():
    from inference.desktop.perception.venue_state_guard import MATCH_TOLERANCE
    clock, guard, payload = setup()
    moved = dict(payload, body_q=[.2 + MATCH_TOLERANCE / 4] * 29)  # never observed exactly
    clock.now += .002
    guard.observe(message(3, q=.2 + MATCH_TOLERANCE / 2))
    clock.now += .002
    guard.observe(message(4, q=.2 + MATCH_TOLERANCE * .9))
    proof = guard.attest(moved)
    assert proof["match"] == "tolerance"
    assert proof["max_diff"] == pytest.approx(MATCH_TOLERANCE / 4)  # tick 2 or 3, not tick 4
    assert proof["tick"] in (2, 3)
