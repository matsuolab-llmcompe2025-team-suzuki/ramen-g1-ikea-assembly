from __future__ import annotations

import threading
import time

from inference.desktop.lower_policy.policies.deferred import DeferredPolicy


class _Inner:
    EXECUTION_HORIZON = 7
    constructed = 0
    closed = 0
    resets = 0

    @classmethod
    def from_ckpt(cls, cfg):  # noqa: ANN001
        cls.constructed += 1
        return cls()

    @staticmethod
    def build_state_from_raw(raw):  # noqa: ANN001
        return ("state", raw)

    def reset(self) -> None:
        type(self).resets += 1

    def predict(self, observation):  # noqa: ANN001
        return ("prediction", observation)

    def close(self) -> None:
        type(self).closed += 1


def _reset_counts() -> None:
    _Inner.constructed = 0
    _Inner.closed = 0
    _Inner.resets = 0


def test_loads_on_reset_and_releases_after_skill() -> None:
    _reset_counts()
    policy = DeferredPolicy(_Inner, label="test", loader=lambda: _Inner.from_ckpt(object()))
    assert _Inner.constructed == 0
    assert policy.EXECUTION_HORIZON == 7      # class から答えられる (読み込み不要)

    policy.reset()
    # state の作り方は ckpt によって変わるので、読み込んだ policy に委譲する
    # (Issue #141 P8-4)。
    assert policy.build_state_from_raw("raw") == ("state", "raw")
    assert _Inner.constructed == 1
    assert _Inner.resets == 1
    assert policy.loaded_during_last_reset
    assert policy.predict("obs") == ("prediction", "obs")

    policy.release_after_skill()
    assert _Inner.closed == 1


def test_residency_management_defers_skill_stop_release() -> None:
    """A skill boundary must not synchronously tear down the model worker."""
    _reset_counts()
    policy = DeferredPolicy(_Inner, label="managed", loader=_Inner)
    policy.prepare()
    policy.set_residency_managed(True)

    policy.release_after_skill()
    assert policy.is_loaded is True
    assert _Inner.closed == 0

    policy.release_from_residency()
    assert policy.is_loaded is False
    assert _Inner.closed == 1
    policy.close()
    assert _Inner.closed == 1


def test_validate_load_is_repeatable_and_leaves_nothing_loaded() -> None:
    _reset_counts()
    policy = DeferredPolicy(_Inner, label="test", loader=lambda: _Inner.from_ckpt(object()))
    policy.validate_load_and_release()
    policy.validate_load_and_release()
    assert _Inner.constructed == 2
    assert _Inner.closed == 2


def test_concurrent_loads_build_the_policy_once() -> None:
    """先読みの worker と tick の thread が同時に読んでも 1 つだけ作ること。

    Issue #141 の review 指摘 6: `_inner is None` の確認と代入を守らないと、
    3 秒かかる読み込みの間にもう一方が入り、13 GiB の ckpt が 2 つ GPU に載る
    (片方はどこからも参照されない)。
    """
    built: list[object] = []
    at_the_gate = threading.Barrier(2)

    def _loader() -> object:
        built.append(object())
        time.sleep(0.05)      # 読み込みの最中にもう一方が入ってくる窓
        return built[-1]

    policy = DeferredPolicy(object, label="race", loader=_loader)

    def _enter() -> None:
        at_the_gate.wait()
        policy.prepare()

    threads = [threading.Thread(target=_enter) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert len(built) == 1


def test_is_loaded_tracks_the_inner_policy():
    """`ModelResidency` がこれを見て「裏で解放された」を検出する。"""
    loads = []

    class _Inner:
        def close(self) -> None:
            pass

    def _loader():
        loads.append(1)
        return _Inner()

    policy = DeferredPolicy(_Inner, label="probe", loader=_loader)

    assert policy.is_loaded is False
    policy.prepare()
    assert policy.is_loaded is True
    policy.close()
    assert policy.is_loaded is False
    assert len(loads) == 1


def test_a_reset_that_waited_for_another_threads_load_says_so() -> None:
    """別 thread の読み込みを待った reset は `loaded_during_last_reset=True` (#159 B3a-03)。

    以前は lock の中で「もう載っている」を読んでいたので、待った事実が消えた。
    VlaSkill はこの flag で「待つ前に取った frame」を捨てるので、待ったのに
    False だと数秒前の画像で推論する。
    """
    loading, gate = threading.Event(), threading.Event()

    def loader():
        loading.set()
        gate.wait(5.0)
        return _Inner()

    policy = DeferredPolicy(_Inner, label="slow", loader=loader)
    worker = threading.Thread(target=policy.prepare)
    worker.start()
    assert loading.wait(5.0)  # worker が lock を握って読み込みに入った
    threading.Timer(0.05, gate.set).start()
    policy.reset()
    worker.join(5.0)
    assert policy.loaded_during_last_reset is True


def test_outputs_waist_delegates_to_the_loaded_policy():
    """腰を出すかは ckpt で決まる (Issue #144)。class ではなく instance に聞く。"""

    class _NoWaist:
        outputs_waist = False

    policy = DeferredPolicy(_NoWaist, label="ramen_ori", loader=_NoWaist)
    assert policy.outputs_waist is False


def test_outputs_waist_defaults_to_true_for_policies_without_the_attribute():
    """GR00T / ACT / DP は腰も学習して出すので、属性が無ければ従来どおり流す。"""

    class _Inner:
        pass

    policy = DeferredPolicy(_Inner, label="groot", loader=_Inner)
    assert policy.outputs_waist is True
