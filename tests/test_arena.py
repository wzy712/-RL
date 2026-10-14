"""``arena.py`` 的对局与统计测试。

运行::

    .venv/bin/python -m pytest tests/test_arena.py -q

全部用例使用 CPU + 极小网络（channels=16, num_res_blocks=1），单文件应在 1 分钟内跑完。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from alphazero import PolicyValueNet
from arena import (
    HeuristicAgent,
    MatchResult,
    ModelAgent,
    RandomAgent,
    SearchAgent,
    play_match,
)
from env import GomokuEnv

# 微小棋盘的快速配置：5x5 连三
_FAST = dict(board_size=5, win_length=3)


def _tiny_model(board_size: int = 5, in_planes: int = 4) -> PolicyValueNet:
    """随机初始化的极小网络（CPU，秒级）。"""
    return PolicyValueNet(board_size=board_size, in_planes=in_planes, channels=16, blocks=1)


def _make_result(wins: int, draws: int, games: int) -> MatchResult:
    """构造指定胜负的 MatchResult（黑白分项自洽）。"""
    losses = games - wins - draws
    return MatchResult(
        wins=wins,
        draws=draws,
        losses=losses,
        games=games,
        wins_as_black=wins,
        games_as_black=games,
        wins_as_white=0,
        games_as_white=0,
        lengths=[10] * games,
    )


# --------------------------------------------------------------------------- #
# 统计自洽 / 可复现
# --------------------------------------------------------------------------- #
def test_random_selfplay_stats_are_consistent() -> None:
    """随机 Agent 自战：三者和 == games，黑白分项与总数自洽。"""
    result = play_match(RandomAgent(seed=11), RandomAgent(seed=22), games=6, seed=0, **_FAST)

    assert result.games == 6
    assert result.wins + result.draws + result.losses == result.games
    assert result.wins_as_black + result.wins_as_white == result.wins
    assert result.games_as_black + result.games_as_white == result.games
    assert result.games_as_black == 3
    assert result.games_as_white == 3
    assert len(result.lengths) == 6
    assert all(1 <= n <= 25 for n in result.lengths)
    assert 0.0 <= result.win_rate <= 1.0


def test_odd_games_give_a_one_extra_black() -> None:
    """奇数局时 A 方多执一次黑，分项仍是整数且自洽。"""
    result = play_match(RandomAgent(seed=1), RandomAgent(seed=2), games=7, seed=0, **_FAST)
    assert result.games_as_black == 4
    assert result.games_as_white == 3
    assert result.games_as_black + result.games_as_white == result.games


def test_play_match_is_reproducible() -> None:
    """同一 seed 两次对局逐字段一致（含重复使用同一对 Agent 对象）。"""
    first = play_match(RandomAgent(seed=3), RandomAgent(seed=4), games=6, seed=7, **_FAST)
    second = play_match(RandomAgent(seed=3), RandomAgent(seed=4), games=6, seed=7, **_FAST)
    assert _fingerprint(first) == _fingerprint(second)

    # 复用同一对 Agent 对象（RNG 会被 play_match 重置）也应当稳定
    a, b = RandomAgent(seed=3), RandomAgent(seed=4)
    again1 = play_match(a, b, games=6, seed=7, **_FAST)
    again2 = play_match(a, b, games=6, seed=7, **_FAST)
    assert _fingerprint(again1) == _fingerprint(again2)
    assert _fingerprint(again1) == _fingerprint(first)


def _fingerprint(result: MatchResult) -> tuple:
    """把结果压成可比较的元组（用于可复现断言）。"""
    return (
        result.wins,
        result.draws,
        result.losses,
        result.games,
        result.wins_as_black,
        result.games_as_black,
        result.wins_as_white,
        result.games_as_white,
        tuple(result.lengths),
        result.a_name,
        result.b_name,
    )


# --------------------------------------------------------------------------- #
# 置信区间
# --------------------------------------------------------------------------- #
def test_score_ci_endpoints_and_center() -> None:
    """0 胜 / 全胜 / 半胜的 95% 区间必须落在 [0, 1] 内且形状合理。"""
    lo0, hi0 = _make_result(0, 0, 40).score_ci
    assert 0.0 <= lo0 <= hi0 <= 1.0
    assert lo0 == pytest.approx(0.0, abs=1e-12)
    assert hi0 < 0.15  # 0/40 的上界约 0.088

    lo1, hi1 = _make_result(40, 0, 40).score_ci
    assert 0.0 <= lo1 <= hi1 <= 1.0
    assert hi1 == pytest.approx(1.0, abs=1e-12)
    assert 0.8 < lo1 < 1.0

    lo2, hi2 = _make_result(20, 0, 40).score_ci
    assert 0.0 <= lo2 <= 0.5 <= hi2 <= 1.0
    assert 0.2 < hi2 - lo2 < 0.4  # 40 局时区间宽度约 0.30


def test_win_rate_counts_draw_as_half() -> None:
    """和棋按半分计入得分率。"""
    assert _make_result(0, 40, 40).win_rate == pytest.approx(0.5)
    assert _make_result(10, 0, 40).win_rate == pytest.approx(0.25)


def test_empty_match_result_does_not_crash() -> None:
    """0 局时 win_rate/score_ci/summary 都应当安全（无信息区间）。"""
    empty = MatchResult(0, 0, 0, 0, 0, 0, 0, 0, [])
    assert empty.win_rate == 0.0
    assert empty.score_ci == (0.0, 1.0)
    assert "\n" not in empty.summary()


# --------------------------------------------------------------------------- #
# 非法动作检测
# --------------------------------------------------------------------------- #
class _ConstantAgent:
    """永远返回同一个格子的假 Agent（用于验证非法动作检测）。"""

    name = "always-zero"

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        return 0


class _OutOfRangeAgent:
    """永远返回越界动作的假 Agent。"""

    name = "out-of-range"

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        return 10**6


def test_illegal_occupied_move_raises_assertion_error() -> None:
    """重复占用同一格子 -> AssertionError，且信息点名该 Agent。"""
    with pytest.raises(AssertionError, match="always-zero"):
        play_match(_ConstantAgent(), RandomAgent(seed=5), games=2, seed=0, **_FAST)


def test_out_of_range_move_raises_assertion_error() -> None:
    """越界动作 -> AssertionError，且信息点名该 Agent。"""
    with pytest.raises(AssertionError, match="out-of-range"):
        play_match(_OutOfRangeAgent(), RandomAgent(seed=5), games=1, seed=0, **_FAST)


# --------------------------------------------------------------------------- #
# 截断与渲染选项
# --------------------------------------------------------------------------- #
def test_max_moves_truncates_as_draw() -> None:
    """``max_moves`` 截断时记和棋，手数恰好等于截断值。

    9x9 连四下前 6 手双方各只有 3 子，不可能成四。
    """
    result = play_match(
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        board_size=9,
        win_length=4,
        games=4,
        seed=0,
        max_moves=6,
    )
    assert result.draws == 4
    assert result.lengths == [6, 6, 6, 6]


def test_render_flag_does_not_change_result(capsys: pytest.CaptureFixture[str]) -> None:
    """``render=True`` 只打印文本棋盘，不影响统计结果。"""
    quiet = play_match(RandomAgent(seed=1), RandomAgent(seed=2), games=2, seed=0, **_FAST)
    loud = play_match(
        RandomAgent(seed=1), RandomAgent(seed=2), games=2, seed=0, render=True, **_FAST
    )
    capsys.readouterr()
    assert _fingerprint(quiet) == _fingerprint(loud)


# --------------------------------------------------------------------------- #
# 内置 Agent
# --------------------------------------------------------------------------- #
def test_heuristic_agent_beats_random_over_a_match() -> None:
    """启发式（先赢再堵）应当明显强于随机基线。"""
    result = play_match(
        HeuristicAgent(seed=0),
        RandomAgent(seed=0),
        board_size=9,
        win_length=4,
        games=12,
        seed=0,
    )
    assert result.wins >= result.losses
    assert result.win_rate > 0.5
    assert "\n" not in result.summary()
    assert "heuristic" in result.summary() and "random" in result.summary()


def test_model_agent_returns_only_legal_actions() -> None:
    """ModelAgent（4 平面，argmax）单步动作始终落在掩码内。"""
    model = _tiny_model(in_planes=4)
    agent = ModelAgent(model, name="tiny4", device="cpu", n_planes=4, temperature=0.0)

    env = GomokuEnv(**_FAST)
    obs, _ = env.reset(seed=0)
    for _ in range(6):
        mask = env.legal_mask()
        action = agent.act(env, obs, mask)
        assert 0 <= action < env.n_cells
        assert bool(mask[action])
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break


def test_model_agent_temperature_sampling_is_legal_and_seeded() -> None:
    """``temperature > 0`` 采样出的动作同样合法，且 reseed 后可复现。"""
    agent = ModelAgent(_tiny_model(in_planes=4), name="t", n_planes=4, temperature=1.0)
    env = GomokuEnv(**_FAST)
    obs, _ = env.reset(seed=1)
    mask = env.legal_mask()

    agent.reseed(123)
    first = [agent.act(env, obs, mask) for _ in range(8)]
    agent.reseed(123)
    second = [agent.act(env, obs, mask) for _ in range(8)]
    assert first == second
    assert all(bool(mask[a]) for a in first)


def test_model_agent_rejects_bad_config() -> None:
    """非法 ``n_planes`` / ``temperature`` 立刻报错。"""
    model = _tiny_model(in_planes=4)
    with pytest.raises(ValueError, match="n_planes"):
        ModelAgent(model, name="bad", n_planes=5)
    with pytest.raises(ValueError, match="temperature"):
        ModelAgent(model, name="bad", n_planes=4, temperature=-0.5)


def test_model_agent_vs_random_match_runs() -> None:
    """ModelAgent 跑完整 2 局不报错，且手数统计与局数一致。"""
    agent = ModelAgent(_tiny_model(in_planes=4), name="tiny4", n_planes=4, temperature=0.0)
    result = play_match(agent, RandomAgent(seed=1), games=2, seed=1, **_FAST)
    assert result.games == 2
    assert len(result.lengths) == 2
    assert result.wins + result.draws + result.losses == 2


def test_model_agent_six_planes_uses_alphazero_encoder() -> None:
    """6 平面观测走 ``alphazero.encode``（与训练时 ``GameState.observation()`` 同一套编码）。"""
    agent = ModelAgent(_tiny_model(in_planes=6), name="tiny6", n_planes=6)
    result = play_match(agent, RandomAgent(seed=2), games=2, seed=3, **_FAST)
    assert result.games == 2


def test_search_agent_reports_missing_search_module() -> None:
    """``search.py`` 不存在时，构造 SearchAgent 给出清晰报错。"""
    if importlib.util.find_spec("search") is not None:
        pytest.skip("search.py 已存在，跳过缺失报错用例")
    with pytest.raises(RuntimeError, match="search"):
        SearchAgent(_tiny_model(in_planes=6), name="search")


def test_search_agent_plays_when_available() -> None:
    """``search.py`` 存在时，SearchAgent 能完成对局（联调用例）。"""
    if importlib.util.find_spec("search") is None:
        pytest.skip("search.py 尚未提供")
    agent = SearchAgent(_tiny_model(in_planes=6), name="search", depth=1, top_k=4, n_planes=6)
    result = play_match(agent, RandomAgent(seed=2), games=2, seed=1, **_FAST)
    assert result.games == 2
