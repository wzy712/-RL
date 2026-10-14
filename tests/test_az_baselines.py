"""``az_baselines.py`` 的测试：自包含规则对手 + 成对开局协议。

重点钉住三件事：

1. **成对开局协议忠实于规则**：``win_length`` / ``board_size`` 必须真的传到环境里
   ——"用连四规则评估连五网络"是这套代码最容易犯、也最难察觉的错误；
2. **协议自洽**：``score(A,B) + score(B,A) == 1``（同一开局、角色互换，确定性对手下严格成立），
   且胜负分项与对数一致；
3. **规则对手可用**：``heuristic`` / ``tactical`` 都只走合法手，``tactical`` 能直接取胜。

运行::

    uv run pytest tests/test_az_baselines.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from arena import HeuristicAgent, RandomAgent, play_match
from az_baselines import (
    HeuristicOpponent,
    RandomOpponent,
    TacticalAgent,
    TacticalHeuristicOpponent,
    paired_eval,
)
from env import BLACK, EMPTY, WHITE, GomokuEnv

_FAST = dict(board_size=5, win_length=3)


# --------------------------------------------------------------------------- #
# 测试用确定性 Agent
# --------------------------------------------------------------------------- #
class _DeterministicAgent:
    """完全确定性的 Agent：固定取"第一个"或"最后一个"合法点（与随机源无关）。"""

    def __init__(self, name: str, *, last: bool = False) -> None:
        self.name = name
        self.last = last
        self.seen_win_length: list[int] = []

    def reseed(self, seed: int) -> None:  # 与 arena.Agent 协议一致（本 Agent 无随机性）
        del seed

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        self.seen_win_length.append(int(env.win_length))
        legal = np.flatnonzero(mask)
        return int(legal[-1] if self.last else legal[0])


# --------------------------------------------------------------------------- #
# 1. 规则（board_size / win_length）真的传进了对局环境
# --------------------------------------------------------------------------- #
def test_paired_eval_uses_the_requested_rules() -> None:
    """默认 9x9 **连五**；显式传参时必须照传（回归：曾被写成默认连四）。"""
    a, b = _DeterministicAgent("first"), _DeterministicAgent("last")
    paired_eval(a, b, pairs=2, plies=1, seed=0, board_size=7, win_length=5)

    assert a.seen_win_length and set(a.seen_win_length) == {5}
    assert set(b.seen_win_length) == {5}


def test_paired_eval_default_is_gomoku_win_length() -> None:
    """不传规则时默认 9x9 连五（AlphaZero 本体的规则）。"""
    a, b = _DeterministicAgent("first"), _DeterministicAgent("last")
    paired_eval(a, b, pairs=1, plies=0, seed=0)
    assert set(a.seen_win_length) == {5}


def test_win_length_changes_the_game() -> None:
    """连四在连五规则下不是胜利：同一局面下 4 连不能结束对局。"""
    # 黑 (0,0)-(0,3) 已成 4 连；若规则是连四，白早该输了
    env = GomokuEnv(board_size=9, win_length=5)
    env.reset()
    for action in [0, 9, 1, 10, 2, 11, 3]:
        env.step(action)
    assert env.winner == EMPTY
    assert not env._done

    # 换成连四规则，同样的落子序列在第 7 手就结束了
    env4 = GomokuEnv(board_size=9, win_length=4)
    env4.reset()
    for action in [0, 9, 1, 10, 2, 11, 3]:
        _, _, terminated, _, info = env4.step(action)
        if terminated:
            break
    assert env4.winner == BLACK
    assert info["winner"] == BLACK


# --------------------------------------------------------------------------- #
# 2. 协议自洽 / 可复现
# --------------------------------------------------------------------------- #
def test_paired_eval_is_complementary_under_role_swap() -> None:
    """同一开局、角色互换：确定性对手下 ``score(A,B) + score(B,A) == 1``。"""
    a = _DeterministicAgent("first")
    b = _DeterministicAgent("last")
    forward = paired_eval(a, b, pairs=6, plies=2, seed=42, **_FAST)
    backward = paired_eval(b, a, pairs=6, plies=2, seed=42, **_FAST)

    assert forward["score_rate"] + backward["score_rate"] == pytest.approx(1.0)
    assert forward["pairs_won"] == backward["pairs_lost"]
    assert forward["pairs_drawn"] == backward["pairs_drawn"]


def test_paired_eval_bookkeeping_is_consistent() -> None:
    """每对恰好 2 局；胜负分项之和 == 对数；CI 落在 [0,1] 且包含均值。"""
    result = paired_eval(
        HeuristicAgent(seed=1), RandomAgent(seed=2), pairs=6, plies=1, seed=5, **_FAST
    )
    lo, hi = result["ci95"]
    assert result["pairs_won"] + result["pairs_drawn"] + result["pairs_lost"] == result["pairs"]
    assert 0.0 <= lo <= result["score_rate"] <= hi <= 1.0
    assert result["avg_length"] > 0.0


def test_paired_eval_is_reproducible() -> None:
    """同 seed 两次调用逐字段一致（含复用同一对 Agent 对象）。"""
    first = paired_eval(RandomAgent(seed=7), RandomAgent(seed=7), pairs=4, plies=2, seed=9, **_FAST)
    second = paired_eval(RandomAgent(seed=7), RandomAgent(seed=7), pairs=4, plies=2, seed=9, **_FAST)
    assert first == second
    assert 0.0 <= first["score_rate"] <= 1.0


# --------------------------------------------------------------------------- #
# 3. 规则对手
# --------------------------------------------------------------------------- #
def test_heuristic_opponent_beats_random() -> None:
    """启发式（先赢再堵）在成对开局下明显强于随机。"""
    result = paired_eval(
        HeuristicAgent(seed=0),
        RandomAgent(seed=0),
        pairs=10,
        plies=1,
        seed=3,
        board_size=9,
        win_length=4,
    )
    assert result["score_rate"] > 0.5


def test_tactical_agent_takes_the_immediate_win() -> None:
    """黑 (0,0)-(0,3) 四连、轮到黑走：``tactical`` 必须直接连五，而不是去防守。"""
    env = GomokuEnv(board_size=9, win_length=5)
    env.reset()
    # 黑 4 手：0/1/2/3 = (0,0)-(0,3)；白 4 手散在远处（不构成任何威胁）
    for action in [0, 40, 1, 51, 2, 62, 3, 73]:
        env.step(action)
    assert env.current_player == BLACK
    assert env.winner == EMPTY

    agent = TacticalHeuristicOpponent(np.random.default_rng(0))
    assert agent._winning_squares(env, BLACK) == {4}  # 唯一胜点是 (0,4)
    assert agent._winning_squares(env, WHITE) == set()
    action = agent.select(env, env._observation(), env.legal_mask())
    assert action == 4
    env.step(action)
    assert env.winner == BLACK


def test_tactical_agent_blocks_a_double_threat() -> None:
    """黑有"活四"（两个立刻获胜点）时，白必须至少堵掉一个，否则下一手就输。"""
    env = GomokuEnv(board_size=9, win_length=5)
    env.reset()
    # 黑 (4,3)(4,4)(4,5)(4,6) 活四；白 (0,0)(5,7)(8,2) 散子
    for action in [39, 0, 40, 52, 41, 74, 42]:
        env.step(action)
    assert env.current_player == WHITE
    assert env.winner == EMPTY

    agent = TacticalHeuristicOpponent(np.random.default_rng(0))
    threats = agent._winning_squares(env, BLACK)
    assert threats == {38, 43}  # 活四：两端都能连五

    action = agent.select(env, env._observation(), env.legal_mask())
    assert action in threats  # 必须占掉一端，把对手的立刻获胜点压到 1


@pytest.mark.parametrize(
    "opponent",
    [HeuristicOpponent(np.random.default_rng(0)), TacticalHeuristicOpponent(np.random.default_rng(0))],
    ids=["heuristic", "tactical"],
)
def test_rule_opponents_only_return_legal_moves(opponent) -> None:
    """随机对局里逐步检查：规则对手只返回合法手，并且能下完整局。"""
    for _ in range(3):
        env = GomokuEnv(**_FAST)
        env.reset()
        while not env._done:
            mask = env.legal_mask()
            if not mask.any():
                break
            before = env.board.copy()
            action = opponent.select(env, env._observation(), mask)
            assert bool(mask[action]), f"{opponent.name} 返回非法动作 {action}"
            env.step(action)
            changed = int((env.board != before).sum())
            assert changed == 1


def test_opponents_accept_the_arena_extra_argument() -> None:
    """``arena.HeuristicAgent`` 会多传一个先验分布，对手必须能接受（签名兼容）。"""
    env = GomokuEnv(**_FAST)
    env.reset()
    mask = env.legal_mask()
    for opponent in (RandomOpponent(np.random.default_rng(0)), HeuristicOpponent(np.random.default_rng(0))):
        action = opponent.select(env, env._observation(), mask, None)
        assert bool(mask[action])


def test_tactical_agent_survives_a_full_match() -> None:
    """``TacticalAgent`` 能走完 ``arena.play_match``（Agent 协议联调）。"""
    result = play_match(TacticalAgent(seed=0), RandomAgent(seed=1), games=4, seed=2, **_FAST)
    assert result.games == 4
    assert len(result.lengths) == 4
    assert result.wins >= result.losses
