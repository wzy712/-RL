"""``env.py`` 的规则与接口测试。

运行::

    uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

import env as env_module
from env import BLACK, EMPTY, N_PLANES, WHITE, GomokuEnv, IllegalMoveError


@pytest.fixture()
def env() -> GomokuEnv:
    e = GomokuEnv(board_size=9, win_length=5)
    e.reset(seed=0)
    return e


def play(env: GomokuEnv, moves: list[tuple[int, int]]) -> None:
    """按 (row, col) 顺序落子（自动过滤已结束的情况）。"""
    for r, c in moves:
        if env.get_state()["done"]:
            return
        env.step(env.xy_to_action(r, c))


# --------------------------------------------------------------------------- #
# 基础接口
# --------------------------------------------------------------------------- #
def test_spaces_and_reset(env: GomokuEnv) -> None:
    obs, info = env.reset(seed=1)
    assert obs.shape == (N_PLANES, 9, 9)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 81
    assert info["player"] == BLACK
    assert info["legal_mask"].all()
    assert env.board.sum() == 0


def test_action_coordinate_roundtrip(env: GomokuEnv) -> None:
    for r in range(9):
        for c in range(9):
            assert env.action_to_xy(env.xy_to_action(r, c)) == (r, c)


def test_gym_registration() -> None:
    import gymnasium as gym

    made = gym.make("Gomoku-v0")
    obs, _ = made.reset(seed=0)
    assert obs.shape == (N_PLANES, 9, 9)
    made.close()


def test_gymnasium_api_conformance() -> None:
    """Gymnasium 接口约定（不用 check_env：它假设"所有动作都合法"，
    而棋盘类环境必须靠掩码约束动作，随机采样的动作会落在已有棋子上）。"""
    e = GomokuEnv(board_size=5, win_length=4)
    reset_result = e.reset(seed=3)
    assert isinstance(reset_result, tuple) and len(reset_result) == 2
    obs, info = reset_result
    assert e.observation_space.contains(obs)
    assert isinstance(info, dict)

    step_result = e.step(int(e.action_space.sample()))
    assert len(step_result) == 5
    obs, reward, terminated, truncated, info = step_result
    assert e.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert e.observation_space.contains(e._observation())
    e.close()


def test_gymnasium_check_env_with_lose_mode() -> None:
    """``illegal_move_mode="lose"`` 时任意动作都合法，可过官方 check_env。"""
    from gymnasium.utils.env_checker import check_env

    check_env(GomokuEnv(board_size=5, win_length=5, illegal_move_mode="lose"), skip_render_check=True)


# --------------------------------------------------------------------------- #
# 胜负判定
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "moves,winner",
    [
        # 横向连五（黑 1,3,5,7,9；白在别处）
        ([(4, 0), (0, 0), (4, 1), (0, 1), (4, 2), (0, 2), (4, 3), (0, 3), (4, 4)], BLACK),
        # 纵向连五
        ([(0, 4), (0, 0), (1, 4), (0, 1), (2, 4), (0, 2), (3, 4), (0, 3), (4, 4)], BLACK),
        # 主对角连五
        ([(0, 0), (8, 0), (1, 1), (8, 1), (2, 2), (8, 2), (3, 3), (8, 3), (4, 4)], BLACK),
        # 副对角连五（白获胜）
        ([(0, 0), (4, 4), (0, 1), (3, 5), (1, 0), (2, 6), (1, 1), (1, 7), (2, 0), (0, 8)], WHITE),
    ],
)
def test_five_in_a_row(env: GomokuEnv, moves: list[tuple[int, int]], winner: int) -> None:
    play(env, moves)
    state = env.get_state()
    assert state["done"] is True
    assert state["winner"] == winner
    assert len(state["winning_line"]) == 5
    assert env.check_winner_full_scan() == winner


def test_no_win_at_four(env: GomokuEnv) -> None:
    play(env, [(4, 0), (0, 0), (4, 1), (0, 1), (4, 2), (0, 2), (4, 3), (0, 3)])
    assert env.get_state()["done"] is False
    assert env.winner == EMPTY


def test_gapped_line_is_not_a_win(env: GomokuEnv) -> None:
    # 四子 + 中间被对手隔断，不应判胜
    play(env, [(4, 0), (4, 2), (4, 1), (0, 0), (4, 3), (0, 1), (4, 5), (0, 2), (4, 6), (0, 3)])
    assert env.winner == EMPTY


def test_win_length_three_on_small_board() -> None:
    e = GomokuEnv(board_size=5, win_length=3)
    e.reset(seed=0)
    e.step(e.xy_to_action(2, 0))
    e.step(e.xy_to_action(0, 0))
    e.step(e.xy_to_action(2, 1))
    e.step(e.xy_to_action(0, 1))
    _, reward, terminated, _, info = e.step(e.xy_to_action(2, 2))
    assert terminated and reward == 1.0
    assert info["winner"] == BLACK


def test_is_winning_move_does_not_mutate(env: GomokuEnv) -> None:
    play(env, [(4, 0), (0, 0), (4, 1), (0, 1), (4, 2), (0, 2), (4, 3), (0, 3)])
    before = env.board.copy()
    assert env.is_winning_move(env.xy_to_action(4, 4), BLACK) is True
    # 白在 (4,4) 不构成五连
    assert env.is_winning_move(env.xy_to_action(4, 4), WHITE) is False
    # 已有棋子的位置
    assert env.is_winning_move(env.xy_to_action(4, 0), BLACK) is False
    np.testing.assert_array_equal(before, env.board)


# --------------------------------------------------------------------------- #
# 平局 / 非法动作 / 掩码
# --------------------------------------------------------------------------- #
def test_draw_when_board_full() -> None:
    # 4x4 棋盘、win_length=4，按"终局无任何四连"的顺序铺满
    e = GomokuEnv(board_size=4, win_length=4)
    e.reset(seed=0)
    moves = [
        (0, 0), (0, 2), (0, 1), (0, 3),
        (1, 2), (1, 0), (1, 3), (1, 1),
        (2, 0), (2, 2), (2, 1), (2, 3),
        (3, 2), (3, 0), (3, 3), (3, 1),
    ]
    for r, c in moves:
        _, reward, terminated, _, info = e.step(e.xy_to_action(r, c))
    expected = np.array(
        [
            [BLACK, BLACK, WHITE, WHITE],
            [WHITE, WHITE, BLACK, BLACK],
            [BLACK, BLACK, WHITE, WHITE],
            [WHITE, WHITE, BLACK, BLACK],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(e.board, expected)
    assert e.check_winner_full_scan() == EMPTY
    assert terminated
    assert info["winner"] == EMPTY
    assert info["is_draw"] is True
    assert reward == 0.0
    assert not e.legal_mask().any()


def test_illegal_move_raises(env: GomokuEnv) -> None:
    env.step(env.xy_to_action(4, 4))
    with pytest.raises(IllegalMoveError):
        env.step(env.xy_to_action(4, 4))
    with pytest.raises(IllegalMoveError):
        env.step(999)


def test_step_after_done_raises(env: GomokuEnv) -> None:
    play(env, [(4, 0), (0, 0), (4, 1), (0, 1), (4, 2), (0, 2), (4, 3), (0, 3), (4, 4)])
    with pytest.raises(IllegalMoveError):
        env.step(env.xy_to_action(5, 5))


def test_illegal_move_mode_lose() -> None:
    """违例方判负：这里黑先下 (0,0)，随后轮到白方也下 (0,0) -> 白违例，黑获胜。"""
    e = GomokuEnv(board_size=5, win_length=5, illegal_move_mode="lose")
    e.reset(seed=0)
    e.step(e.xy_to_action(0, 0))  # 黑
    assert e.current_player == WHITE
    _, reward, terminated, _, info = e.step(e.xy_to_action(0, 0))  # 白违例
    assert terminated and reward == 1.0
    assert info["winner"] == BLACK
    assert e.board.sum() == 1  # 违例动作没有改变棋盘


def test_legal_mask_matches_board(env: GomokuEnv) -> None:
    env.step(env.xy_to_action(3, 3))
    mask = env.legal_mask()
    assert mask.shape == (81,)
    assert mask.sum() == 80
    assert mask[env.xy_to_action(3, 3)] == np.False_
    np.testing.assert_array_equal(mask, env.action_masks())
    np.testing.assert_array_equal(np.flatnonzero(mask), env.legal_actions())


# --------------------------------------------------------------------------- #
# 视角与奖励约定（自博弈正确性的核心）
# --------------------------------------------------------------------------- #
def test_observation_is_relative_to_player_to_move(env: GomokuEnv) -> None:
    env.step(env.xy_to_action(4, 4))  # 黑下，轮到白
    obs = env._observation()
    # 轮到白：通道 0（己方）= 白子所在处
    assert obs[0, 4, 4] == 0.0
    assert obs[1, 4, 4] == 1.0
    # 通道 2 = 行棋方指示平面（执黑为 1）
    assert obs[2].min() == 0.0 and obs[2].max() == 0.0
    # 通道 3 = 最后一手
    assert obs[3, 4, 4] == 1.0 and obs[3].sum() == 1.0

    env.step(env.xy_to_action(0, 0))  # 白下，轮到黑
    obs2 = env._observation()
    assert obs2[0, 4, 4] == 1.0  # 黑是己方
    assert obs2[1, 0, 0] == 1.0
    assert obs2[2].min() == 1.0  # 执黑指示平面


def test_reward_is_from_mover_perspective(env: GomokuEnv) -> None:
    """赢棋那一步 reward=+1；此前所有步 reward=0，输方不会收到负奖励。"""
    rewards = []
    for r, c in [(4, 0), (0, 0), (4, 1), (0, 1), (4, 2), (0, 2), (4, 3), (0, 3), (4, 4)]:
        _, reward, terminated, _, _ = env.step(env.xy_to_action(r, c))
        rewards.append(reward)
        if terminated:
            break
    assert rewards[:-1] == [0.0] * 8
    assert rewards[-1] == 1.0
    assert env.winner == BLACK


def test_first_move_last_action_channel(env: GomokuEnv) -> None:
    obs, _ = env.reset(seed=0)
    assert obs[3].sum() == 0.0  # 开局没有最后一手


# --------------------------------------------------------------------------- #
# 状态存取 / 克隆 / 渲染
# --------------------------------------------------------------------------- #
def test_snapshot_roundtrip(env: GomokuEnv) -> None:
    play(env, [(4, 4), (0, 0), (4, 5)])
    state = env.get_state()
    obs_before = env._observation()
    clone = env.clone()
    np.testing.assert_array_equal(clone.board, env.board)
    np.testing.assert_array_equal(clone._observation(), obs_before)

    env.step(env.xy_to_action(2, 2))
    env.set_state(state)
    np.testing.assert_array_equal(env.board, clone.board)
    np.testing.assert_array_equal(env._observation(), obs_before)


def test_clone_is_independent(env: GomokuEnv) -> None:
    clone = env.clone()
    clone.step(clone.xy_to_action(4, 4))
    assert env.board.sum() == 0
    assert clone.board.sum() == 1


def test_reset_with_state_continues_game(env: GomokuEnv) -> None:
    play(env, [(4, 0), (0, 0), (4, 1), (0, 1), (4, 2), (0, 2), (4, 3), (0, 3)])
    state = env.get_state()
    env.reset(options={"state": state})
    assert env.move_count == 8
    _, reward, terminated, _, info = env.step(env.xy_to_action(4, 4))
    assert terminated and reward == 1.0 and info["winner"] == BLACK


def test_starting_player_option(env: GomokuEnv) -> None:
    _, info = env.reset(options={"starting_player": WHITE})
    assert info["player"] == WHITE
    obs = env._observation()
    assert obs[1].sum() == 0.0  # 己方（白）无子


def test_ansi_render(env: GomokuEnv, capsys: pytest.CaptureFixture[str]) -> None:
    e = GomokuEnv(board_size=5, win_length=5, render_mode="ansi")
    e.reset(seed=0)
    e.step(e.xy_to_action(0, 0))
    e.render()
    out = capsys.readouterr().out
    assert "X" in out
    assert "轮到" in out
    e.close()


def test_board_str_symbols(env: GomokuEnv) -> None:
    env.step(env.xy_to_action(0, 0))
    env.step(env.xy_to_action(1, 1))
    text = env.board_str()
    assert "X" in text and "O" in text and "." in text


@pytest.mark.parametrize("render_mode", ["ansi", "rgb_array"])
def test_render_modes_headless(render_mode: str) -> None:
    pytest.importorskip("pygame")
    e = GomokuEnv(board_size=5, win_length=5, render_mode=render_mode)
    e.reset(seed=0)
    e.step(e.xy_to_action(2, 2))
    result = e.render()
    if render_mode == "rgb_array":
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3 and result.shape[2] == 3
        assert result.dtype == np.uint8
    e.close()


def test_module_constants_documented() -> None:
    assert (EMPTY, BLACK, WHITE) == (0, 1, 2)
    assert N_PLANES == 4
    assert set(env_module.GomokuEnv.metadata["render_modes"]) == {"human", "rgb_array", "ansi"}
