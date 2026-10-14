"""``alphazero.py`` / ``az_train.py`` 的测试：9x9 连五 + AlphaZero 式 MCTS + 独立价值网络。

重点钉住四件事：

1. **战术平面正确**（与 ``env.is_winning_move`` 全盘对拍）——它是网络唯一能直接看到威胁的输入；
2. **GameState 与 ``env.GomokuEnv`` 行为一致**（随机对局逐步对拍）——MCTS 内部用的是前者，
   一旦两者语义分叉，搜索出来的棋就和环境里的棋不是一回事；
3. **MCTS 不会漏必胜/漏封堵**——这是 AlphaZero 在五子棋上最容易翻车的地方
   （纯 PUCT 会因为先验太小而漏掉必胜手，本实现加了显式战术处理）；
4. **网络是"策略/价值各自独立主干"**（本次需求），且共享主干选项也能跑。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero import (
    MCTS,
    N_PLANES,
    GameState,
    PolicyValueNet,
    encode,
    immediate_win_mask,
)
from az_train import (
    AZAgent,
    AZConfig,
    ReplayBuffer,
    augment_batch,
    build_net,
    fill_value_targets,
    self_play_game,
    state_from_env,
    symmetric_maps,
    symmetry_variants,
)
from env import BLACK, EMPTY, WHITE, GomokuEnv


# --------------------------------------------------------------------------- #
# 1. 观测 / 战术平面
# --------------------------------------------------------------------------- #
def test_immediate_win_mask_matches_env_on_random_boards():
    """与 ``env.is_winning_move`` 全盘对拍（9x9 连五）。"""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        env = GomokuEnv(board_size=9, win_length=5)
        env.reset()
        for _ in range(int(rng.integers(0, 22))):
            if env._done:
                break
            env.step(int(rng.choice(np.flatnonzero(env.legal_mask()))))
        for player in (BLACK, WHITE):
            got = immediate_win_mask(env.board, player, 5)
            expected = np.array(
                [[env.is_winning_move(r * 9 + c, player) for c in range(9)] for r in range(9)]
            )
            assert np.array_equal(got, expected), f"战术平面不一致:\n{env.board_str()}"
            checked += 1
    assert checked == 120


def test_encode_is_canonical_and_shaped():
    env = GomokuEnv(board_size=9, win_length=5)
    obs, _ = env.reset()
    encoded = encode(env)
    assert encoded.shape == (N_PLANES, 9, 9)
    assert encoded.dtype == np.float32
    assert np.array_equal(encoded[:4], obs)  # 前 4 平面必须与 env 原生观测一致
    # 平面 4/5 就是两个战术平面
    assert np.array_equal(encoded[4] > 0, immediate_win_mask(env.board, BLACK, 5))
    assert np.array_equal(encoded[5] > 0, immediate_win_mask(env.board, WHITE, 5))


# --------------------------------------------------------------------------- #
# 2. GameState 与 env 一致性
# --------------------------------------------------------------------------- #
def test_gamestate_matches_env_on_random_games():
    rng = np.random.default_rng(1)
    for _ in range(12):
        env = GomokuEnv(board_size=9, win_length=5)
        env.reset()
        state = GameState(9, 5)
        while not env._done:
            assert np.array_equal(env.board, state.board)
            assert env.current_player == state.player
            assert env.move_count == state.move_count
            assert np.array_equal(env.legal_mask(), state.legal_mask())
            assert state.terminal() == bool(env._done)
            action = int(rng.choice(np.flatnonzero(env.legal_mask())))
            env.step(action)
            state.step(action)
        assert state.winner == env.winner


def test_gamestate_detects_win():
    state = GameState(9, 5)
    for action in [0, 40, 1, 41, 2, 42, 3, 43, 4]:  # 黑连成 (0,0..4)
        state.step(action)
    assert state.winner == BLACK
    assert state.terminal()


# --------------------------------------------------------------------------- #
# 3. 网络
# --------------------------------------------------------------------------- #
def _tiny_net(shared: bool = False) -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet(board_size=9, channels=16, blocks=1, shared_trunk=shared)


@pytest.mark.parametrize("shared", [False, True])
def test_net_forward_shapes_and_masking(shared: bool):
    net = _tiny_net(shared)
    env = GomokuEnv(board_size=9, win_length=5)
    obs, info = env.reset()
    obs_t = torch.as_tensor(encode(env)).unsqueeze(0)
    mask_t = torch.as_tensor(info["legal_mask"]).unsqueeze(0)
    logits, value = net(obs_t, mask_t)
    assert logits.shape == (1, 81)
    assert value.shape == (1,)
    assert -1.0 <= float(value.item()) <= 1.0
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)[0]
        illegal = ~torch.as_tensor(info["legal_mask"])
        illegal_mass = float(probs[illegal].sum())
        total = float(probs.sum())
    assert illegal_mass < 1e-6
    assert abs(total - 1.0) < 1e-5


def test_independent_trunk_has_no_shared_module():
    """需求：策略头与价值头**各自独立主干**（默认 ``shared_trunk=False``）。"""
    net = _tiny_net(shared=False)
    assert net.trunk is None
    assert net.policy_tower is not None and net.value_tower is not None
    # 两套主干参数互不相同
    p_params = {id(p) for p in net.policy_tower.parameters()}
    v_params = {id(p) for p in net.value_tower.parameters()}
    assert not (p_params & v_params)

    shared = _tiny_net(shared=True)
    assert shared.trunk is not None
    assert shared.policy_tower is None and shared.value_tower is None


def test_net_predict_batch_matches_single_forward():
    net = _tiny_net()
    net.eval()
    env = GomokuEnv(board_size=9, win_length=5)
    obs, info = env.reset()
    obs_batch = np.stack([encode(env), encode(env)])
    mask_batch = np.stack([info["legal_mask"], info["legal_mask"]])
    probs, values = net.predict_batch(obs_batch, mask_batch, torch.device("cpu"))
    assert probs.shape == (2, 81)
    assert values.shape == (2,)
    assert np.allclose(probs[0], probs[1])
    assert np.allclose(values[0], values[1])


# --------------------------------------------------------------------------- #
# 4. MCTS 战术正确性
# --------------------------------------------------------------------------- #
def _four_in_a_row_state() -> GameState:
    """黑 (0,0)(0,1)(0,2)(0,3) 已成四，轮到黑 -> 唯一胜点 (0,4)=4。"""
    state = GameState(9, 5)
    for action in [0, 40, 1, 41, 2, 42, 3, 43]:
        state.step(action)
    assert state.player == BLACK
    assert np.flatnonzero(state.winning_moves(BLACK)).tolist() == [4]
    return state


def _must_block_state() -> GameState:
    """白 (4,0..3) 已成四，轮到黑 -> 必须堵 (4,4)=40。"""
    state = GameState(9, 5)
    for action in [0, 36, 9, 37, 18, 38, 27, 39]:
        state.step(action)
    assert state.player == BLACK
    assert np.flatnonzero(state.winning_moves(WHITE)).tolist() == [40]
    return state


def test_mcts_takes_the_win_even_with_untrained_prior():
    """**回归测试**：未训练网络的先验把胜点排到很后，纯 PUCT 会漏掉它。

    实测未训练网络给胜点的先验约 0.002（73 个候选里排不进前 48 次模拟），
    所以实现里加了「有立刻获胜点就把候选裁成胜点」的战术处理。
    """
    net = _tiny_net()
    net.eval()
    state = _four_in_a_row_state()
    search = MCTS(net, torch.device("cpu"), sims=48, leaf_batch=8, dirichlet_eps=0.0)
    root = search.run(state, add_noise=False)
    probs = search.tree_policy_vector(root, 81, 1e-3)
    assert int(np.argmax(probs)) == 4


def test_mcts_blocks_a_four_with_untrained_prior():
    """必须封堵：对手有唯一胜点时，搜索要把预算集中到该点。"""
    net = _tiny_net()
    net.eval()
    state = _must_block_state()
    search = MCTS(net, torch.device("cpu"), sims=48, leaf_batch=8, dirichlet_eps=0.0)
    root = search.run(state, add_noise=False)
    probs = search.tree_policy_vector(root, 81, 1e-3)
    assert int(np.argmax(probs)) == 40


def test_mcts_policy_is_distribution_and_respects_temperature():
    net = _tiny_net()
    net.eval()
    state = GameState(9, 5)
    for action in [40, 0, 41, 1]:
        state.step(action)
    search = MCTS(net, torch.device("cpu"), sims=24, leaf_batch=4, dirichlet_eps=0.0)
    root = search.run(state, add_noise=True)
    hot = search.tree_policy_vector(root, 81, 1.0)
    cold = search.tree_policy_vector(root, 81, 1e-3)
    assert hot.sum() == pytest.approx(1.0)
    assert cold.sum() == pytest.approx(1.0)
    assert np.count_nonzero(cold) == 1  # τ→0 时退化成 one-hot
    assert root.visit > 0
    # 访问次数是 MCTS 的唯一统计来源
    assert sum(child.visit for child in root.children.values()) >= root.visit - 1


def test_mcts_never_proposes_illegal_move():
    net = _tiny_net()
    net.eval()
    rng = np.random.default_rng(3)
    for _ in range(6):
        state = GameState(9, 5)
        for _ in range(int(rng.integers(0, 8))):
            legal = np.flatnonzero(state.legal_mask())
            state.step(int(rng.choice(legal)))
        if state.terminal():
            continue
        search = MCTS(net, torch.device("cpu"), sims=16, leaf_batch=4)
        root = search.run(state, add_noise=True)
        probs = search.tree_policy_vector(root, 81, 1.0)
        legal = state.legal_mask()
        assert probs[~legal].sum() == pytest.approx(0.0)
        assert probs[legal].sum() == pytest.approx(1.0)


def test_mcts_tree_reuse_advance():
    net = _tiny_net()
    net.eval()
    state = GameState(9, 5)
    state.step(40)
    search = MCTS(net, torch.device("cpu"), sims=16, leaf_batch=4)
    root = search.run(state, add_noise=False)
    action = int(np.argmax(search.tree_policy_vector(root, 81, 1e-3)))
    child = search.advance(root, action)
    assert child is not None
    assert child.prior == 1.0
    assert search.advance(root, -1) is None


# --------------------------------------------------------------------------- #
# 5. az_train 组件
# --------------------------------------------------------------------------- #
def test_symmetry_variants_preserve_game_semantics():
    """8 种对称变换必须同时一致地变换 obs / mask / policy。"""
    env = GomokuEnv(board_size=9, win_length=5)
    _obs, info = env.reset()
    state = GameState(9, 5)
    for action in [40, 0, 41, 1]:
        env.step(action)
        state.step(action)
    obs = state.observation()
    mask = state.legal_mask()
    policy = np.zeros(81, dtype=np.float32)
    policy[3] = 0.7
    policy[7] = 0.3

    for variant_obs, variant_mask, variant_policy in symmetry_variants(obs, mask, policy):
        assert variant_obs.shape == obs.shape
        assert variant_mask.shape == mask.shape
        assert variant_policy.sum() == pytest.approx(1.0)
        # 棋盘上的子位置必须跟着一起变
        assert variant_obs[0].sum() == obs[0].sum()
        assert variant_obs[1].sum() == obs[1].sum()
        # 掩码与棋盘保持一致：被占的点不能合法
        occupied = (variant_obs[0] + variant_obs[1]).reshape(-1) > 0
        assert not variant_mask[occupied].any()


def test_replay_buffer_ring_and_sample():
    buffer = ReplayBuffer(capacity=5)
    obs = np.zeros((N_PLANES, 9, 9), dtype=np.float32)
    mask = np.ones(81, dtype=bool)
    policy = np.full(81, 1 / 81, dtype=np.float32)
    for i in range(8):
        buffer.add(obs, mask, policy, float(i))
    assert len(buffer) == 5  # 环形容量
    batch = buffer.sample(4, np.random.default_rng(0))
    assert batch["obs"].shape == (4, N_PLANES, 9, 9)
    assert batch["values"].shape == (4,)


def test_fill_value_targets_is_negamax():
    records = [
        (np.zeros((N_PLANES, 9, 9), dtype=np.float32), np.ones(81, bool), np.full(81, 1 / 81), BLACK),
        (np.zeros((N_PLANES, 9, 9), dtype=np.float32), np.ones(81, bool), np.full(81, 1 / 81), WHITE),
    ]
    filled = fill_value_targets(records, BLACK)
    assert [round(v, 3) for _o, _m, _p, v in filled] == [1.0, -1.0]  # 视角翻转
    filled_draw = fill_value_targets(records, EMPTY)
    assert [v for _o, _m, _p, v in filled_draw] == [0.0, 0.0]


def test_state_from_env_roundtrip():
    env = GomokuEnv(board_size=9, win_length=5)
    env.reset()
    for action in [40, 0, 41]:
        env.step(action)
    state = state_from_env(env)
    assert np.array_equal(state.board, env.board)
    assert state.player == env.current_player
    assert np.array_equal(state.observation(), encode(env))


def test_az_agent_returns_legal_action():
    net = _tiny_net()
    net.eval()
    agent = AZAgent(net, name="az-test", device="cpu", sims=12, leaf_batch=4)
    env = GomokuEnv(board_size=9, win_length=5)
    _obs, info = env.reset()
    action = agent.act(env, info["obs"] if "obs" in info else env._observation(), info["legal_mask"])
    assert bool(info["legal_mask"][action])
    # 同一局面确定性 agent 给出同一动作
    again = agent.act(env, env._observation(), env.legal_mask())
    assert again == action


def test_az_agent_takes_the_win():
    """Agent 层也要保证不漏必胜手。"""
    net = _tiny_net()
    net.eval()
    agent = AZAgent(net, name="az-test", device="cpu", sims=32, leaf_batch=8)
    env = GomokuEnv(board_size=9, win_length=5)
    env.reset()
    for action in [0, 40, 1, 41, 2, 42, 3, 43]:
        env.step(action)
    assert env.current_player == BLACK
    action = agent.act(env, env._observation(), env.legal_mask())
    assert action == 4


def test_build_net_uses_independent_trunks_by_default():
    cfg = AZConfig(channels=16, blocks=1)
    assert cfg.shared_trunk is False
    net = build_net(cfg)
    assert net.trunk is None
    assert net.policy_tower is not None


def test_short_training_run_produces_finite_losses(tmp_path):
    """端到端冒烟：自我对弈 -> 回放池 -> 梯度更新，损失必须有限。"""
    from az_train import train

    cfg = AZConfig(
        iterations=2,
        games_per_iter=2,
        sims=8,
        leaf_batch=2,
        channels=16,
        blocks=1,
        train_steps=2,
        batch_size=8,
        warmup_positions=1,
        eval_interval=0,
        save_interval=0,
        log_interval=1,
        run_name="smoke",
        save_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "runs"),
        device="cpu",
    )
    summary = train(cfg)
    assert summary["iterations"] == 2
    rows = [json_line for json_line in (tmp_path / "runs" / "smoke" / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    import json

    last = json.loads(rows[-1])
    assert np.isfinite(last["policy_loss"])
    assert np.isfinite(last["value_loss"])
    assert last["buffer_size"] > 0
    assert (tmp_path / "ckpt" / "smoke" / "final.pt").exists()


def test_augment_batch_matches_reference_variants():
    """批量对称增强必须与逐样本参考实现给出完全相同的某一种变体。

    训练里用的是 ``augment_batch``（快一个量级），如果它和参考实现不一致，
    数据增强就会把策略目标与观测错配——那是最难 debug 的一类 bug。
    """
    rng = np.random.default_rng(0)
    obs = rng.random((6, 9, 9)).astype(np.float32)
    mask = rng.random(81) > 0.3
    policy = rng.random(81).astype(np.float32)
    policy /= policy.sum()
    batch_obs = np.stack([obs] * 4)
    batch_mask = np.stack([mask] * 4)
    batch_policy = np.stack([policy] * 4)
    values = np.zeros(4, dtype=np.float32)

    out_obs, out_mask, out_policy, out_values = augment_batch(
        batch_obs, batch_mask, batch_policy, values, rng
    )
    reference = symmetry_variants(obs, mask, policy)
    for i in range(4):
        assert any(
            np.array_equal(out_obs[i], r_obs)
            and np.array_equal(out_mask[i], r_mask)
            and np.array_equal(out_policy[i], r_policy)
            for r_obs, r_mask, r_policy in reference
        ), f"第 {i} 个样本的批量增强结果不在参考变体集合里"
    assert np.array_equal(out_values, values)


def test_symmetric_maps_are_permutations():
    for index_map in symmetric_maps(9):
        assert sorted(index_map.tolist()) == list(range(81))


def test_self_play_records_tactical_one_hot_targets():
    """``tactical_targets`` 打开时，战术局面的策略目标必须是 one-hot。

    这是打破五子棋冷启动的关键：让网络从第一步就学到"有杀必杀、被叫杀必堵"。
    未训练网络 + 少量模拟时，纯 MCTS 几乎不可能自己发现这件事。
    """
    torch.manual_seed(0)
    cfg = AZConfig(
        channels=16, blocks=1, sims=8, leaf_batch=2, temp_moves=0,
        tactical_targets=True, board_size=9, win_length=5,
    )
    net = build_net(cfg)
    net.eval()
    rng = np.random.default_rng(0)
    tactical_positions = 0
    for _ in range(4):
        records, _winner, _moves = self_play_game(net, cfg, torch.device("cpu"), rng)
        for obs, mask, policy, _player in records:
            board = np.where(obs[0] > 0, 1, np.where(obs[1] > 0, 2, 0))
            me = 1 if obs[2, 0, 0] > 0.5 else 2
            mine = immediate_win_mask(board, me, 5)
            theirs = immediate_win_mask(board, 3 - me, 5)
            if mine.any() or theirs.any():
                tactical_positions += 1
                assert policy.max() == pytest.approx(1.0), "战术局面必须是 one-hot 目标"
                assert policy.argmax() in (np.flatnonzero(mine).tolist() + np.flatnonzero(theirs).tolist())
                assert mask[policy.argmax()]
    assert tactical_positions > 0, "自我对弈里应当出现过战术局面"


def test_tactical_targets_can_be_disabled():
    torch.manual_seed(0)
    cfg = AZConfig(
        channels=16, blocks=1, sims=8, leaf_batch=2, temp_moves=0,
        tactical_targets=False, board_size=9, win_length=5,
    )
    net = build_net(cfg)
    net.eval()
    records, _w, _m = self_play_game(net, cfg, torch.device("cpu"), np.random.default_rng(0))
    assert records, "至少要有若干局面"
    assert all(policy.shape == (81,) for _o, _m_, policy, _p in records)
