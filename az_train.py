"""AlphaZero 式训练入口：MCTS 自我对弈 + 独立价值网络（9x9 连五）。

算法要点（与 ``alphazero.py`` 里的 MCTS 实现对应）：

* **自我对弈**：每步用 MCTS（PUCT + 根节点 Dirichlet 噪声）搜索，访问次数分布作为策略目标；
  对局结束后把**终局结果**按 negamax 视角回填成每个局面的价值目标 ``z ∈ {-1, 0, +1}``。
* **网络训练**：``L = (z - v)² - πᵀ log p + c‖θ‖²``，策略头与价值头**各自独立主干**
  （``--shared-trunk`` 可切回经典共享主干做对照）。
* **温度日程**：前 ``temp_moves`` 手用 ``τ=1``（探索），之后 ``τ→0``（走最优），
  与 AlphaZero 论文一致。
* **评估**：黑白各半 vs 规则对手 / 旧 checkpoint；另有成对开局协议（``az_baselines.paired_eval``）

用法::

    # 冒烟（CPU 也能跑，几十秒）
    uv run python az_train.py --run-name az_smoke --iterations 2 --games-per-iter 4 \
        --sims 16 --leaf-batch 4 --channels 32 --blocks 2 --train-steps 4 --eval-interval 0

    # 正式训练（9x9 连五）
    uv run python az_train.py --run-name az9x9w5 --iterations 40 --games-per-iter 60 --sims 100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from alphazero import MCTS, GameState, PolicyValueNet
from env import BLACK, EMPTY, WHITE, GomokuEnv


# =========================================================================== #
# 超参数
# =========================================================================== #
@dataclass
class AZConfig:
    """全部超参数；命令行参数由本 dataclass 自动生成。"""

    # ---------------- 环境（五子棋规则） ----------------
    board_size: int = field(default=9, metadata={"help": "棋盘边长（本次固定 9x9）"})
    win_length: int = field(default=5, metadata={"help": "连成多少子获胜（五子棋 = 5）"})

    # ---------------- 网络 ----------------
    channels: int = field(default=96, metadata={"help": "主干通道数（策略/价值各自一套）"})
    blocks: int = field(default=5, metadata={"help": "每个主干的残差块数"})
    value_channels: int = field(default=32, metadata={"help": "价值头 1x1 卷积通道数"})
    value_hidden: int = field(default=128, metadata={"help": "价值头隐层宽度"})
    shared_trunk: bool = field(
        default=False,
        metadata={"help": "True=经典 AlphaZero 共享主干；False（默认）=策略/价值各自独立主干"},
    )

    # ---------------- MCTS ----------------
    sims: int = field(default=100, metadata={"help": "每步 MCTS 模拟次数"})
    c_puct: float = field(default=1.5, metadata={"help": "PUCT 探索常数"})
    dirichlet_alpha: float = field(default=0.3, metadata={"help": "根节点 Dirichlet 噪声 α"})
    dirichlet_eps: float = field(default=0.25, metadata={"help": "根节点噪声混合比例 ε"})
    leaf_batch: int = field(default=8, metadata={"help": "每次迭代并行模拟的叶子数（批量前向）"})
    fpu_reduction: float = field(default=0.2, metadata={"help": "未访问子节点的 FPU 惩罚"})
    temp_moves: int = field(default=12, metadata={"help": "前多少手用 τ=1 采样（之后 τ→0）"})
    tactical_targets: bool = field(
        default=True,
        metadata={
            "help": "自我对弈的策略目标用『战术优先』：有立刻获胜点/必须封堵时直接写 one-hot，"
            "否则用 MCTS 访问分布。五子棋上这是打破冷启动的关键——否则网络要把"
            "『下一手连五』这件事从零学起，实测 17 轮后 policy/value 双双卡住、对规则对手胜率仅 5%"
        },
    )
    tactical_selfplay_prob: float = field(
        default=1.0,
        metadata={"help": "遇到战术局面时采用战术手的概率（1.0=总是；<1 时其余概率走 MCTS 采样）"},
    )

    # ---------------- 训练 ----------------
    iterations: int = field(default=40, metadata={"help": "训练迭代轮数（每轮=自我对弈+更新）"})
    games_per_iter: int = field(default=60, metadata={"help": "每轮自我对弈局数"})
    train_steps: int = field(default=400, metadata={"help": "每轮梯度步数"})
    batch_size: int = field(default=256, metadata={"help": "训练 batch 大小（按局面）"})
    lr: float = field(default=2e-3, metadata={"help": "Adam 学习率"})
    weight_decay: float = field(default=1e-4, metadata={"help": "L2 正则系数"})
    buffer_size: int = field(default=120_000, metadata={"help": "经验回放池容量（按局面）"})
    warmup_positions: int = field(default=512, metadata={"help": "回放池里少于这么多局面时不训练"})
    value_loss_weight: float = field(default=1.0, metadata={"help": "价值损失权重"})
    grad_clip: float = field(default=1.0, metadata={"help": "梯度裁剪（0 关闭）"})
    augment_symmetry: bool = field(
        default=True, metadata={"help": "用 8 种对称变换做数据增强（连五是对称的）"}
    )

    # ---------------- 训练工程 ----------------
    device: str = field(default="auto", metadata={"help": "auto | cpu | cuda | mps"})
    seed: int = field(default=20240, metadata={"help": "随机种子"})
    save_dir: str = field(default="checkpoints", metadata={"help": "checkpoint 目录"})
    log_dir: str = field(default="runs", metadata={"help": "日志目录"})
    run_name: str = field(default="az9x9w5", metadata={"help": "实验名"})
    log_interval: int = field(default=1, metadata={"help": "每多少轮打印一次"})
    save_interval: int = field(default=5, metadata={"help": "每多少轮存一次 checkpoint（0 关闭）"})
    eval_interval: int = field(default=5, metadata={"help": "每多少轮评估一次（0 关闭）"})
    eval_games: int = field(default=20, metadata={"help": "每次评估对每个对手的对局数"})
    eval_sims: int = field(default=0, metadata={"help": "评估时的模拟次数（0=与训练相同）"})
    num_threads: int = field(default=0, metadata={"help": "torch CPU 线程数（0 不设置）"})
    resume: str = field(default="", metadata={"help": "从 checkpoint 续训"})

    # ---------------- 派生量 ----------------
    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.save_dir) / self.run_name

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir) / self.run_name

    @property
    def n_cells(self) -> int:
        return self.board_size * self.board_size


# =========================================================================== #
# 设备 / 工具
# =========================================================================== #
def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_net(cfg: AZConfig) -> PolicyValueNet:
    return PolicyValueNet(
        board_size=cfg.board_size,
        in_planes=6,
        channels=cfg.channels,
        blocks=cfg.blocks,
        value_channels=cfg.value_channels,
        value_hidden=cfg.value_hidden,
        shared_trunk=cfg.shared_trunk,
    )


# =========================================================================== #
# 与 env.GomokuEnv 对接（给 arena / paired_eval 用）
# =========================================================================== #
def state_from_env(env: GomokuEnv) -> GameState:
    """把 ``GomokuEnv`` 当前局面搬进 MCTS 用的 :class:`GameState`。"""
    state = GameState(env.board_size, env.win_length)
    state.board = env.board.astype(np.int8).copy()
    state.player = int(env.current_player)
    state.last_action = int(env.last_action)
    state.move_count = int(env.move_count)
    state.winner = int(env.winner)
    return state


class AZAgent:
    """``arena`` / ``paired_eval`` 的 Agent 协议适配器：落子交给 MCTS。"""

    n_planes = 6

    def __init__(
        self,
        net: PolicyValueNet,
        *,
        name: str,
        device: torch.device | str = "cpu",
        sims: int = 100,
        temperature: float = 0.0,
        leaf_batch: int = 8,
        c_puct: float = 1.5,
    ) -> None:
        self.net = net
        self.name = name
        self.device = torch.device(device)
        self.sims = int(sims)
        self.temperature = float(temperature)
        self.leaf_batch = int(leaf_batch)
        self.c_puct = float(c_puct)
        self._rng = np.random.default_rng(0)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        del obs  # 观测由 GameState 自己编码（保证与训练完全一致）
        state = state_from_env(env)
        search = MCTS(
            self.net,
            self.device,
            sims=self.sims,
            c_puct=self.c_puct,
            leaf_batch=self.leaf_batch,
            dirichlet_eps=0.0,  # 评估不加噪声
        )
        root = search.run(state, add_noise=False)
        probs = search.tree_policy_vector(root, env.n_cells, self.temperature)
        legal = np.flatnonzero(mask)
        if probs[legal].sum() <= 0:
            return int(self._rng.choice(legal))
        if self.temperature <= 1e-3:
            return int(legal[int(np.argmax(probs[legal]))])
        weights = probs[legal] / probs[legal].sum()
        return int(self._rng.choice(legal, p=weights))


# =========================================================================== #
# 自我对弈
# =========================================================================== #
class ReplayBuffer:
    """按局面存的环形回放池：``obs / mask / policy / value``。"""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[float] = []

    def __len__(self) -> int:
        return len(self.values)

    def add(self, obs: np.ndarray, mask: np.ndarray, policy: np.ndarray, value: float) -> None:
        self.obs.append(obs.astype(np.float32, copy=False))
        self.masks.append(mask.astype(bool, copy=False))
        self.policies.append(policy.astype(np.float32, copy=False))
        self.values.append(float(value))
        if len(self.values) > self.capacity:
            drop = len(self.values) - self.capacity
            del self.obs[:drop]
            del self.masks[:drop]
            del self.policies[:drop]
            del self.values[:drop]

    def sample(self, batch: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, len(self.values), size=batch)
        return {
            "obs": np.stack([self.obs[i] for i in idx]),
            "masks": np.stack([self.masks[i] for i in idx]),
            "policies": np.stack([self.policies[i] for i in idx]),
            "values": np.array([self.values[i] for i in idx], dtype=np.float32),
        }


def symmetric_maps(board_size: int) -> list[np.ndarray]:
    """8 种对称（4 旋转 × 2 翻转）的**位置置换表**：``index_map[cell] = 变换后的 cell``。

    9x9 连五对旋转/镜像完全对称，所以对局面做这些置换是**严格等价**的数据增强
    （不是近似）。返回 8 个 ``(N*N,)`` 的 int 数组。
    """
    grid = np.arange(board_size * board_size).reshape(board_size, board_size)
    maps = []
    for k in range(4):
        rotated = np.rot90(grid, k)
        for flip in (False, True):
            maps.append((rotated[:, ::-1] if flip else rotated).reshape(-1).copy())
    return maps


def _gather(
    obs: np.ndarray, mask: np.ndarray, policy: np.ndarray, index_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按置换表 ``index_map`` 同时变换 ``obs / mask / policy``。

    ``index_map[g] = 原棋盘上第 g 个位置变换后的位置``；``mask`` 与 ``policy``
    是「按落点索引」的向量，所以同样的置换直接索引即可（两者必须用同一套置换，
    否则策略目标会和观测对不上——这正是下面参考实现要钉住的性质）。
    """
    board_size = obs.shape[-1]
    grid = index_map.reshape(board_size, board_size)
    new_obs = np.ascontiguousarray(obs[:, grid // board_size, grid % board_size])
    return new_obs, mask[index_map], policy[index_map]


def symmetry_variants(
    obs: np.ndarray, mask: np.ndarray, policy: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """8 种对称下的 ``(obs, mask, policy)`` 变体（逐个算的参考实现，供测试对拍）。"""
    return [
        _gather(obs, mask, policy, index_map) for index_map in symmetric_maps(obs.shape[-1])
    ]


def augment_batch(
    obs: np.ndarray,
    masks: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对**整个 batch**做 8 种对称增强（每次随机选一种，逐样本独立）。

    用置换表在 numpy 上一次批量完成，比逐样本 python 循环快一个量级
    （训练里每步都要用，这是热点）。
    """
    board_size = obs.shape[-1]
    index_maps = symmetric_maps(board_size)
    n = len(values)
    choices = rng.integers(0, len(index_maps), size=n)
    obs_out = np.empty_like(obs)
    masks_out = np.empty_like(masks)
    policies_out = np.empty_like(policies)
    for i in range(n):
        o, m, p = _gather(obs[i], masks[i], policies[i], index_maps[choices[i]])
        obs_out[i], masks_out[i], policies_out[i] = o, m, p
    return obs_out, masks_out, policies_out, values


@torch.no_grad()
def self_play_game(
    net: PolicyValueNet,
    cfg: AZConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, int]], int, int]:
    """一局自我对弈。

    返回 ``(records, winner, moves)``，``records`` 里每项是
    ``(obs, mask, policy_target, player)``；价值目标在事后按终局结果回填。
    """
    state = GameState(cfg.board_size, cfg.win_length)
    search = MCTS(
        net,
        device,
        sims=cfg.sims,
        c_puct=cfg.c_puct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
        leaf_batch=cfg.leaf_batch,
        fpu_reduction=cfg.fpu_reduction,
    )
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    root = None

    while not state.terminal():
        move_index = state.move_count
        # 树复用：对手刚落的那一手若在我们的树里，就把它提为新根（省一次重算）
        if root is not None and state.last_action >= 0:
            reused = search.advance(root, state.last_action)
            if reused is not None:
                root = reused
        root = search.run(state, add_noise=True)

        temperature = 1.0 if move_index < cfg.temp_moves else 1e-3
        probs = search.tree_policy_vector(root, cfg.n_cells, temperature)
        legal = np.flatnonzero(state.legal_mask())

        # 战术优先：把"立刻能连五"或"必须封堵"写成 one-hot 目标。
        # 这不是作弊，而是五子棋的领域知识：这些局面下的正确手是确定的，
        # 让网络从第一步就学到"有杀必杀、被叫杀必堵"，而不是靠试错碰运气。
        action = -1
        if cfg.tactical_targets and rng.random() < cfg.tactical_selfplay_prob:
            mine = np.flatnonzero(state.winning_moves())
            if mine.size:
                action = int(mine[0])
                probs = np.zeros(cfg.n_cells, dtype=np.float32)
                probs[action] = 1.0
            else:
                theirs = np.flatnonzero(state.winning_moves(state.player ^ 3))
                if theirs.size:
                    action = int(theirs[0])
                    probs = np.zeros(cfg.n_cells, dtype=np.float32)
                    probs[action] = 1.0

        if action < 0:
            weights = probs[legal]
            if weights.sum() <= 0:
                weights = np.ones(len(legal))
            weights = weights / weights.sum()
            action = int(rng.choice(legal, p=weights))

        records.append((state.observation(), state.legal_mask().copy(), probs.copy(), state.player))
        state.step(action)

    winner = state.winner
    moves = state.move_count
    del search, root
    return records, int(winner), int(moves)


def fill_value_targets(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]], winner: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """把终局结果回填成每个局面的价值目标（当前行棋方视角）。"""
    out = []
    for obs, mask, policy, player in records:
        # negamax 视角：和棋 0，赢 +1，输 -1（value 是"该局面行棋方"的预期结果）
        value = 0.0 if winner == EMPTY else (1.0 if winner == player else -1.0)
        out.append((obs, mask, policy, value))
    return out


# =========================================================================== #
# 训练
# =========================================================================== #
def train_epochs(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    cfg: AZConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    """从回放池采样做 ``train_steps`` 次梯度更新。"""
    net.train()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "value_dev": 0.0}
    steps = 0
    for _ in range(cfg.train_steps):
        batch = buffer.sample(cfg.batch_size, rng)
        if cfg.augment_symmetry:
            o, m, p, _v = augment_batch(
                batch["obs"], batch["masks"], batch["policies"], batch["values"], rng
            )
            batch["obs"], batch["masks"], batch["policies"] = o, m, p

        obs_t = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
        mask_t = torch.as_tensor(batch["masks"], device=device, dtype=torch.bool)
        policy_t = torch.as_tensor(batch["policies"], device=device, dtype=torch.float32)
        value_t = torch.as_tensor(batch["values"], device=device, dtype=torch.float32)

        logits, values = net(obs_t, mask_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(policy_t * log_probs).sum(dim=-1).mean()
        value_loss = ((values - value_t) ** 2).mean()
        loss = policy_loss + cfg.value_loss_weight * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        optimizer.step()

        totals["policy_loss"] += float(policy_loss.item())
        totals["value_loss"] += float(value_loss.item())
        totals["total_loss"] += float(loss.item())
        totals["value_dev"] += float((values.detach() - value_t).abs().mean().item())
        steps += 1

    net.eval()
    return {k: v / max(1, steps) for k, v in totals.items()}


def evaluate_az(
    net: PolicyValueNet,
    cfg: AZConfig,
    device: torch.device,
    *,
    games: int,
    sims: int,
    rng: np.random.Generator,
    render: bool = False,
) -> dict[str, float]:
    """黑白各半评估：AZ（确定性）vs 内置规则对手 / 旧 checkpoint。"""
    from az_baselines import HeuristicOpponent, RandomOpponent

    env = GomokuEnv(board_size=cfg.board_size, win_length=cfg.win_length, render_mode="human" if render else None)
    agent = AZAgent(net, name="az", device=device, sims=sims, leaf_batch=cfg.leaf_batch)
    opponents = {"heuristic": HeuristicOpponent(rng), "random": RandomOpponent(rng)}
    results: dict[str, float] = {}
    for kind, opponent in opponents.items():
        agent.reseed(int(rng.integers(1 << 30)))
        wins = draws = losses = 0
        lengths: list[int] = []
        for game in range(games):
            az_color = BLACK if game % 2 == 0 else WHITE
            _obs, info = env.reset(options={"starting_player": BLACK})
            while True:
                player = int(info["player"])
                mask = info["legal_mask"]
                if player == az_color:
                    action = agent.act(env, env._observation(), mask)
                else:
                    action = opponent.select(env, env._observation(), mask)
                _obs, _r, terminated, truncated, info = env.step(action)
                if render:
                    env.render()
                if terminated or truncated:
                    winner = int(info["winner"])
                    lengths.append(int(info["move_count"]))
                    if winner == EMPTY:
                        draws += 1
                    elif winner == az_color:
                        wins += 1
                    else:
                        losses += 1
                    break
        total = max(1, games)
        results[f"{kind}_win_rate"] = wins / total
        results[f"{kind}_draw_rate"] = draws / total
        results[f"{kind}_loss_rate"] = losses / total
        results[f"{kind}_avg_length"] = float(np.mean(lengths)) if lengths else 0.0
    env.close()
    return results


def save_checkpoint(
    path: Path,
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    cfg: AZConfig,
    iteration: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "iteration": iteration,
            "best_metric": best_metric,
            "in_planes": 6,
            "kind": "alphazero_independent_value",
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


class JSONLogger:
    """写 ``metrics.jsonl``（每轮一行）。"""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "metrics.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, iteration: int, metrics: dict[str, Any]) -> None:
        record: dict[str, Any] = {"iteration": iteration}
        for key, value in metrics.items():
            if isinstance(value, (bool, np.bool_)):
                record[key] = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                fv = float(value)
                record[key] = fv if math.isfinite(fv) else None
            else:
                record[key] = str(value)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def train(cfg: AZConfig) -> dict[str, Any]:
    """完整 AlphaZero 训练流程。"""
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    net = build_net(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"[setup] device={device} {cfg.board_size}x{cfg.board_size} 连{cfg.win_length} "
        f"通道={cfg.channels} 残差块={cfg.blocks} 共享主干={cfg.shared_trunk} "
        f"参数={n_params:,} sims={cfg.sims} 每轮对局={cfg.games_per_iter} 轮数={cfg.iterations}"
    )

    start_iter = 1
    best_metric = -1.0
    if cfg.resume:
        meta = load_checkpoint(cfg.resume, net, optimizer, device)
        start_iter = int(meta.get("iteration", 0)) + 1
        best_metric = float(meta.get("best_metric", -1.0))
        print(f"[resume] 从 {cfg.resume} 续训，从第 {start_iter} 轮开始")

    rng = np.random.default_rng(cfg.seed + start_iter)
    buffer = ReplayBuffer(cfg.buffer_size)
    logger = JSONLogger(cfg.log_path)
    net.eval()

    t0 = time.time()
    history: list[dict[str, float]] = []

    for iteration in range(start_iter, cfg.iterations + 1):
        net.eval()
        iter_start = time.time()
        wins = {BLACK: 0, WHITE: 0, EMPTY: 0}
        lengths: list[int] = []
        positions = 0
        for _game in range(cfg.games_per_iter):
            records, winner, moves = self_play_game(net, cfg, device, rng)
            for obs, mask, policy, value in fill_value_targets(records, winner):
                buffer.add(obs, mask, policy, value)
            wins[winner if winner in wins else EMPTY] += 1
            lengths.append(moves)
            positions += len(records)

        metrics: dict[str, Any] = {
            "selfplay_positions": positions,
            "buffer_size": len(buffer),
            "sp_black_win_rate": wins[BLACK] / max(1, cfg.games_per_iter),
            "sp_white_win_rate": wins[WHITE] / max(1, cfg.games_per_iter),
            "sp_draw_rate": wins[EMPTY] / max(1, cfg.games_per_iter),
            "sp_avg_length": float(np.mean(lengths)) if lengths else 0.0,
            "selfplay_seconds": time.time() - iter_start,
        }

        if len(buffer) >= cfg.warmup_positions:
            train_metrics = train_epochs(net, optimizer, buffer, cfg, device, rng)
            metrics.update(train_metrics)
        else:
            metrics.update({"policy_loss": float("nan"), "value_loss": float("nan")})

        if cfg.eval_interval > 0 and iteration % cfg.eval_interval == 0:
            eval_metrics = evaluate_az(
                net, cfg, device, games=cfg.eval_games,
                sims=cfg.eval_sims or cfg.sims, rng=rng,
            )
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            metric = 0.6 * eval_metrics["heuristic_win_rate"] + 0.4 * eval_metrics["random_win_rate"]
            metrics["eval_metric"] = metric
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(
                    cfg.checkpoint_dir / "best.pt", net, optimizer, cfg, iteration, best_metric
                )

        logger.log(iteration, metrics)
        history.append({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        if iteration % cfg.log_interval == 0:
            elapsed = time.time() - t0
            line = (
                f"[{iteration:>4}/{cfg.iterations}] "
                f"对局={cfg.games_per_iter} 位置={positions} 池={len(buffer)} "
                f"黑/白/和={metrics['sp_black_win_rate']:.2f}/{metrics['sp_white_win_rate']:.2f}/"
                f"{metrics['sp_draw_rate']:.2f} 手数={metrics['sp_avg_length']:.1f} "
                f"自对弈={metrics['selfplay_seconds']:.0f}s "
                f"pi={metrics.get('policy_loss', float('nan')):.3f} "
                f"v={metrics.get('value_loss', float('nan')):.3f} "
                f"|v-z|={metrics.get('value_dev', float('nan')):.3f}"
            )
            if "eval_metric" in metrics:
                line += (
                    f" | vs 规则 W={metrics['eval_heuristic_win_rate']:.2f} "
                    f"D={metrics['eval_heuristic_draw_rate']:.2f} L={metrics['eval_heuristic_loss_rate']:.2f}"
                )
            line += f" | {elapsed:.0f}s"
            print(line, flush=True)

        if cfg.save_interval > 0 and iteration % cfg.save_interval == 0:
            save_checkpoint(
                cfg.checkpoint_dir / "latest.pt", net, optimizer, cfg, iteration, best_metric
            )

    save_checkpoint(cfg.checkpoint_dir / "final.pt", net, optimizer, cfg, cfg.iterations, best_metric)
    logger.close()
    print(
        f"[done] 参数={n_params:,} best_metric={best_metric:.3f} "
        f"checkpoint={cfg.checkpoint_dir} 耗时={time.time() - t0:.0f}s"
    )
    return {
        "iterations": cfg.iterations,
        "best_metric": best_metric,
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "log": str(logger.path),
        "final_metrics": history[-1] if history else {},
    }


# =========================================================================== #
# 命令行
# =========================================================================== #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaZero 式 MCTS + 独立价值网络（9x9 连五）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for f in fields(AZConfig):
        name = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):
            parser.add_argument(
                name,
                action=argparse.BooleanOptionalAction,
                default=f.default,
                help=f.metadata.get("help", ""),
            )
        else:
            parser.add_argument(
                name, type=type(f.default), default=f.default, help=f.metadata.get("help", "")
            )
    return parser


def config_from_args(argv: list[str] | None = None) -> AZConfig:
    args = build_arg_parser().parse_args(argv)
    known = {f.name for f in fields(AZConfig)}
    return AZConfig(**{k: v for k, v in vars(args).items() if k in known})


def main(argv: list[str] | None = None) -> None:
    train(config_from_args(argv))


if __name__ == "__main__":
    main()
