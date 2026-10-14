"""对局竞技场：双 Agent head-to-head 评估与胜负统计。

用途
----
给定两个 :class:`Agent`，让它们轮换黑白对弈 ``games`` 局，产出可复现的
:class:`MatchResult`（胜负、和棋、黑白分项、每局手数、得分率与 95% 置信区间）。
训练端可以用它回答"新策略是否真的比旧策略 / 启发式 / 随机强"。

设计要点
--------
* **可复现**：``play_match`` 只依赖 ``seed`` 与两个 Agent。若 Agent 实现了
  ``reseed(seed)``（内置 Agent 都实现了），开赛前会用派生种子重置其随机源，
  因此同一对 Agent 重复调用、或重建后调用，结果逐字段一致。
* **非法动作零容忍**：Agent 返回越界或已占用格子时立刻
  ``AssertionError``，并在信息里点名是哪个 Agent——绝不静默改手，
  否则评估数据会被悄悄污染。
* **轮换黑白**：``games`` 为偶数时黑白各半，避免先手优势污染比较结论。
* **观测契约**：``n_planes=4`` 直接用 ``env`` 的原始观测；
  ``n_planes=6`` 走 ``learning.encode_observation(env)``（延迟 import，
  4 平面路径完全不碰 ``learning``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from env import BLACK, EMPTY, WHITE, GomokuEnv

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
#: 95% 正态分位数（Wilson 区间用）
_Z_95: float = 1.959963984540054

#: 派生种子时用于区分"环境 / A / B"的标签
_SEED_ENV: int = 0
_SEED_A: int = 1
_SEED_B: int = 2


# --------------------------------------------------------------------------- #
# Agent 接口
# --------------------------------------------------------------------------- #
class Agent(Protocol):
    """对弈参与者：给定当前 ``env`` / 观测 / 合法掩码，返回一个动作索引。"""

    name: str

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int: ...


# --------------------------------------------------------------------------- #
# 统计结果
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    """一次 head-to-head 对局的结果汇总（全部以 A 方为视角）。

    Parameters
    ----------
    wins / draws / losses:
        A 方（第一个 Agent）的胜 / 和 / 负局数，满足三者和等于 ``games``。
    games:
        总局数。
    wins_as_black / games_as_black:
        A 方执黑时的胜局数与局数。
    wins_as_white / games_as_white:
        A 方执白时的胜局数与局数。
    lengths:
        每局的手数（ply），长度等于 ``games``。
    a_name / b_name:
        便于 :meth:`summary` 输出的双方名称（额外字段，带默认值）。
    """

    wins: int
    draws: int
    losses: int
    games: int
    wins_as_black: int
    games_as_black: int
    wins_as_white: int
    games_as_white: int
    lengths: list[int]
    a_name: str = "A"
    b_name: str = "B"

    @property
    def win_rate(self) -> float:
        """A 的得分率：``(wins + 0.5 * draws) / games``（和棋算半分）。"""
        if self.games <= 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def score_ci(self) -> tuple[float, float]:
        """得分率的 95% Wilson 置信区间。

        Wilson 区间在 p=0 / p=1 端点不会越界，比正态近似更适合小样本评估。
        ``games == 0`` 时返回无信息区间 ``(0.0, 1.0)``。
        """
        n = self.games
        if n <= 0:
            return (0.0, 1.0)
        p = self.win_rate
        z2 = _Z_95 * _Z_95
        denom = 1.0 + z2 / n
        center = (p + z2 / (2.0 * n)) / denom
        half = _Z_95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
        lo = max(0.0, min(1.0, center - half))
        hi = max(0.0, min(1.0, center + half))
        return (lo, hi)

    @property
    def average_length(self) -> float:
        """平均每局手数（无对局时为 0）。"""
        if not self.lengths:
            return 0.0
        return float(sum(self.lengths)) / len(self.lengths)

    def summary(self) -> str:
        """单行可读文本，适合直接打进终端或实验日志。"""
        lo, hi = self.score_ci
        return (
            f"{self.a_name} vs {self.b_name} | {self.games} 局: "
            f"{self.wins}胜 {self.draws}和 {self.losses}负 | 得分率 {self.win_rate:.3f} "
            f"| 95%CI [{lo:.3f}, {hi:.3f}] "
            f"| 执黑 {self.wins_as_black}/{self.games_as_black} "
            f"| 执白 {self.wins_as_white}/{self.games_as_white} "
            f"| 平均 {self.average_length:.1f} 手"
        )


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _derive_seed(seed: int, *extra: int) -> int:
    """由主种子和若干标签派生一个稳定的 31 位整数种子。"""
    state = np.random.SeedSequence([int(seed), *(int(x) for x in extra)]).generate_state(1)
    return int(state[0] % (2**31 - 1))


def _reseed_agent(agent: Agent, seed: int) -> None:
    """若 Agent 支持 ``reseed`` 则重置其随机源，保证同 seed 可复现。"""
    reseed = getattr(agent, "reseed", None)
    if callable(reseed):
        reseed(int(seed))


def _prepare_observation(env: GomokuEnv, obs: np.ndarray, n_planes: int) -> np.ndarray:
    """按 ``n_planes`` 准备网络输入观测。

    * ``4``：直接用 ``env`` 给出的原始观测（不 import ``learning``）。
    * ``6``：调用 ``learning.encode_observation(env)``（延迟 import）。
    """
    if n_planes == 4:
        return np.asarray(obs, dtype=np.float32)
    if n_planes == 6:
        try:
            from learning import encode_observation  # 主工程：6 平面编码在 learning.py
        except ImportError:  # 本交付：改用 alphazero.py 里等价的 6 平面编码
            from alphazero import encode as encode_observation
        return np.asarray(encode_observation(env), dtype=np.float32)
    raise ValueError(f"n_planes 只能是 4 或 6，收到 {n_planes}")


def _validate_planes(n_planes: int) -> int:
    """校验观测平面数并原样返回。"""
    if n_planes not in (4, 6):
        raise ValueError(f"n_planes 只能是 4 或 6，收到 {n_planes}")
    return int(n_planes)


# --------------------------------------------------------------------------- #
# 对局主循环
# --------------------------------------------------------------------------- #
def play_match(
    a: Agent,
    b: Agent,
    *,
    board_size: int = 9,
    win_length: int = 4,
    games: int = 40,
    seed: int = 0,
    render: bool = False,
    max_moves: int | None = None,
) -> MatchResult:
    """让 ``a`` 与 ``b`` 轮换黑白对弈 ``games`` 局。

    Parameters
    ----------
    a, b:
        参与对弈的两个 Agent；统计一律以 ``a`` 为视角（``wins`` 表示 a 赢）。
    board_size, win_length:
        棋盘边长与连子数，默认 ``9x9`` 连四（训练配置）。
    games:
        对局总数；为偶数时黑白各半，为奇数时 a 多执一次黑。
    seed:
        主随机种子。环境种子按局派生；实现了 ``reseed`` 的 Agent 也会被重置，
        因此同一 ``seed`` 下结果逐字段可复现。
    render:
        ``True`` 时每步打印文本棋盘（用 ``ansi`` 渲染，无显示器也能跑）。
    max_moves:
        手数截断；``None``（默认）表示下满棋盘才算和棋。

    Raises
    ------
    AssertionError
        Agent 返回越界或已占用格子的动作时抛出，信息中包含该 Agent 的 ``name``。
    """
    if games < 0:
        raise ValueError(f"games 不能为负，收到 {games}")
    if max_moves is not None and max_moves < 0:
        raise ValueError(f"max_moves 不能为负，收到 {max_moves}")

    _reseed_agent(a, _derive_seed(seed, _SEED_A))
    _reseed_agent(b, _derive_seed(seed, _SEED_B))

    env = GomokuEnv(
        board_size=board_size,
        win_length=win_length,
        render_mode="ansi" if render else None,
    )

    wins = draws = losses = 0
    wins_as_black = games_as_black = 0
    wins_as_white = games_as_white = 0
    lengths: list[int] = []

    for game_index in range(games):
        obs, _ = env.reset(seed=_derive_seed(seed, _SEED_ENV, game_index))
        a_is_black = game_index % 2 == 0
        if a_is_black:
            games_as_black += 1
        else:
            games_as_white += 1

        steps = 0
        while True:
            if max_moves is not None and steps >= max_moves:
                break  # 截断 -> 记和棋
            mask = env.legal_mask()
            if not mask.any():
                break  # 棋盘已满（env 通常已判和，这里只做兜底）

            current = env.current_player
            agent = a if (current == BLACK) == a_is_black else b
            action = int(agent.act(env, obs, mask))
            if not (0 <= action < env.n_cells) or not bool(mask[action]):
                raise AssertionError(
                    f"Agent {agent.name!r} 返回非法动作 {action}"
                    f"（该动作越界或已被占用；本回合合法点 {int(mask.sum())} 个）"
                )

            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if render:
                env.render()
            if terminated or truncated:
                break

        lengths.append(int(env.move_count))
        winner = int(env.winner)
        a_color = BLACK if a_is_black else WHITE
        if winner == EMPTY:
            draws += 1
        elif winner == a_color:
            wins += 1
            if a_is_black:
                wins_as_black += 1
            else:
                wins_as_white += 1
        else:
            losses += 1

    return MatchResult(
        wins=wins,
        draws=draws,
        losses=losses,
        games=games,
        wins_as_black=wins_as_black,
        games_as_black=games_as_black,
        wins_as_white=wins_as_white,
        games_as_white=games_as_white,
        lengths=lengths,
        a_name=str(getattr(a, "name", "A")),
        b_name=str(getattr(b, "name", "B")),
    )


# --------------------------------------------------------------------------- #
# 内置 Agent
# --------------------------------------------------------------------------- #
class RandomAgent:
    """均匀随机合法落子的基线 Agent。"""

    def __init__(self, name: str = "random", seed: int = 0) -> None:
        self.name = name
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        """重置随机源（``play_match`` 会调用，保证可复现）。"""
        self.rng = np.random.default_rng(int(seed))

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(np.asarray(mask, dtype=bool))
        if legal.size == 0:
            raise ValueError("RandomAgent 收到全 False 的合法掩码")
        return int(self.rng.choice(legal))


class HeuristicAgent:
    """包装 :class:`train.HeuristicOpponent` 的规则型基线（"先赢再堵 + 邻域打分"）。"""

    def __init__(self, name: str = "heuristic", seed: int = 0) -> None:
        try:
            from az_baselines import HeuristicOpponent  # 本交付：自包含规则对手
        except ImportError:
            from train import HeuristicOpponent  # 主工程

        self.name = name
        self.rng = np.random.default_rng(seed)
        # 惰性构造，避免 import arena 时就把 torch 拉起来
        self.opponent = HeuristicOpponent(self.rng)

    def reseed(self, seed: int) -> None:
        """重置随机源（``HeuristicOpponent`` 同分随机用）。"""
        self.rng = np.random.default_rng(int(seed))
        self.opponent.rng = self.rng

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        # HeuristicOpponent.select 不使用 dist 参数（只读 env/mask），传 None 即可
        return int(self.opponent.select(env, obs, np.asarray(mask, dtype=bool), None))


class ModelAgent:
    """包装 :class:`train.ActorCritic` 的策略 Agent。

    ``temperature == 0`` 走 argmax（确定性）；``> 0`` 走温度采样。
    观测平面数由 ``n_planes`` 决定：4 = 原始 env 观测，6 = ``learning`` 威胁平面。
    模型在构造时搬到 ``device`` 并 ``eval()``，之后每步复用，不会重建。
    """

    def __init__(
        self,
        model,
        *,
        name: str,
        device="cpu",
        n_planes: int = 6,
        temperature: float = 0.0,
    ) -> None:
        import torch

        self.n_planes = _validate_planes(n_planes)
        if temperature < 0.0:
            raise ValueError(f"temperature 必须 >= 0，收到 {temperature}")
        self.name = name
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(0)

    def reseed(self, seed: int) -> None:
        """重置温度采样的随机源（argmax 路径本身与随机数无关）。"""
        self.generator.manual_seed(int(seed))

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        import torch

        if env.board_size != self.model.board_size:
            raise ValueError(
                f"ModelAgent {self.name!r} 的网络 board_size={self.model.board_size} "
                f"与环境 board_size={env.board_size} 不一致"
            )
        obs_np = _prepare_observation(env, obs, self.n_planes)
        mask_np = np.asarray(mask, dtype=bool)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask_np, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits, _value = self.model(obs_t, mask_t)
            if self.temperature == 0.0:
                return int(torch.argmax(logits, dim=-1).item())
            probs = torch.softmax(logits / self.temperature, dim=-1).detach().to("cpu")
            return int(torch.multinomial(probs, 1, generator=self.generator).item())


class SearchAgent:
    """包装 ``search.search_move`` 的搜索 Agent（惰性 import ``search.py``）。

    ``search.py`` 尚不存在时，构造阶段就抛出带说明的 ``RuntimeError``；
    调用签名遵循 ``search_move(model, obs, mask, env=env, depth=..., top_k=...,
    temperature=0.0)``，并额外传入其签名中已定义的 ``n_planes`` / ``device``，
    以便 4 平面模型与显式设备也能正确工作。
    """

    def __init__(
        self,
        model,
        *,
        name: str,
        depth: int = 3,
        top_k: int = 8,
        device="cpu",
        n_planes: int = 6,
    ) -> None:
        import torch

        try:
            from search import search_move
        except ImportError as exc:
            raise RuntimeError(
                "SearchAgent 需要 search.py 提供的 search_move(...)；"
                "当前无法 import search，请先实现 search.py"
            ) from exc

        if depth < 1:
            raise ValueError(f"depth 必须 >= 1，收到 {depth}")
        if top_k < 1:
            raise ValueError(f"top_k 必须 >= 1，收到 {top_k}")

        self.n_planes = _validate_planes(n_planes)
        self.name = name
        self.depth = int(depth)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._search_move = search_move

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_np = _prepare_observation(env, obs, self.n_planes)
        return int(
            self._search_move(
                self.model,
                obs_np,
                np.asarray(mask, dtype=bool),
                env=env,
                depth=self.depth,
                top_k=self.top_k,
                temperature=0.0,
                n_planes=self.n_planes,
                device=str(self.device),
            )
        )


__all__ = [
    "Agent",
    "HeuristicAgent",
    "MatchResult",
    "ModelAgent",
    "RandomAgent",
    "SearchAgent",
    "play_match",
]
