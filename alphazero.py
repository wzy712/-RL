"""AlphaZero 式 MCTS + **独立价值网络**（策略/价值不共享主干）+ 自我对弈训练。

与前面两个实现的关系
--------------------
* ``train.py``：4 平面观测 + PPO + 单对手（旧基线）
* ``learning.py`` / ``learn_train.py``：6 平面观测 + 对手池 + PPO（上一轮的交付）
* **本文件**：换成 AlphaZero 范式——MCTS 搜索产生**访问次数分布**作为策略目标、
  用**对局结果**作为价值目标，网络只负责提供先验与叶子估值；
  并且按需求把价值网络做成**独立主干**（``shared_trunk=False``，默认），
  同时保留 ``shared_trunk=True`` 选项以便对照"共享主干"是否更好。

围棋/五子棋的 AlphaZero 要素（逐条对应实现）
---------------------------------------------
1. **PUCT 选点**：``a* = argmax Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))``
2. **根节点 Dirichlet 噪声**：``P ← (1-ε)P + ε·Dir(α)``，保证自我对弈有探索
3. **访问次数即策略目标**：``π(a) ∝ N(s,a)^{1/τ}``（对局前若干手用 τ=1，之后 τ→0）
4. **价值目标是终局结果**：从**该状态行棋方**视角取 ±1/0（negamax 视角翻转）
5. **网络双头**：``p, v = f_θ(s)``，损失 ``L = (z - v)² - πᵀ log p + c‖θ‖²``
6. **树复用**：把上一步搜索的子树提为新的根，省掉重算
7. **批量叶子评估**：多个叶子节点一次前向（``leaf_batch``），这是 CPU/GPU 上最关键的加速

环境是 ``env.GomokuEnv``：本文件用 **9x9 连五**（``board_size=9, win_length=5``）。
观测在 ``env`` 原生 4 平面上追加两个战术平面（"我能立刻连五的点 / 对手能立刻连五的点"），
即 6 平面——战术平面是局面的确定性函数，不破坏马尔可夫性，但让 4 连/5 连的威胁一眼可见。

用法::

    # 短跑（验证流程；CPU 也能跑）
    uv run python az_train.py --run-name az9x9w5_smoke --iterations 2 --games-per-iter 4 \
        --sims 24 --leaf-batch 4 --channels 32 --blocks 2 --train-steps 4 --eval-interval 0

    # 正式训练
    uv run python az_train.py --run-name az9x9w5 --iterations 40 --games-per-iter 60 --sims 100

命令行参数由 :class:`AZConfig` 字段自动生成，``--help`` 永远和代码一致。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from env import _DIRECTIONS, BLACK, EMPTY, WHITE, GomokuEnv

# =========================================================================== #
# 观测编码（9x9 连五，6 平面）
# =========================================================================== #
N_PLANES: int = 6


def immediate_win_mask(board: np.ndarray, player: int, win_length: int) -> np.ndarray:
    """``player`` 落子后立刻连成 ``win_length`` 的空点，``(N,N)`` bool。

    窗口法：枚举长度 k 的连续窗口，窗口内恰有 k-1 个 ``player`` 子、剩 1 空格 → 该空格是胜点。
    与 ``GomokuEnv.is_winning_move`` 语义一致（自由规则，长连也算胜）。
    """
    n = board.shape[0]
    k = win_length
    empty = board == EMPTY
    if not empty.any():
        return np.zeros((n, n), dtype=bool)
    mine = board == player
    out = np.zeros((n, n), dtype=bool)
    rows = np.arange(n).reshape(n, 1)
    cols = np.arange(n).reshape(1, n)
    for dr, dc in _DIRECTIONS:
        end_r = rows + (k - 1) * dr
        end_c = cols + (k - 1) * dc
        valid = (end_r >= 0) & (end_r < n) & (end_c >= 0) & (end_c < n)
        starts = np.argwhere(valid)
        if starts.size == 0:
            continue
        r0 = starts[:, 0].reshape(-1, 1)
        c0 = starts[:, 1].reshape(-1, 1)
        steps = np.arange(k).reshape(1, k)
        wr = r0 + steps * dr
        wc = c0 + steps * dc
        win_mine = mine[wr, wc].sum(axis=1)
        win_empty = empty[wr, wc]
        deadly = (win_mine == k - 1) & (win_empty.sum(axis=1) == 1)
        if not deadly.any():
            continue
        holes = np.argmax(win_empty[deadly], axis=1)
        idx = np.arange(holes.size)
        out[wr[deadly][idx, holes], wc[deadly][idx, holes]] = True
    return out & empty


def encode(env: GomokuEnv) -> np.ndarray:
    """``(6, N, N)`` float32，永远是当前行棋方视角（canonical）。"""
    n = env.board_size
    obs = np.zeros((N_PLANES, n, n), dtype=np.float32)
    me = env.current_player
    obs[0] = env.board == me
    obs[1] = env.board == (WHITE if me == BLACK else BLACK)
    obs[2] = 1.0 if me == BLACK else 0.0
    if env.last_action >= 0:
        r, c = divmod(env.last_action, n)
        obs[3, r, c] = 1.0
    obs[4] = immediate_win_mask(env.board, me, env.win_length)
    obs[5] = immediate_win_mask(env.board, WHITE if me == BLACK else BLACK, env.win_length)
    return obs


# =========================================================================== #
# 网络：策略头 / 价值头（默认各自独立主干）
# =========================================================================== #
class ResidualBlock(nn.Module):
    """``x + conv(relu(norm(conv(x))))``。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_num_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(channels), channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Tower(nn.Module):
    """``in_planes -> channels`` 的 3x3 卷积 + N 个残差块。"""

    def __init__(self, in_planes: int, channels: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class PolicyValueNet(nn.Module):
    """AlphaZero 双头网络。

    ``shared_trunk=False``（**默认**）：策略头与价值头**各自独立主干**——
    这是本次需求指定的结构，好处是价值目标（对局结果，±1）与策略目标（MCTS 访问分布）
    不会通过共享主干互相干扰；代价是参数量约翻倍。

    ``shared_trunk=True``：经典 AlphaZero 共享主干，仅用于对照实验。
    """

    def __init__(
        self,
        board_size: int = 9,
        in_planes: int = N_PLANES,
        channels: int = 64,
        blocks: int = 4,
        value_channels: int = 32,
        value_hidden: int = 128,
        shared_trunk: bool = False,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.n_cells = board_size * board_size
        self.in_planes = in_planes
        self.shared_trunk = shared_trunk

        if shared_trunk:
            self.trunk = Tower(in_planes, channels, blocks)
            self.policy_tower: nn.Module | None = None
            self.value_tower: nn.Module | None = None
        else:
            self.trunk = None
            self.policy_tower = Tower(in_planes, channels, blocks)
            self.value_tower = Tower(in_planes, channels, blocks)

        # 策略头：1x1 卷积保留空间结构 -> 展平 -> 每格一个 logit
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_norm = nn.GroupNorm(_num_groups(32), 32)
        self.policy_fc = nn.Linear(32 * self.n_cells, self.n_cells)

        # 价值头：1x1 卷积 -> 全局平均池化 -> MLP -> tanh
        self.value_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.value_norm = nn.GroupNorm(_num_groups(value_channels), value_channels)
        self.value_fc1 = nn.Linear(value_channels, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.apply(_init_weights)

    # ------------------------------------------------------------------ #
    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_trunk:
            trunk = self.trunk(x)
            return trunk, trunk
        assert self.policy_tower is not None and self.value_tower is not None
        return self.policy_tower(x), self.value_tower(x)

    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(logits, value)``；``legal_mask`` 为 True 处保留，False 处置 ``-1e8``。

        约定：``legal_mask`` 的形状是 "True = 合法"（与 MCTS 内部一致）。
        """
        policy_feat, value_feat = self.features(x)
        logits = self.policy_fc(self.relu(self.policy_norm(self.policy_conv(policy_feat))).flatten(1))
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), -1e8)

        pooled = self.value_conv(value_feat).mean(dim=(2, 3))
        value = torch.tanh(self.value_fc2(self.relu(self.value_fc1(self.relu(self.value_norm(pooled))))))
        return logits, value.squeeze(-1)

    @torch.no_grad()
    def predict_batch(
        self, obs: np.ndarray, masks: np.ndarray, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量前向：返回 ``(probs, values)``（numpy，probs 已 softmax 且非法点置 0）。"""
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        mask_t = torch.as_tensor(masks, device=device, dtype=torch.bool)
        logits, values = self.forward(obs_t, mask_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, values.cpu().numpy()


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


# =========================================================================== #
# 游戏状态（MCTS 内部用的轻量实现，避免每步重建 Gym 环境）
# =========================================================================== #
class GameState:
    """9x9 连五的状态：``board`` + 行棋方 + 最后手。

    只做 MCTS 需要的事：合法手、落子（返回终局标记）、立刻获胜点、观测编码。
    全部是 numpy 操作，比 ``GomokuEnv.clone()/step()`` 快很多。
    """

    __slots__ = ("board", "player", "last_action", "move_count", "win_length", "n", "winner")

    def __init__(self, n: int = 9, win_length: int = 5) -> None:
        self.n = n
        self.win_length = win_length
        self.board = np.zeros((n, n), dtype=np.int8)
        self.player = BLACK
        self.last_action = -1
        self.move_count = 0
        self.winner = EMPTY

    def copy(self) -> GameState:
        other = GameState(self.n, self.win_length)
        other.board = self.board.copy()
        other.player = self.player
        other.last_action = self.last_action
        other.move_count = self.move_count
        other.winner = self.winner
        return other

    # ------------------------------------------------------------------ #
    @property
    def n_cells(self) -> int:
        return self.n * self.n

    def legal_mask(self) -> np.ndarray:
        if self.winner != EMPTY:
            return np.zeros(self.n_cells, dtype=bool)
        return (self.board.reshape(-1) == EMPTY).copy()

    def terminal(self) -> bool:
        return self.winner != EMPTY or self.move_count >= self.n_cells

    def step(self, action: int) -> None:
        """落子并更新胜负（增量检查最后一手所在的 4 个方向）。"""
        row, col = divmod(int(action), self.n)
        if self.board[row, col] != EMPTY:
            raise ValueError(f"非法落子 {action}")
        mover = self.player
        self.board[row, col] = mover
        self.last_action = int(action)
        self.move_count += 1
        k = self.win_length
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.n and 0 <= c < self.n and self.board[r, c] == mover:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= k:
                self.winner = mover
                return
        self.player = WHITE if mover == BLACK else BLACK

    def winning_moves(self, player: int | None = None) -> np.ndarray:
        who = self.player if player is None else player
        return immediate_win_mask(self.board, who, self.win_length).reshape(-1)

    def observation(self) -> np.ndarray:
        n = self.n
        obs = np.zeros((N_PLANES, n, n), dtype=np.float32)
        me = self.player
        obs[0] = self.board == me
        obs[1] = self.board == (WHITE if me == BLACK else BLACK)
        obs[2] = 1.0 if me == BLACK else 0.0
        if self.last_action >= 0:
            r, c = divmod(self.last_action, n)
            obs[3, r, c] = 1.0
        obs[4] = immediate_win_mask(self.board, me, self.win_length)
        obs[5] = immediate_win_mask(self.board, WHITE if me == BLACK else BLACK, self.win_length)
        return obs

    def key(self) -> tuple[bytes, int]:
        """不可变局面 key（棋盘字节 + 行棋方），用于置换表 / 树复用。"""
        return self.board.tobytes(), int(self.player)


# =========================================================================== #
# MCTS（PUCT + 置换表 + 批量叶子评估）
# =========================================================================== #
class Node:
    """搜索树节点。``is_expanded`` 与 ``is_terminal`` 用先验概率是否为空来区分。"""

    __slots__ = ("prior", "visit", "value_sum", "children", "is_expanded", "is_terminal")

    def __init__(self, prior: float) -> None:
        self.prior = float(prior)
        self.visit = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        self.is_expanded = False
        self.is_terminal = False

    def q(self) -> float:
        return self.value_sum / self.visit if self.visit else 0.0


class MCTS:
    """AlphaZero 式 MCTS。

    关键实现点：

    * **置换表**（``self.table``）：同一局面多次到达时复用同一个 ``Node``
      （连五的对局里转置很常见），既省内存也让统计更集中。
      ``add_dirichlet_noise=False``（搜索内部）时不加噪声，只在自我对弈的根节点加。
    * **批量叶子评估**：每次迭代并行模拟 ``leaf_batch`` 条路径，把它们的叶子**一次前向**。
      这是本实现里最关键的加速——CPU 上小网络单次前向的固定开销远大于张量计算。
    * **negamax 视角**：子节点的 ``v`` 是从**对手**视角得到的，回传时取负。
    * **终局节点**：不评估网络，直接按"当前行棋方已输"取 ``-1`` 回传（对手刚赢）。
    """

    def __init__(
        self,
        net: PolicyValueNet,
        device: torch.device,
        *,
        sims: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        leaf_batch: int = 8,
        fpu_reduction: float = 0.2,
    ) -> None:
        self.net = net
        self.device = device
        self.sims = max(1, int(sims))
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)
        self.leaf_batch = max(1, int(leaf_batch))
        self.fpu_reduction = float(fpu_reduction)
        self.table: dict[tuple[bytes, int], Node] = {}
        self.stats: dict[str, float] = {"nodes": 0.0, "leaf_evals": 0.0, "forward_batches": 0.0}
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ #
    def _node(self, state: GameState) -> Node:
        key = state.key()
        node = self.table.get(key)
        if node is None:
            node = Node(prior=1.0)
            self.table[key] = node
        return node

    def _evaluate(self, states: list[GameState]) -> list[tuple[np.ndarray, float]]:
        """对一批叶子局面做一次网络前向，返回 ``[(priors, value), ...]``。"""
        obs = np.stack([s.observation() for s in states])
        masks = np.stack([s.legal_mask() for s in states])
        probs, values = self.net.predict_batch(obs, masks, self.device)
        self.stats["leaf_evals"] += len(states)
        self.stats["forward_batches"] += 1
        out: list[tuple[np.ndarray, float]] = []
        for i in range(len(states)):
            legal = masks[i]
            priors = np.where(legal, probs[i], 0.0)
            total = priors.sum()
            if total <= 0:
                priors = legal.astype(np.float64)
                total = priors.sum()
            out.append((priors / total, float(values[i])))
        return out

    def _expand(self, state: GameState, node: Node, priors: np.ndarray, value: float) -> float:
        """把 ``priors`` 装进节点，返回从 ``state`` 行棋方视角的叶子价值。"""
        legal = state.legal_mask()
        for action in np.flatnonzero(legal):
            node.children[int(action)] = Node(prior=float(priors[int(action)]))
        node.is_expanded = True
        self.stats["nodes"] += 1
        return value

    def _terminal_value(self, state: GameState) -> float:
        """终局节点从**当前行棋方**视角的价值：对手刚刚连成 -> -1；棋盘满 -> 0。"""
        if state.winner != EMPTY:
            return -1.0
        return 0.0

    # ------------------------------------------------------------------ #
    def _select_child(self, node: Node, total_visit: int) -> tuple[int, Node]:
        """PUCT：``Q + c_puct * P * sqrt(N) / (1 + N_a)``。未访问节点用 FPU 估值。"""
        sqrt_total = math.sqrt(max(1, total_visit))
        best_score = -float("inf")
        best_action = -1
        best_child: Node | None = None
        for action, child in node.children.items():
            # FPU（first play urgency）：还没访问过的子节点用「父节点 Q 减一个惩罚」估值，
            # 而不是假装 Q=0（后者在父节点已经明显劣势时会过度乐观）。
            q = child.q() if child.visit else node.q() - self.fpu_reduction
            score = q + self.c_puct * child.prior * sqrt_total / (1 + child.visit)
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        assert best_child is not None
        return best_action, best_child

    def _select_forced(self, node: Node, forced: set[int]) -> tuple[int, Node]:
        """只在 ``forced``（必须封堵的点）里按 PUCT 选；没有可用 forced 时退回全候选。

        为什么这样不会"为了堵而堵"：这些点只会被优先**访问**，不会被强制走。
        如果访问下来发现它们确实差（Q 很低），它们的 PUCT 分数会掉下去，
        而 ``forced`` 只在**根节点**生效，所以最坏情况也只是多花了几次模拟。
        """
        available = {a: node.children[a] for a in forced if a in node.children}
        if not available:
            return self._select_child(node, node.visit)
        sqrt_total = math.sqrt(max(1, node.visit))
        best_score = -float("inf")
        best_action, best_child = -1, None
        for action, child in available.items():
            q = child.q() if child.visit else node.q() - self.fpu_reduction
            score = q + self.c_puct * child.prior * sqrt_total / (1 + child.visit)
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        assert best_child is not None
        return best_action, best_child

    def run(self, root_state: GameState, *, add_noise: bool = False) -> Node:
        """从 ``root_state`` 出发跑 ``sims`` 次模拟，返回根节点。

        **战术处理**（必须有，否则会输棋）：纯 PUCT 只按网络先验分配搜索预算，
        而"立刻能连五的一手"往往先验很小（实测未训练网络给胜点先验 0.002，
        在 73 个候选里排不进前 48 次模拟）——于是搜索会**漏掉必胜手**。
        围棋里一手棋的价值由网络估值兜住，五子棋不会，所以这里显式处理：

        * 有立刻获胜点 → 直接把该点钉成唯一选项（MCTS 只访问它），保证必胜不漏；
        * 有对手立刻获胜点（= 必须封堵）→ 把这些点加进 ``forced`` 集合，
          选点时优先访问；只有当所有 ``forced`` 点的 PUCT 分数都被证明更差时才走别的。
        """
        root = self._node(root_state)
        root.prior = 1.0

        # 根节点先扩展（拿到 priors 与 root value）
        if not root.is_expanded and not root_state.terminal():
            (priors, value) = self._evaluate([root_state])[0]
            if add_noise and len(priors):
                noise = self._rng.dirichlet([self.dirichlet_alpha] * int((priors > 0).sum()))
                legal = np.flatnonzero(priors > 0)
                priors = priors.copy()
                priors[legal] = (1 - self.dirichlet_eps) * priors[legal] + self.dirichlet_eps * noise
            self._expand(root_state, root, priors, value)
        elif root_state.terminal():
            root.is_terminal = True
            root.is_expanded = True

        if root.is_terminal:
            return root

        # ---- 战术处理：必胜手 / 必堵手（见 docstring）----
        winning = np.flatnonzero(root_state.winning_moves())
        if winning.size:
            # 直接连五：把候选裁成"可胜点"，MCTS 全部预算都花在它们身上
            keep = {int(a) for a in winning}
            root.children = {a: c for a, c in root.children.items() if a in keep}
            forced_actions: set[int] = set()
        else:
            # 对手下一手就能连五 -> 这些都是"必须考虑"的封堵点
            forced_actions = {
                int(a) for a in np.flatnonzero(root_state.winning_moves(root_state.player ^ 3))
            }

        sims_done = 1
        while sims_done < self.sims:
            batch = min(self.leaf_batch, self.sims - sims_done)
            leaves: list[tuple[Node, GameState]] = []
            pending: list[tuple[list[Node], float]] = []  # (路径节点, 该路径将要回传的 v)

            for _ in range(batch):
                state = root_state.copy()
                node = root
                path = [node]
                while node.is_expanded and not node.is_terminal and node.children:
                    if node is root and forced_actions:
                        action, child = self._select_forced(root, forced_actions)
                    else:
                        action, child = self._select_child(node, node.visit)
                    state.step(action)
                    node = child
                    path.append(node)
                    if state.terminal():
                        break

                if state.terminal():
                    value = self._terminal_value(state)  # 当前行棋方视角
                    pending.append((path, value))
                    sims_done += 1
                    continue

                leaves.append((node, state))
                pending.append((path, float("nan")))  # 占位，稍后填网络价值
                sims_done += 1

            # 批量评估正常叶子
            if leaves:
                evals = self._evaluate([s for _node, s in leaves])
                it = iter(evals)
                for index, (node, state) in enumerate(leaves):
                    priors, value = next(it)
                    self._expand(state, node, priors, value)
                    path, _ = pending[index]
                    pending[index] = (path, value)

            # 从叶子往回回传（视角每层翻转一次符号）
            for path, value in pending:
                v = value
                for node in reversed(path):
                    node.visit += 1
                    node.value_sum += v
                    v = -v

        return root

    # ------------------------------------------------------------------ #
    def tree_policy_vector(self, root: Node, n_cells: int, temperature: float) -> np.ndarray:
        """``π(a) ∝ N(a)^{1/τ}``，展开成 ``(n_cells,)`` 的分布（策略训练目标）。

        ``τ → 0``（用 1e-3 代替）退化为 one-hot argmax：这是 AlphaZero 后期
        "不再探索、只走最优"的做法。
        """
        out = np.zeros(n_cells, dtype=np.float32)
        if not root.children:
            return out
        actions = sorted(root.children)
        visits = np.array([root.children[a].visit for a in actions], dtype=np.float64)
        if visits.sum() <= 0:
            visits = np.ones_like(visits)
        if temperature <= 1e-3:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
        else:
            scaled = visits ** (1.0 / temperature)
            probs = scaled / scaled.sum()
        out[actions] = probs
        return out

    def advance(self, root: Node, action: int) -> Node | None:
        """树复用：把 ``action`` 对应的子节点提为新根。"""
        child = root.children.get(int(action))
        if child is None:
            return None
        child.prior = 1.0
        return child
