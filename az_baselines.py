"""自包含的规则型对手 + 成对开局评估（本交付不依赖任何 PPO 模块）。

为什么有这个文件
----------------
主工程里这几个能力分散在 ``learning.py`` / ``train.py`` / ``paired_eval.py`` /
``compare_models.py``——那些都是 **PPO 侧**的代码。本仓库只交付「环境 + AlphaZero 训练」，
所以在这里给出等价的最小实现，使 ``az_train.py`` / ``az_eval.py`` 可以独立运行：

* :class:`RandomOpponent` / :class:`HeuristicOpponent` / :class:`TacticalHeuristicOpponent`
  ——与主工程 ``learning.py`` 里的同名类**策略完全一致**（"先赢再堵 + 邻域打分"、
  显式化解"活三/双威胁"），供 ``az_train.py`` 的周期性评估使用；
* :class:`TacticalAgent`——把 :class:`TacticalHeuristicOpponent` 包成 ``arena`` 的 Agent 协议，
  供 ``az_eval.py`` 使用（``arena.HeuristicAgent`` / ``arena.RandomAgent`` 已可直接用）；
* :func:`paired_eval`——**成对开局**协议：同一个随机开局下让 A 先手、后手各下一局，
  一对两局取半分。9x9 连五先手优势很大，普通"黑白各半"统计会被颜色效应淹没，
  成对开局把它无论方向都抵消掉（详见 README §评估）。

规则对手为什么重要
------------------
``tactical`` **没有参与任何训练**，是 held-out 对手。只用"训练内对手"评估会高估棋力：
``heuristic`` 只会堵"对手下一手就能连成"的点，看不到"活三"（下一手会有两个获胜点），
实测它会在 7~10 手内被学会斜向双威胁的策略杀掉。
"""

from __future__ import annotations

import numpy as np

from env import _DIRECTIONS, BLACK, EMPTY, WHITE, GomokuEnv

__all__ = [
    "HeuristicOpponent",
    "RandomOpponent",
    "TacticalAgent",
    "TacticalHeuristicOpponent",
    "paired_eval",
]


# =========================================================================== #
# 规则型对手（select 协议：env / obs / mask -> action）
# =========================================================================== #
class RandomOpponent:
    """均匀随机合法落子。"""

    is_self = False
    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def select(
        self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray, dist=None
    ) -> int:
        del env, obs, dist
        return int(self.rng.choice(np.flatnonzero(mask)))


class HeuristicOpponent:
    """"先赢再堵 + 邻域打分"的规则对手。

    ``dist`` 参数只是为了让 ``arena.HeuristicAgent``（会多传一个先验分布）也能复用本类，
    本类不使用它。
    """

    is_self = False
    name = "heuristic"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def select(
        self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray, dist=None
    ) -> int:
        del obs, dist
        me = env.current_player
        them = env.opponent(me)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("无合法动作")

        for a in legal:  # 1. 直接获胜
            if env.is_winning_move(int(a), me):
                return int(a)
        for a in legal:  # 2. 挡住对手
            if env.is_winning_move(int(a), them):
                return int(a)

        board = env.board
        n = env.board_size
        radius = 2
        best_score = -1.0
        best: list[int] = []
        for a in legal:
            r, c = divmod(int(a), n)
            score = 0.0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < n and 0 <= cc < n):
                        continue
                    v = int(board[rr, cc])
                    if v == EMPTY:
                        continue
                    weight = 1.0 / (1 + abs(dr) + abs(dc))
                    score += weight * (1.0 if v == me else 0.8)
            if score > best_score + 1e-9:
                best_score, best = score, [int(a)]
            elif abs(score - best_score) <= 1e-9:
                best.append(int(a))
        if not best or best_score <= 0.0:
            return int(self.rng.choice(legal))
        return int(self.rng.choice(best))


class TacticalHeuristicOpponent:
    """比 :class:`HeuristicOpponent` 强得多的**规则**对手：显式处理"双重威胁"。

    只用 :class:`HeuristicOpponent`（先赢再堵 + 邻域打分）当评估基线是不够的：
    它只会堵「对手立刻能连成的一手」，看不到 "活三"（下一手会有**两个**获胜点）。
    实测它会在 7~10 手内被学会斜向双威胁的策略杀掉，因此拿它当唯一基线会**高估**棋力。

    本对手对每个合法点做一次"模拟落子 + 枚举对手的立刻获胜点"，然后：

    1. 自己下一手能连成就赢（优先）；
    2. 否则选「让对手的立刻获胜点最少」的点：把对手压到 0 个就是唯一正确的防守；
       做不到 0 时（对手已有双威胁）退化为尽量拖；
    3. 同样安全（对手 0 个获胜点）的点之间，用邻域权重打分挑一个（制造自己的威胁）。

    它**不参与训练**，因此可以作为"held-out 对手"检验策略是不是真的会封堵，
    而不是只对训练对手过拟合。
    """

    is_self = False
    name = "tactical"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def _winning_squares(self, env: GomokuEnv, player: int) -> set[int]:
        """``player`` 落子即可获胜的空点集合。

        不能直接用 ``env.is_winning_move``：它按"该 player 已有的棋子"判断，
        而这里要问的是"换个 player 下这一手会不会连成"，
        所以临时放子、检查是否连成 ``win_length``，再还原。
        """
        board = env.board
        n, k = env.board_size, env.win_length
        out: set[int] = set()
        for index in np.flatnonzero(board.reshape(-1) == EMPTY):
            row, col = divmod(int(index), n)
            board[row, col] = player
            try:
                if self._connects(board, row, col, player, n, k):
                    out.add(int(index))
            finally:
                board[row, col] = EMPTY
        return out

    @staticmethod
    def _connects(board: np.ndarray, row: int, col: int, player: int, n: int, k: int) -> bool:
        """``board[row, col]`` 刚被 ``player`` 占住，判断是否形成 ``k`` 连。"""
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < n and 0 <= c < n and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= k:
                return True
        return False

    def _neighbor_score(self, env: GomokuEnv, action: int, me: int) -> float:
        board = env.board
        n = env.board_size
        r, c = divmod(action, n)
        score = 0.0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                value = int(board[rr, cc])
                if value == EMPTY:
                    continue
                weight = 1.0 / (1 + abs(dr) + abs(dc))
                score += weight * (1.0 if value == me else 0.8)
        return score

    def select(
        self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray, dist=None
    ) -> int:
        del obs, dist
        me = env.current_player
        them = env.opponent(me)
        legal = [int(a) for a in np.flatnonzero(mask)]
        if not legal:
            raise RuntimeError("无合法动作")

        my_wins = self._winning_squares(env, me)
        if my_wins:  # 1. 直接获胜
            return int(self.rng.choice(sorted(my_wins)))

        # 2. 我落子只是占掉一个空点，**不会给对手造出新威胁**，
        #    所以"我下完之后对手的立刻获胜点" = 原威胁集合去掉我占的那一点。
        opponent_threats = self._winning_squares(env, them)
        scored: list[tuple[int, float, int]] = []
        for action in legal:
            replies = len(opponent_threats) - (1 if action in opponent_threats else 0)
            scored.append((replies, self._neighbor_score(env, action, me), action))

        min_reply = min(item[0] for item in scored)
        safe = [item for item in scored if item[0] == min_reply]
        top = max(item[1] for item in safe)
        picks = [item[2] for item in safe if abs(item[1] - top) <= 1e-9]
        return int(self.rng.choice(picks))


# =========================================================================== #
# arena.Agent 协议适配器
# =========================================================================== #
class TacticalAgent:
    """把 :class:`TacticalHeuristicOpponent` 包成 ``arena`` 的 Agent。"""

    def __init__(self, name: str = "tactical", seed: int = 0) -> None:
        self.name = name
        self.opponent = TacticalHeuristicOpponent(np.random.default_rng(seed))

    def reseed(self, seed: int) -> None:
        self.opponent.rng = np.random.default_rng(int(seed))

    def act(self, env: GomokuEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        return int(self.opponent.select(env, obs, np.asarray(mask, dtype=bool)))


# =========================================================================== #
# 成对开局评估
# =========================================================================== #
def _random_opening(
    rng: np.random.Generator, *, plies: int, board_size: int, win_length: int
) -> dict:
    """随机走 ``plies`` 手作为开局（无胜负）后的状态快照。"""
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset()
    for _ in range(plies):
        legal = np.flatnonzero(env.legal_mask())
        if legal.size == 0 or env._done:
            break
        env.step(int(rng.choice(legal)))
    return env.get_state()


def _play_from(
    state: dict,
    black,
    white,
    *,
    board_size: int,
    win_length: int,
    max_moves: int | None = None,
) -> tuple[int, int]:
    """从给定局面下完一局，返回 ``(赢家, 手数)``（``EMPTY`` 为和/截断）。"""
    env = GomokuEnv(board_size=board_size, win_length=win_length)
    env.reset(options={"state": state})
    while not env._done:
        if max_moves is not None and env.move_count >= max_moves:
            return EMPTY, int(env.move_count)
        mask = env.legal_mask()
        if not mask.any():
            return EMPTY, int(env.move_count)
        agent = black if env.current_player == BLACK else white
        # 观测只是协议占位：AZAgent 用 GameState 自行编码，规则对手也不读 obs；
        # 若传入需要 6 平面观测的 arena.ModelAgent，arena 会按 n_planes 自行准备。
        obs = env._observation()
        action = int(agent.act(env, obs, mask))
        if not bool(mask[action]):
            raise AssertionError(f"Agent {agent.name!r} 返回非法动作 {action}")
        env.step(action)
    return int(env.winner), int(env.move_count)


def paired_eval(
    a,
    b,
    *,
    pairs: int,
    plies: int,
    seed: int,
    board_size: int = 9,
    win_length: int = 5,
) -> dict:
    """成对开局评估：``a`` 对 ``b``，每对同一个开局下两局（a 先/后手各一）。

    一对两局的得分：A 两胜 ``1.0``、一胜一负 ``0.5``、两负 ``0.0``，跨对取平均并给标准误。
    ``win_length`` 默认 **5**（本交付是 9x9 连五）；``az_eval.py`` 会按 checkpoint 里
    记录的规则显式传入，避免"用连四规则评估连五网络"这类规则不一致。
    """
    # 与 arena.play_match 一致：开赛前重置双方 Agent 的随机源，
    # 否则复用同一对 Agent 重复调用会因随机数流前进而得到漂移的结果。
    for agent in (a, b):
        reseed = getattr(agent, "reseed", None)
        if callable(reseed):
            reseed(seed)

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    lengths: list[int] = []
    for _pair in range(pairs):
        state = _random_opening(
            rng, plies=plies, board_size=board_size, win_length=win_length
        )
        # A 执黑
        first, length_first = _play_from(
            state, a, b, board_size=board_size, win_length=win_length
        )
        # A 执白（同一个开局）
        second, length_second = _play_from(
            state, b, a, board_size=board_size, win_length=win_length
        )
        lengths.extend([length_first, length_second])

        score = 0.0
        if first == EMPTY:  # A 执黑的一局和棋
            score += 0.5
        elif first == BLACK:
            score += 1.0
        if second == EMPTY:  # A 执白的一局和棋
            score += 0.5
        elif second == WHITE:
            score += 1.0
        scores.append(score / 2.0)

    n = max(1, len(scores))
    mean = float(np.mean(scores)) if scores else 0.0
    stderr = float(np.std(scores, ddof=1) / np.sqrt(n)) if len(scores) > 1 else 0.0
    return {
        "pairs": pairs,
        "plies": plies,
        "score_rate": mean,
        "stderr": stderr,
        "ci95": [max(0.0, mean - 1.96 * stderr), min(1.0, mean + 1.96 * stderr)],
        "pairs_won": sum(1 for s in scores if s == 1.0),
        "pairs_drawn": sum(1 for s in scores if s == 0.5),
        "pairs_lost": sum(1 for s in scores if s == 0.0),
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
    }
