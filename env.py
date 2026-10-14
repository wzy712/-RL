"""五子棋 (Gomoku / Five-in-a-Row) 强化学习环境。

设计要点
--------
* **Gymnasium 兼容**：实现 ``reset/step/close/render``，使用
  ``spaces.Discrete`` + ``spaces.Box``，并注册为 ``Gomoku-v0``。
* **统一视角观测 (canonical observation)**：观测永远以 *当前行棋方* 的视角给出，
  因此同一个策略网络可以同时执黑与执白（自博弈的关键）。
* **奖励视角**：``step()`` 返回的 ``reward`` 是 *刚刚落子那一方* 的收益。
  赢棋 ``+1``、和棋/未结束 ``0``。输棋方不会收到 ``-1``——因为对局在赢棋那一步就结束了；
  这份 "负收益" 由训练端的 negamax 自举 (``V(s_{t+1})`` 取负) 正确补齐，
  详见 ``train.py`` 中 ``compute_gae_negamax`` 的推导注释。
* **非法动作掩码**：``legal_mask()`` 返回形状 ``(board_size**2,)`` 的 bool 数组，
  供采样、log-prob、entropy 三处统一使用。
* **零依赖渲染**：``render()`` 懒加载 pygame，训练时不会 import，
  因此在无显示器的服务器上也能正常训练。

观测张量 (float32, 形状 ``(4, N, N)``)
--------------------------------------
===== ==========================================
通道   含义
===== ==========================================
0     当前行棋方的棋子
1     对手的棋子
2     行棋方指示平面 (执黑全 1 / 执白全 0)
3     最后一手落点 one-hot (无则为全 0)
===== ==========================================
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
EMPTY: int = 0
BLACK: int = 1
WHITE: int = 2

#: 观测通道数，修改时需同步 ``_observation`` 与网络输入层
N_PLANES: int = 4

#: 四个检查方向：横、竖、主对角、副对角
_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

DEFAULT_RENDER_FPS: int = 4


class IllegalMoveError(ValueError):
    """落在已有棋子的位置，或对局已结束时继续落子。"""


class GomokuEnv(gym.Env):
    """双人零和五子棋环境。

    Parameters
    ----------
    board_size:
        棋盘边长，默认 9（训练快）；标准五子棋为 15。
    win_length:
        连成多少子算赢，默认 5。``win_length=4/board_size=9`` 可用于快速冒烟测试。
    illegal_move_mode:
        ``"raise"``（默认，便于抓出掩码 bug）或 ``"lose"``（判负，便于和外部 agent 对接）。
    render_mode:
        ``None`` / ``"human"`` / ``"rgb_array"`` / ``"ansi"``。
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": DEFAULT_RENDER_FPS,
        "name": "Gomoku-v0",
    }

    def __init__(
        self,
        board_size: int = 9,
        win_length: int = 5,
        illegal_move_mode: str = "raise",
        render_mode: str | None = None,
        window_size: int = 512,
    ) -> None:
        super().__init__()
        if board_size < 3:
            raise ValueError(f"board_size 至少为 3，收到 {board_size}")
        if not 3 <= win_length <= board_size:
            raise ValueError(
                f"需要 3 <= win_length <= board_size，收到 win_length={win_length}, "
                f"board_size={board_size}"
            )
        if illegal_move_mode not in {"raise", "lose"}:
            raise ValueError(f"非法的 illegal_move_mode: {illegal_move_mode}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"不支持的 render_mode: {render_mode}")
        if window_size < board_size * 8:
            raise ValueError(f"window_size 太小: {window_size}（棋盘 {board_size}x{board_size}）")

        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.illegal_move_mode = illegal_move_mode
        self.render_mode = render_mode

        self.n_cells = self.board_size * self.board_size

        # 观测：float32, (4, N, N)，取值 {0, 1}
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_PLANES, self.board_size, self.board_size), dtype=np.float32
        )
        # 动作：棋盘上的扁平索引 action = row * N + col
        self.action_space = spaces.Discrete(self.n_cells)

        # --- 内部状态 ---
        self.board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player: int = BLACK
        self.last_action: int = -1
        self.move_count: int = 0
        self.winner: int = EMPTY  # EMPTY 表示未结束或是和棋
        self.winning_line: list[tuple[int, int]] = []
        self._done: bool = False

        # pygame 相关（懒加载）
        self._screen: Any = None
        self._clock: Any = None
        self.window_size = int(window_size)  # 渲染窗口边长（棋盘会被缩放到这个尺寸内）

    # ------------------------------------------------------------------ #
    # 基础工具
    # ------------------------------------------------------------------ #
    def action_to_xy(self, action: int) -> tuple[int, int]:
        """扁平动作 -> (row, col)。"""
        return divmod(int(action), self.board_size)

    def xy_to_action(self, row: int, col: int) -> int:
        """(row, col) -> 扁平动作。"""
        return int(row) * self.board_size + int(col)

    def opponent(self, player: int | None = None) -> int:
        """返回对手棋子颜色。"""
        p = self.current_player if player is None else player
        return WHITE if p == BLACK else BLACK

    def legal_mask(self) -> np.ndarray:
        """形状 ``(N*N,)`` 的 bool 掩码，True 表示该点可落子。

        对局结束后返回全 False（``step`` 已不允许继续）。
        """
        mask = (self.board.reshape(-1) == EMPTY) & (not self._done)
        return np.ascontiguousarray(mask)

    # sb3-contrib / 常见约定别名
    action_masks = legal_mask

    def legal_actions(self) -> np.ndarray:
        """所有合法动作的索引数组。"""
        return np.flatnonzero(self.legal_mask())

    def sample_legal_action(self) -> int:
        """在合法动作中均匀随机采样一个（随机对手 / rollout 用）。"""
        legal = self.legal_actions()
        if legal.size == 0:
            raise IllegalMoveError("棋盘已满，无合法动作")
        return int(self.np_random.choice(legal))

    def empty_mask(self) -> np.ndarray:
        """形状 ``(N, N)`` 的空位掩码（便于可视化与调试）。"""
        return self.board == EMPTY

    # ------------------------------------------------------------------ #
    # 状态存取（MCTS / 对手池 / 复盘存档用）
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict[str, Any]:
        """导出可深拷贝的最小状态快照。"""
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "last_action": self.last_action,
            "move_count": self.move_count,
            "winner": self.winner,
            "winning_line": list(self.winning_line),
            "done": self._done,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """恢复 ``get_state()`` 导出的快照。"""
        self.board = np.array(state["board"], dtype=np.int8).reshape(
            self.board_size, self.board_size
        )
        self.current_player = int(state["current_player"])
        self.last_action = int(state["last_action"])
        self.move_count = int(state["move_count"])
        self.winner = int(state["winner"])
        self.winning_line = [tuple(x) for x in state["winning_line"]]
        self._done = bool(state["done"])

    def clone(self) -> GomokuEnv:
        """返回一个状态完全相同的独立环境（不含 pygame 资源）。"""
        other = GomokuEnv(
            board_size=self.board_size,
            win_length=self.win_length,
            illegal_move_mode=self.illegal_move_mode,
            render_mode=None,
        )
        other.set_state(self.get_state())
        other.np_random = self.np_random
        return other

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置对局。

        ``options`` 支持：

        * ``starting_player``: ``BLACK``(1) 或 ``WHITE``(2)，默认 1（黑先）。
        * ``state``: 由 :meth:`get_state` 产生的快照，用于从任意局面续弈。
        """
        super().reset(seed=seed)
        options = options or {}

        if "state" in options:
            self.set_state(options["state"])
        else:
            self.board.fill(EMPTY)
            self.current_player = int(options.get("starting_player", BLACK))
            if self.current_player not in (BLACK, WHITE):
                raise ValueError(f"starting_player 只能是 1 或 2，收到 {self.current_player}")
            self.last_action = -1
            self.move_count = 0
            self.winner = EMPTY
            self.winning_line = []
            self._done = False

        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """落子一步。

        Returns
        -------
        obs, reward, terminated, truncated, info

        ``reward`` 以 *刚落子的一方* 为视角：赢棋 ``+1``，其余 ``0``。
        ``obs`` 以 *接下来要行棋的一方* 为视角。
        """
        if self._done:
            raise IllegalMoveError("对局已结束，请先调用 reset()")

        action = int(action)
        if not 0 <= action < self.n_cells:
            raise IllegalMoveError(f"动作越界: {action}")
        row, col = divmod(action, self.board_size)
        if self.board[row, col] != EMPTY:
            if self.illegal_move_mode == "lose":
                mover = self.current_player
                self.winner = self.opponent(mover)
                self._done = True
                self.last_action = action
                return self._observation(), 1.0, True, False, self._info()
            raise IllegalMoveError(f"位置 ({row}, {col}) 已有棋子")

        mover = self.current_player
        self.board[row, col] = mover
        self.last_action = action
        self.move_count += 1

        line = self._find_winning_line(row, col, mover)
        reward = 0.0
        if line is not None:
            self.winner = mover
            self.winning_line = line
            self._done = True
            reward = 1.0
        elif self.move_count >= self.n_cells:
            self.winner = EMPTY  # 棋盘填满 -> 和棋
            self._done = True
        else:
            self.current_player = self.opponent(mover)

        return self._observation(), reward, self._done, False, self._info()

    def render(self) -> np.ndarray | None:
        """渲染当前局面。

        * ``render_mode="ansi"``：返回文本棋盘的字符串（同时 print）。
        * ``render_mode="rgb_array"``：返回 ``(H, W, 3)`` uint8 数组（可在无显示器下工作）。
        * ``render_mode="human"``：弹出 pygame 窗口并刷新。
        * ``render_mode=None``：什么也不做。
        """
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            text = self.board_str()
            print(text)
            return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        return self._render_pygame(capture=self.render_mode == "rgb_array")

    def close(self) -> None:
        """释放 pygame 窗口。"""
        if self._screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None
        super().close()

    # ------------------------------------------------------------------ #
    # 规则：胜负判定
    # ------------------------------------------------------------------ #
    def _find_winning_line(
        self, row: int, col: int, player: int
    ) -> list[tuple[int, int]] | None:
        """只围绕最后一手做增量检查（O(win_length)，比全盘扫描快得多）。

        返回构成连线的坐标列表；未连成则返回 ``None``。
        """
        n, k = self.board_size, self.win_length
        for dr, dc in _DIRECTIONS:
            cells: list[tuple[int, int]] = [(row, col)]
            # 正方向延伸
            r, c = row + dr, col + dc
            while 0 <= r < n and 0 <= c < n and self.board[r, c] == player:
                cells.append((r, c))
                r += dr
                c += dc
            # 反方向延伸
            r, c = row - dr, col - dc
            while 0 <= r < n and 0 <= c < n and self.board[r, c] == player:
                cells.insert(0, (r, c))
                r -= dr
                c -= dc
            if len(cells) >= k:
                # 取包含最后一手的那一段连续 k 子
                start = cells.index((row, col))
                lo = max(0, min(start - k + 1, len(cells) - k))
                return cells[lo : lo + k]
        return None

    def is_winning_move(self, action: int, player: int | None = None) -> bool:
        """假设 ``player`` 在 ``action`` 落子，判断是否能立刻连成 ``win_length`` 子。

        不改变环境状态，供启发式对手 / 单元测试使用。
        """
        row, col = self.action_to_xy(action)
        if self.board[row, col] != EMPTY:
            return False
        p = self.current_player if player is None else player
        n = self.board_size
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < n and 0 <= c < n and self.board[r, c] == p:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.win_length:
                return True
        return False

    def check_winner_full_scan(self) -> int:
        """全盘扫描判定赢家（仅供测试，训练路径走增量判定）。返回 EMPTY/BLACK/WHITE。"""
        n, k = self.board_size, self.win_length
        for r in range(n):
            for c in range(n):
                p = int(self.board[r, c])
                if p == EMPTY:
                    continue
                for dr, dc in _DIRECTIONS:
                    er, ec = r + (k - 1) * dr, c + (k - 1) * dc
                    if not (0 <= er < n and 0 <= ec < n):
                        continue
                    if all(self.board[r + i * dr, c + i * dc] == p for i in range(k)):
                        return p
        return EMPTY

    # ------------------------------------------------------------------ #
    # 观测 / info
    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        """以当前行棋方视角生成 ``(4, N, N)`` float32 观测。"""
        n = self.board_size
        me = self.current_player
        opp = self.opponent(me)

        obs = np.zeros((N_PLANES, n, n), dtype=np.float32)
        obs[0] = self.board == me
        obs[1] = self.board == opp
        obs[2] = 1.0 if me == BLACK else 0.0
        if self.last_action >= 0:
            r, c = divmod(self.last_action, n)
            obs[3, r, c] = 1.0
        return obs

    def _info(self) -> dict[str, Any]:
        mask = self.legal_mask()
        return {
            "legal_mask": mask,
            # 兼容 sb3-contrib / PettingZoo 的命名习惯
            "action_mask": mask,
            "player": self.current_player,
            "move_count": self.move_count,
            "winner": self.winner,
            "winning_line": list(self.winning_line),
            "is_draw": self._done and self.winner == EMPTY,
        }

    # ------------------------------------------------------------------ #
    # 可视化
    # ------------------------------------------------------------------ #
    def board_str(self) -> str:
        """文本棋盘（``.`` 空、``X`` 黑、``O`` 白），最后一行是列号。"""
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        n = self.board_size
        header = "   " + " ".join(f"{c % 10}" for c in range(n))
        rows = [header]
        for r in range(n):
            cells = " ".join(symbols[int(v)] for v in self.board[r])
            rows.append(f"{r:2d} {cells}")
        if self._done:
            result = (
                "和棋"
                if self.winner == EMPTY
                else ("黑棋胜" if self.winner == BLACK else "白棋胜")
            )
            rows.append(f"结果: {result}")
        else:
            turn = "黑" if self.current_player == BLACK else "白"
            rows.append(f"轮到: {turn}  手数: {self.move_count}")
        return "\n".join(rows)

    def board_geometry(self) -> tuple[int, int, int]:
        """返回 pygame 绘制几何 ``(board_px, margin, cell)``。

        ``render()`` 与 ``play.py`` / ``watch.py`` 的坐标换算共用同一套几何，
        避免两处不一致。
        """
        n = self.board_size
        margin = self.window_size // (n + 1)
        cell = (self.window_size - 2 * margin) // max(1, n - 1)
        size = 2 * margin + cell * (n - 1)
        return size, margin, cell

    def pixel_to_action(self, pos: tuple[int, int], tolerance: float = 0.45) -> int | None:
        """把窗口像素坐标映射成最近的交叉点动作；离最近交叉点太远则返回 ``None``。"""
        _, margin, cell = self.board_geometry()
        col = round((pos[0] - margin) / max(1, cell))
        row = round((pos[1] - margin) / max(1, cell))
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return None
        if abs(pos[0] - (margin + col * cell)) > cell * tolerance:
            return None
        if abs(pos[1] - (margin + row * cell)) > cell * tolerance:
            return None
        return self.xy_to_action(row, col)

    def _render_pygame(self, capture: bool) -> np.ndarray | None:
        """用 pygame 绘制棋盘。``capture=True`` 时返回 RGB 数组（离屏）。"""
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        size, margin, cell = self.board_geometry()
        stone_r = int(cell * 0.42)

        if capture and self._screen is None:
            # 离屏 Surface，无需真实窗口/显示器
            surf = pygame.Surface((size, size))
            self.draw_board(surf, margin, cell, stone_r)
            return self._surface_to_array(surf)

        if self._screen is None:
            pygame.init()
            n = self.board_size
            pygame.display.set_caption(f"Gomoku {n}x{n} (win {self.win_length})")
            self._screen = pygame.display.set_mode((size, size))
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 关窗即安全退出
                self.close()
                return None

        self.draw_board(self._screen, margin, cell, stone_r)
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.metadata["render_fps"])
        return None

    def draw_board(self, surf: Any, margin: int, cell: int, stone_r: int) -> None:
        """把棋盘（木纹 + 网格 + 星位 + 棋子 + 最后一手标记 + 连五高亮）画到 ``surf``。

        公开方法：``render()`` 内部使用；``watch.py`` 也可以把它画到子 Surface 上，
        从而做出带侧栏与策略热力图的界面。
        """
        import pygame

        n = self.board_size
        wood = (222, 184, 135)
        line_color = (60, 40, 20)
        surf.fill(wood)

        for i in range(n):
            offset = margin + i * cell
            end = margin + (n - 1) * cell
            pygame.draw.line(surf, line_color, (margin, offset), (end, offset), 1)
            pygame.draw.line(surf, line_color, (offset, margin), (offset, end), 1)

        # 星位（天元 / 四星）
        if n >= 9:
            mid = n // 2
            edge = 3 if n >= 13 else 2
            for r in {edge, mid, n - 1 - edge}:
                for c in {edge, mid, n - 1 - edge}:
                    pygame.draw.circle(
                        surf, line_color, (margin + c * cell, margin + r * cell), 3
                    )

        for r in range(n):
            for c in range(n):
                v = int(self.board[r, c])
                if v == EMPTY:
                    continue
                center = (margin + c * cell, margin + r * cell)
                color = (25, 25, 25) if v == BLACK else (245, 245, 245)
                pygame.draw.circle(surf, color, center, stone_r)
                pygame.draw.circle(surf, line_color, center, stone_r, 1)
                if self.xy_to_action(r, c) == self.last_action:
                    pygame.draw.circle(surf, (220, 40, 40), center, max(2, stone_r // 4))

        if self._done and self.winning_line:
            for r, c in self.winning_line:
                center = (margin + c * cell, margin + r * cell)
                pygame.draw.circle(surf, (230, 60, 60), center, stone_r, 3)

    @staticmethod
    def _surface_to_array(surf: Any) -> np.ndarray:
        import pygame

        array = pygame.surfarray.array3d(surf)  # (W, H, 3)
        return np.transpose(array, (1, 0, 2)).copy()


# --------------------------------------------------------------------------- #
# Gymnasium 注册
# --------------------------------------------------------------------------- #
def _register() -> None:
    try:
        gym.register(
            id="Gomoku-v0",
            entry_point=GomokuEnv,
            kwargs={"board_size": 9, "win_length": 5},
        )
        gym.register(
            id="Gomoku15-v0",
            entry_point=GomokuEnv,
            kwargs={"board_size": 15, "win_length": 5},
        )
    except Exception:  # 重复注册（模块被多次 import）时静默跳过
        pass


_register()


__all__ = [
    "BLACK",
    "EMPTY",
    "N_PLANES",
    "WHITE",
    "GomokuEnv",
    "IllegalMoveError",
]
