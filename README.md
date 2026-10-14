# 五子棋 AlphaZero（环境 + 自我对弈训练）

用 **AlphaZero 范式**下五子棋：MCTS 搜索产生策略目标、对局结果产生价值目标，
策略头与价值头**各自独立主干**。环境是标准 `gymnasium.Env`，训练/评估脚本都是纯 Python 脚本，
无需安装成包。

```
env.py            GomokuEnv：9x9 连五（可配 board_size / win_length），Gymnasium 接口
alphazero.py      PUCT-MCTS（置换表 + 批量叶子评估 + 子树复用）+ PolicyValueNet + 6 平面观测编码
az_train.py       AlphaZero 自我对弈训练（回放池、对称增强、周期性评估、checkpoint）
az_eval.py        checkpoint 棋力评估（成对开局协议，对阵规则对手 / 另一个 checkpoint）
az_baselines.py   自包含的规则型对手 + 成对开局评估实现
arena.py          Agent 协议、对局框架（play_match / 统计 / 置信区间）与内置 Agent
tests/            env / alphazero / arena / az_baselines 的回归测试
```

## 快速开始

需要 Python 3.12–3.13（`pygame` 目前只有到 cp313 的 wheel）。

```bash
uv sync                       # 或 pip install torch numpy gymnasium

# 冒烟：CPU 上约 1 分钟，验证整条链路
uv run python az_train.py --run-name az_smoke --iterations 2 --games-per-iter 2 \
    --sims 8 --leaf-batch 2 --channels 16 --blocks 1 --train-steps 2 --eval-interval 0

# 正式训练（9x9 连五；MPS 上每轮约 4 分钟）
uv run python az_train.py --run-name az9x9w5 --iterations 50 --games-per-iter 40 \
    --sims 100 --channels 96 --blocks 5

# 评估（成对开局；棋盘大小与连子数自动取 checkpoint 里的规则）
uv run python az_eval.py --checkpoint checkpoints/az9x9w5/best.pt --sims 100 \
    --pairs 40 --opponents heuristic tactical random

# 测试
uv run pytest -q
```

`az_train.py` 的全部超参数由 `AZConfig` 数据类自动生成命令行参数，`--help` 永远和代码一致：

```bash
uv run python az_train.py --help
```

## 环境（`env.py`）

* 棋盘 `board_size × board_size`（默认 9），连 `win_length` 子获胜（默认 5），下满为和棋。
* 自由规则（长连也算胜）；Renju 禁手未实现。
* 胜负判定是**增量**的：只检查最后一手所在的行/列/两条对角线（`O(4·win_length)`）；
  `check_winner_full_scan()` 保留全盘扫描用于测试对拍。

**观测** `(4, N, N)` float32，永远是**当前行棋方视角**：

| 通道 | 含义 |
|---|---|
| 0 | 己方棋子 |
| 1 | 对方棋子 |
| 2 | 行棋方指示平面（执黑全 1 / 执白全 0） |
| 3 | 最后一手 one-hot（开局全 0） |

AlphaZero 在此之上再追加两个**战术平面**（见下），编码为 6 平面。

**奖励**：落子后连成 `win_length` 子记 `+1`，其它（含和棋）记 `0`；输方不记 `-1`
（对局在赢棋那一步就结束，负收益由训练端的 negamax 自举补齐）。

**接口**：

```python
mask = env.legal_mask()                  # (N*N,) bool，True = 可落子；step/reset 的 info 里也有
env.get_state() / env.set_state(state)   # 状态快照（MCTS / 复盘 / 断点续弈）
env.clone()                              # 独立副本
env.reset(options={"state": s})          # 从任意局面续弈
env.reset(options={"starting_player": WHITE})
env.is_winning_move(action, player)      # 某点落下是否立刻连成（供规则对手 / 测试）
env.board_str()                          # 文本棋盘
env.render()                             # render_mode: "human" / "rgb_array" / "ansi" / None
gym.make("Gomoku-v0") / gym.make("Gomoku15-v0")
```

非法动作默认 `illegal_move_mode="raise"`（直接抛 `IllegalMoveError`，方便抓掩码 bug），
也可设 `"lose"` 违例判负。`render_mode="rgb_array"` 走离屏 `pygame.Surface`，headless 可用。

## 算法（`alphazero.py` / `az_train.py`）

**网络**：`PolicyValueNet` 双头。默认 `shared_trunk=False`，即策略头与价值头**各有一套残差主干**
（9x9 / 通道 96 / 5 残差块时约 1.89M 参数，参数约翻倍换取两个目标互不干扰）；
`--shared-trunk True` 可切到经典 AlphaZero 共享主干做对照。

**MCTS 要点**：

| 机制 | 实现 |
|---|---|
| 选点 | PUCT `Q + c_puct·P·√N/(1+N_a)`，未访问子节点用 FPU 估值 `Q_parent - fpu` |
| 探索 | 根节点 Dirichlet 噪声 `P ← (1-ε)P + ε·Dir(α)`（默认 `ε=0.25, α=0.3`） |
| 策略目标 | `π(a) ∝ N(s,a)^{1/τ}`；前 `--temp-moves` 手 `τ=1`，之后 `τ→0` |
| 置换表 | 按 `(棋盘字节, 行棋方)` 复用 `Node`——连五里转置局面很常见 |
| 批量叶子评估 | 每次迭代并行模拟 `--leaf-batch` 条路径，叶子**一次前向**（最大的加速项） |
| 子树复用 | `advance(root, action)` 把上一步搜索的子树提为新根 |
| 视角 | negamax：子节点价值取负回传；终局节点精确取 `∓1` |

**战术处理（五子棋必须有，否则会输棋）**：纯 PUCT 只按网络先验分配预算，而"立刻连五"的一手
先验往往极小（实测未训练网络给胜点的先验约 0.002，73 个合法点里排不进前 48 次模拟），
搜索会**漏掉必胜手**。所以 `MCTS.run` 显式做了两件事：

* 有立刻获胜点 → 根节点候选裁成胜点，搜索预算全部给它；
* 有对手立刻获胜点（必须封堵）→ 放进 `forced` 集合优先访问（只影响访问顺序，不强制走）。

两条都有回归测试守着，**测试用的网络是随机初始化的**——"战术不漏"不依赖网络有多聪明。

**训练循环**：自我对弈产生 `(obs, mask, π, z)` 存环形回放池 → 每轮 `--train-steps` 次梯度更新
（`Adam`、梯度裁剪 1.0）→ 每 `--save-interval` 轮存 checkpoint → 每 `--eval-interval` 轮做
黑白各半评估，按 `0.6×对规则对手胜率 + 0.4×对随机胜率` 存 `best.pt`。
连五对 4 旋转 × 2 镜像完全对称，因此对每个样本随机取一个对称变换做数据增强（严格等价，不是近似）。

**6 平面观测**：在 env 原生 4 平面上追加"当前行棋方立刻能获胜的点 / 对手立刻能获胜的点"。
战术平面是局面的确定性函数，不破坏马尔可夫性，却把"做杀 / 堵四"从稀疏奖励里最难学的两件事
变成 1 个卷积就能利用的特征；训练时 `GameState.observation()` 与网络输入是同一套编码。

## 评估（`az_eval.py`）

用 **成对开局**（paired openings）：同一个随机开局下让被测方先手、后手各下一局，一对两局取半分
（两胜 1.0 / 一胜一负 0.5 / 两负 0.0），跨对取平均并给标准误与 95% CI。
9x9 连五先手优势很大，普通"黑白各半"统计会被颜色效应淹没（自己打自己也会是"执黑 N 胜、执白 0 胜"），
成对开局把它无论方向都抵消掉。

对手：

* `heuristic`：先赢再堵 + 邻域打分（弱基线，会被斜向双威胁轻易杀死）；
* `tactical`：显式化解"活三 / 双威胁"的规则对手，**没有参与任何训练**，作为 held-out 对手；
* `random`：均匀随机，只作为"没有崩坏"的哨兵；
* `az`：另一个 checkpoint（`--opponent-checkpoint`），自战对比。

⚠️ **规则必须与 checkpoint 一致**：棋盘大小与连子数取自 checkpoint 的 `config`
（`board_size` / `win_length`）并显式传给对局环境。若沿用"连四"这类其它默认值，
网络会在**另一套规则**里被评估，数字没有意义（`tests/test_az_baselines.py`
用 `_DeterministicAgent` 记录 `env.win_length` 把这条钉死）。

## 已知限制

* 未实现 Renju 禁手；长连算胜（自由规则）。
* 评估对手全是规则型：它们弱于人类高手，得分率高不等于棋力强，只能用于版本间对比。
* 单机单进程自我对弈，未做并行/分布式；CPU 上大规模训练很慢（用 `--device mps|cuda`）。
* `arena.SearchAgent` 依赖 `search.py`（本交付不含），构造时会给出明确的 `RuntimeError`。

## 参考资料

* [AlphaZero (Silver et al., 2017)](https://arxiv.org/abs/1712.01815) —— PUCT、`π ∝ N^{1/τ}`、`L = (z-v)² - πᵀlog p`
* [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) —— 五子棋 + MCTS 自对弈的经典参考实现
* [whyb/Gomoku-AI](https://github.com/whyb/Gomoku-AI) —— 历史快照对手池的思路
* [Chess Programming: Match Statistics](https://www.chessprogramming.org/Match_Statistics) —— 成对开局协议与置信区间

## License

MIT
