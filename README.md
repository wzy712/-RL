# 五子棋 AlphaZero（环境 + 训练）

AlphaZero 范式五子棋：MCTS 搜索产生策略目标、对局结果产生价值目标，策略头与价值头**各自独立主干**。

```
env.py            GomokuEnv：9x9 连五，Gymnasium 接口（4 平面观测，增量胜负判定）
alphazero.py      PUCT-MCTS（置换表 / 批量叶子评估 / 子树复用）+ PolicyValueNet + 6 平面观测
az_train.py       自我对弈训练（回放池 / 对称增强 / 周期评估 / checkpoint）
az_eval.py        成对开局评估（规则对手 / 另一个 checkpoint）
az_baselines.py   自包含的规则对手 + 成对开局协议
arena.py          Agent 协议、对局框架与统计
tests/            回归测试
```

## 运行

需要 Python 3.12–3.13（`pygame` 目前只有到 cp313 的 wheel）。

```bash
uv sync                          # 或 pip install torch numpy gymnasium

# 冒烟：CPU 约 1 分钟，验证整条链路
uv run python az_train.py --run-name az_smoke --iterations 2 --games-per-iter 2 \
    --sims 8 --leaf-batch 2 --channels 16 --blocks 1 --train-steps 2 --eval-interval 0

# 正式训练（9x9 连五；MPS 上每轮约 4 分钟）
uv run python az_train.py --run-name az9x9w5 --iterations 50 --games-per-iter 40 \
    --sims 100 --channels 96 --blocks 5

# 评估：board_size / win_length 自动取自 checkpoint 的 config
uv run python az_eval.py --checkpoint checkpoints/az9x9w5/final.pt --sims 100 \
    --pairs 40 --opponents heuristic tactical random

uv run pytest -q                  # 测试
uv run python az_train.py --help  # 全部超参数（由 AZConfig 自动生成）
```

`--device auto` 自动选 cuda / mps / cpu；`--render` 看 pygame 窗口（需 `uv sync --extra render`）。

## 关键约定

* **环境**：`board_size × board_size`（默认 9）连 `win_length` 子获胜（默认 5）；自由规则
  （长连也算胜），下满为和；观测永远是**当前行棋方视角**；非法动作默认抛 `IllegalMoveError`。
* **观测**：AlphaZero 用 6 平面 = env 原生 4 平面 + "我能立刻连五的点 / 对手能立刻连五的点"。
* **MCTS 战术处理**：对"立刻获胜 / 必须封堵"显式优先，否则纯 PUCT 会因先验太小漏掉必胜手。
* **评估**：成对开局（同一随机开局黑白各下一局）抵消先手优势；规则按 checkpoint 的
  `board_size` / `win_length` 传入——**规则不一致的分数没有意义**。

## License

MIT
