"""AlphaZero checkpoint 的棋力评估：成对开局协议，对阵三个规则对手 / 另一个 AZ checkpoint。

评估协议
--------
用 **成对开局**（paired openings，象棋/围棋引擎评测的标准手段）：
同一个随机开局下让被测方先手、后手各下一局，一对两局取半分。
9x9 连五的先手优势很大，普通"黑白各半"统计会被颜色效应淹没（自己打自己也能是
"执黑 N 胜、执白 0 胜"），成对开局把它无论方向都抵消掉，得到的分数才是棋力差。

对手
----
* ``heuristic``：先赢再堵 + 邻域打分（会被斜向双威胁轻易杀死，只作弱基线）
* ``tactical``：显式化解"活三 / 双威胁"的规则对手，**没有参与任何训练**，
  作为 held-out 对手防止"只对训练内对手过拟合"
* ``random``：均匀随机，只作为"没有崩坏"的哨兵
* ``az``：另一个 AlphaZero checkpoint（``--opponent-checkpoint``），自战对比

⚠️ **规则必须与 checkpoint 一致**：棋盘大小与连子数取自 checkpoint 里的 ``config``
（``board_size`` / ``win_length``），而不是写死。用连四规则评估连五网络得到的数字没有意义。

用法::

    uv run python az_eval.py --checkpoint checkpoints/az9x9w5/best.pt \
        --sims 100 --pairs 40 --opponents heuristic tactical random
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alphazero import PolicyValueNet
from arena import HeuristicAgent, RandomAgent
from az_baselines import TacticalAgent, paired_eval
from az_train import AZAgent


def load_az(path: str, device: torch.device, sims: int, leaf_batch: int, c_puct: float = 1.5,
            temperature: float = 0.0) -> tuple[AZAgent, dict]:
    """加载 AlphaZero checkpoint，返回 ``(MCTS 驱动的 Agent, 规则配置)``。"""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    board_size = int(cfg.get("board_size", 9))
    win_length = int(cfg.get("win_length", 5))
    net = PolicyValueNet(
        board_size=board_size,
        in_planes=int(ckpt.get("in_planes", 6)),
        channels=int(cfg.get("channels", 96)),
        blocks=int(cfg.get("blocks", 5)),
        value_channels=int(cfg.get("value_channels", 32)),
        value_hidden=int(cfg.get("value_hidden", 128)),
        shared_trunk=bool(cfg.get("shared_trunk", False)),
    ).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(
        f"[load] {path} iteration={ckpt.get('iteration', '?')} "
        f"{board_size}x{board_size} 连{win_length} "
        f"通道={cfg.get('channels')} 残差块={cfg.get('blocks')} "
        f"共享主干={cfg.get('shared_trunk', False)} 参数={sum(p.numel() for p in net.parameters()):,}"
    )
    agent = AZAgent(
        net, name=f"az{ckpt.get('iteration', '?')}", device=device, sims=sims,
        leaf_batch=leaf_batch, c_puct=c_puct, temperature=temperature,
    )
    return agent, {"board_size": board_size, "win_length": win_length}


def build_opponent(kind: str, seed: int):
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if kind == "tactical":
        return TacticalAgent(seed=seed)
    if kind == "random":
        return RandomAgent(seed=seed)
    raise ValueError(f"未知对手 {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero checkpoint 的成对开局评估")
    parser.add_argument("--checkpoint", default="checkpoints/az9x9w5/best.pt")
    parser.add_argument("--opponent-checkpoint", default="", help="另一个 AZ checkpoint（--opponents az）")
    parser.add_argument(
        "--opponents", nargs="+", default=["heuristic", "tactical", "random"],
        choices=["heuristic", "tactical", "random", "az"],
    )
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--plies", type=int, default=2, help="随机开局手数")
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--eval-sims", type=int, default=-1, help="对手若是 AZ，用它的默认 sims")
    parser.add_argument("--leaf-batch", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint 不存在: {args.checkpoint}")
    agent, rules = load_az(args.checkpoint, device, args.sims, args.leaf_batch, args.c_puct)

    report: dict = {
        "checkpoint": args.checkpoint,
        "board_size": rules["board_size"],
        "win_length": rules["win_length"],
        "sims": args.sims,
        "pairs": args.pairs,
        "plies": args.plies,
        "seed": args.seed,
        "matches": {},
    }
    for kind in args.opponents:
        if kind == "az":
            if not args.opponent_checkpoint:
                raise SystemExit("--opponents az 需要 --opponent-checkpoint")
            opponent, _rules = load_az(
                args.opponent_checkpoint, device,
                args.eval_sims if args.eval_sims > 0 else args.sims, args.leaf_batch, args.c_puct,
            )
        else:
            opponent = build_opponent(kind, args.seed + 1)
        # 关键：棋盘大小与连子数按 checkpoint 的规则显式传入，避免"用连四评估连五网络"
        result = paired_eval(
            agent, opponent, pairs=args.pairs, plies=args.plies, seed=args.seed,
            board_size=rules["board_size"], win_length=rules["win_length"],
        )
        report["matches"][kind] = {
            "score": round(result["score_rate"], 4),
            "stderr": round(result["stderr"], 4),
            "ci95": [round(x, 4) for x in result["ci95"]],
            "W/D/L": f"{result['pairs_won']}/{result['pairs_drawn']}/{result['pairs_lost']}",
            "avg_length": round(result["avg_length"], 1),
        }
        print(
            f"{agent.name} vs {kind:>9} | 得分率 {result['score_rate']:.3f} ± {result['stderr']:.3f} "
            f"95%CI=[{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}] "
            f"W/D/L={result['pairs_won']}/{result['pairs_drawn']}/{result['pairs_lost']} "
            f"平均 {result['avg_length']:.1f} 手",
            flush=True,
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
