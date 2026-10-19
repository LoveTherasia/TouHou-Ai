"""从回放采集的 (obs, action) 做行为克隆 —— 学真人操作。

数据由 `tools/record_replay.py --bc-out` 采集: 逐帧 (观测, 由真人位移反推的动作)。

    python tools/bc_from_replay.py --data .runtime/bc/th11_replay_bc.npz `
        --out .runtime/bc_policy --epochs 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from touhou_rl_mcp.training.policy import MLPPolicy  # noqa: E402
from touhou_rl_mcp.training.trainer import build_action_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--load", default="")
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    d = np.load(args.data)
    X = np.asarray(d["obs"], dtype=np.float64)
    A = np.asarray(d["action"], dtype=np.int64)
    n_actions = len(build_action_table(use_focus=True))
    print("[bc] 数据: obs=%s action=%s 动作数=%d" % (X.shape, A.shape, n_actions), flush=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * args.val_split)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Atr = X[tr_idx], A[tr_idx]
    Xva, Ava = X[val_idx], A[val_idx]

    hidden = tuple(int(x) for x in args.hidden.split(","))
    if args.load and Path(args.load).with_suffix(".npz").exists():
        policy = MLPPolicy.load(Path(args.load).with_suffix(".npz"))
        print("[bc] 载入", args.load, flush=True)
    else:
        policy = MLPPolicy(X.shape[1], n_actions, hidden=hidden, seed=args.seed,
                           lr=args.lr, entropy_coef=0.0)
    policy.lr = args.lr
    policy.entropy_coef = 0.0

    for epoch in range(args.epochs):
        order = rng.permutation(len(Xtr))
        last = None
        for s in range(0, len(Xtr), args.minibatch):
            b = order[s:s + args.minibatch]
            last = policy.imitate_batch(Xtr[b], Atr[b])
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logp, _, _ = policy.evaluate(Xva, Ava)
            acc = float((policy.logits_batch(Xva).argmax(1) == Ava).mean())
            print("[bc] epoch %3d  loss=%.4f  val_acc=%.3f  entropy=%.3f"
                  % (epoch, -float(logp.mean()), acc, last["entropy"] if last else 0.0),
                  flush=True)
    p = policy.save(Path(args.out))
    print("[bc] 保存:", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
