"""邀请码管理工具：生成 / 查看 / 作废一次性邀请码。

在服务器上、仓库根目录执行（需与服务使用同一个 DATA_DIR）：
  python -m demo.invites new -n 5 --note "给内测用户"   生成 5 个新码（明文只显示这一次）
  python -m demo.invites list                          查看全部（未使用 / 已使用 / 已作废）
  python -m demo.invites revoke 3                      作废第 3 号（仅限未使用的）

每个码只能注册一个账号，用过即失效；库里只存哈希，所以忘记没发出去的码只能重新生成。
"""
from __future__ import annotations

import argparse
import sys

from demo import accounts, settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m demo.invites", description="邀请码管理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="生成邀请码")
    p_new.add_argument("-n", "--count", type=int, default=1, help="生成个数（1~200，默认 1）")
    p_new.add_argument("--note", default="", help="备注，如发给谁（最多 64 字）")

    sub.add_parser("list", help="查看全部邀请码状态")

    p_rev = sub.add_parser("revoke", help="作废一个未使用的邀请码")
    p_rev.add_argument("id", type=int)

    args = ap.parse_args(argv)

    if args.cmd == "new":
        if not 1 <= args.count <= 200:
            ap.error("--count 需在 1~200 之间")
        codes = accounts.generate_invites(args.count, args.note)
        print(f"已生成 {len(codes)} 个邀请码（数据目录：{settings.DATA_DIR}）：")
        for c in codes:
            print(f"  {c}")
        print("明文只显示这一次，请当场保存；每个码只能使用一次。")
        return 0

    if args.cmd == "list":
        rows = accounts.list_invites()
        if not rows:
            print("还没有邀请码。用 `python -m demo.invites new -n 5` 生成。")
            return 0
        print(f"{'ID':>4}  {'状态':<6}  {'创建时间':<16}  {'使用者':<16}  备注")
        for r in rows:
            print(f"{r['id']:>4}  {r['status']:<6}  {r['created_at']:<16}  {r['used_by']:<16}  {r['note']}")
        print(f"未使用：{sum(1 for r in rows if r['status'] == '未使用')} / 共 {len(rows)}")
        return 0

    if args.cmd == "revoke":
        if accounts.revoke_invite(args.id):
            print(f"已作废 #{args.id}")
            return 0
        print(f"#{args.id} 不存在或已被使用，无法作废", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
