"""``python -m forge.store <command>`` - inspect, verify and maintain an artifact store.

Commands: ``stats``, ``ls``, ``verify``, ``gc``, ``pin``, ``unpin``, ``tier``, ``lineage``, ``ckpt``,
``lossless`` (verify / measure a ``.safetensors`` file or ``.flc`` container).  Exit status 1 when a verification fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from forge.core.util import human_bytes


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{k}: {v if not isinstance(v, (dict, list)) else json.dumps(v, default=str)}")
    else:
        print(obj)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m forge.store", description=__doc__)
    p.add_argument("--root", help="store root (default: $FORGE_HOME/store)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats", help="logical vs unique bytes, dedup ratio, tiers")
    ls = sub.add_parser("ls", help="list artifacts")
    ls.add_argument("--kind")
    ls.add_argument("--limit", type=int, default=50)
    v = sub.add_parser("verify", help="verify integrity (all rooted artifacts when no hash given)")
    v.add_argument("hashes", nargs="*")
    v.add_argument("--shallow", action="store_true", help="do not re-hash member objects")
    g = sub.add_parser("gc", help="mark-and-sweep garbage collection (dry run unless --apply)")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--grace", type=float, default=300.0, help="skip objects touched in the last N seconds")
    g.add_argument("--strict-pins", action="store_true", help="treat only pins as roots")
    for name in ("pin", "unpin"):
        s = sub.add_parser(name)
        s.add_argument("hash")
        if name == "pin":
            s.add_argument("--reason", default="cli")
    t = sub.add_parser("tier", help="move an artifact between hot/warm/cold (addresses never change)")
    t.add_argument("hash")
    t.add_argument("to", choices=["hot", "warm", "cold"])
    ln = sub.add_parser("lineage")
    ln.add_argument("hash")
    ln.add_argument("--depth", type=int)
    c = sub.add_parser("ckpt", help="checkpoint series")
    c.add_argument("run", nargs="?")
    c.add_argument("--verify", action="store_true")
    lo = sub.add_parser("lossless", help="verify/measure the lossless codec on a file")
    lo.add_argument("path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "lossless":
        from forge.store.lossless import verify_file

        r = verify_file(args.path)
        r.pop("per_tensor", None) if not args.json else None
        _print(r, args.json)
        return 0
    from forge.store import ArtifactStore, get_store

    st = ArtifactStore(args.root) if args.root else get_store()
    rc = 0
    if args.cmd == "stats":
        s = st.stats()
        if not args.json:
            print(f"logical {human_bytes(s['logical_bytes'])}  unique {human_bytes(s['unique_bytes'])}  "
                  f"dedup x{s['dedup_ratio']:.2f}  compaction {s['compaction_ratio']:.3f}")
        _print(s, args.json)
    elif args.cmd == "ls":
        rows = st.list_artifacts(args.kind, limit=args.limit)
        if args.json:
            _print(rows, True)
        else:
            for r in rows:
                print(f"{r['hash']}  {r['kind']:<12} {human_bytes(r['size']):>10}  {r['tier']:<5} {'PIN' if r['pinned'] else ''}")
    elif args.cmd == "verify":
        if args.hashes:
            res = {h: st.check(h, deep=not args.shallow) for h in args.hashes}
            rc = 0 if all(not v for v in res.values()) else 1
            _print({"ok": rc == 0, "problems": {k: v for k, v in res.items() if v}}, args.json)
        else:
            r = st.verify_all(deep=not args.shallow)
            rc = 0 if r["ok"] else 1
            _print(r, args.json)
    elif args.cmd == "gc":
        rep = st.gc(dry_run=not args.apply, grace_s=args.grace, strict_pins=args.strict_pins)
        _print(rep.to_dict(), args.json)
    elif args.cmd == "pin":
        st.pin(args.hash, args.reason)
    elif args.cmd == "unpin":
        st.unpin(args.hash)
    elif args.cmd == "tier":
        _print({"moved": st.demote(args.hash, args.to)}, args.json)
    elif args.cmd == "lineage":
        _print(st.lineage(args.hash, args.depth), True)
    elif args.cmd == "ckpt":
        from forge.store.checkpoints import CheckpointReader, list_checkpoints

        rows = list_checkpoints(st, args.run, with_sizes=True)
        if args.verify and args.run:
            r = CheckpointReader(st, args.run).verify_all()
            rc = 0 if r["ok"] else 1
            _print(r, args.json)
        else:
            _print(rows, True) if args.json else [print(f"{r['run_id']} step {r['step']:>8} {'KF' if r['is_keyframe'] else 'd '} "
                                                       f"{human_bytes(r['logical_bytes']):>10} -> {human_bytes(r.get('stored_bytes') or 0):>10} {r['tags']}")
                                                 for r in rows]
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
