"""Broad-protocol deploy for any train_v3 run json (select on VAL+seen on the select block, test on the other block)."""
import json, sys
for t in sys.argv[1:]:
    r = json.load(open(f"runs/{t}.json")); best = {}
    for f, (sel, test) in {"A": ("Q3", "Q4"), "B": ("Q4", "Q3")}.items():
        c = max(((32 * h["eval"][sel][k]["seen"][0] + 7 * h["eval"][sel][k]["val_zs"][0]) / 39, h["epoch"], k) for h in r["history"] for k in h["eval"][sel] if k.startswith("ctx|"))
        e = r["history"][c[1]]["eval"][test][c[2]]; best[f] = ((32 * e["seen"][0] + 7 * e["eval_zs"][0]) / 39, e["seen"][0], e["eval_zs"][0], c[1], c[2])
    s0 = r["summary"]["deploy"] if r["args"].get("deep") else float("nan")
    print(f"{t}: bf {r['args']['broad_frac']} ctx {r['args'].get('ctx_layers', 2)} ep {r['args']['epochs']} | subj0 {s0:.3f} | Broad deploy A {best['A'][0]:.3f} B {best['B'][0]:.3f} mean {(best['A'][0] + best['B'][0]) / 2:.3f} (seen {best['A'][1]:.3f}/{best['B'][1]:.3f} zs {best['A'][2]:.3f}/{best['B'][2]:.3f}; sel ep {best['A'][3]}/{best['B'][3]} {best['A'][4]}/{best['B'][4]})")
