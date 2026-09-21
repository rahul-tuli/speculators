# Synthetic figure prototype

Question: which acceptance layouts, aggregation rules, and export contract should the drafter quantization experiment use?

**Review pending. Every number is synthetic. No model or evaluation was run.**

Open `index.html` to switch A/B/C, or view the PNG/PDF/SVG files. See `schema-proposal.md` for the proposed contract. Rebuild with `python local/drafter-quant/wayfinder/plot-prototype/prototype.py`.

This is a throwaway asset for [Agree on plots and aggregation that answer the experiment questions](https://github.com/rahul-tuli/speculators/issues/107), preserved on `prototype/drafter-quant-plot-layouts`. Production plotting implementation belongs to later execution work after the human decision.

Validation: all five figures rendered in PNG/PDF/SVG; lead, recipe-pair and performance previews inspected visually; embedded JavaScript passed `node --check`. The repository-wide mypy hook checked unchanged `src/speculators` and `tests` and failed with 14 errors involving missing `hs_connectors` exports and missing `tqdm`/`yaml` stubs. It does not check this standalone prototype. The capture commit skips that unrelated failing hook, retaining Ruff and Markdown checks.

The preview requires Python, NumPy and Matplotlib to regenerate, and a browser to switch layouts. Generated files need no Python installation to view. The HTML stores no persistent state beyond its shareable `?variant=` parameter.
