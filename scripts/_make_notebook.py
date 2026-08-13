#!/usr/bin/env python3
"""Generate notebooks/run_all.ipynb."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    def md(s: str):
        return nbf.v4.new_markdown_cell(s)

    def code(s: str):
        return nbf.v4.new_code_cell(s)

    nb.cells = [
        md(
            "# LightShap — run all\n\n"
            "User-facing orchestrator. Scientific logic lives in `src/lightshap/`, "
            "not here. Re-running this notebook is safe: completed stages are skipped "
            "when their artifacts still validate.\n\n"
            "**Estimand.** Inference-time extra-hop inclusion on one frozen uniform "
            "`K=3`, `d=64` LightGCN. Skip patterns are cache scoring rules. The flag "
            "is not a prune claim. Beauty is **2014** 5-core. `I_12 = 1/2`, never `1`."
        ),
        md("## 1. Environment"),
        code(
            "from pathlib import Path\n"
            "import json, sys, platform\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'pyproject.toml').exists() and (ROOT.parent / 'pyproject.toml').exists():\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from lightshap import __version__\n"
            "from lightshap.constants import (\n"
            "    FIXTURE_PHI, FIXTURE_I, LIGHTGCN_FROZEN, STABILITY_FLOOR,\n"
            "    N_VAL_ALPHA, BEAUTY_YEAR, FORBIDDEN_I12,\n"
            ")\n"
            "from lightshap.config import load_experiment_config\n"
            "from lightshap.pipeline.stages import STAGE_ORDER\n"
            "from lightshap.pipeline.runner import PipelineRunner\n"
            "from lightshap.pipeline.state import load_state\n"
            "print('lightshap', __version__)\n"
            "print('python', sys.version.split()[0], platform.platform())\n"
            "print('repo', ROOT)\n"
            "print('stages', len(STAGE_ORDER))\n"
            "from lightshap.utils.device import resolve_device\n"
            "print('device', resolve_device('auto'))\n"
            "try:\n"
            "    import torch\n"
            "    mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())\n"
            "    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', mps)\n"
            "except Exception as exc:\n"
            "    print('torch missing:', exc)\n"
        ),
        md("## 2. Frozen invariants (must not be edited for convenience)"),
        code(
            "print('LightGCN', LIGHTGCN_FROZEN)\n"
            "print('stability floor', STABILITY_FLOOR)\n"
            "print('val-alpha candidates', N_VAL_ALPHA)\n"
            "print('fixture phi', FIXTURE_PHI, 'I', FIXTURE_I, 'forbidden I12', FORBIDDEN_I12)\n"
            "print('Beauty year', BEAUTY_YEAR)\n"
            "assert abs(FIXTURE_I - 0.5) < 1e-15\n"
            "assert abs(FIXTURE_I - 1.0) > 1e-9\n"
        ),
        md(
            "## 3. Configuration\n\n"
            "On a Mac M4 Pro start with `synthetic`, then set `PROFILE = 'experiment'` "
            "for ML-1M + Beauty 2014. `device: auto` selects MPS on Apple Silicon. "
            "Paste SHA256s from `python scripts/fetch_data.py` before a scientific run."
        ),
        code(
            "# synthetic_fast | synthetic | experiment\n"
            "PROFILE = 'synthetic'\n"
            "CONFIG = ROOT / 'configs' / f'{PROFILE}.yaml'\n"
            "cfg = load_experiment_config(CONFIG, project_root=ROOT)\n"
            "print('profile', cfg.profile)\n"
            "print('config', CONFIG)\n"
            "print('datasets', cfg.datasets)\n"
            "print('seeds', cfg.seeds)\n"
            "print('neg_pool', cfg.neg_pool)\n"
            "print('device requested', cfg.device, '->', resolve_device(cfg.device))\n"
            "print('config_hash', cfg.config_hash())\n"
            "print('LightGCN', cfg.lightgcn)\n"
        ),
        md("## 4. Dataset / run status"),
        code(
            "runs_root = ROOT / 'results' / 'runs'\n"
            "existing = sorted(p.name for p in runs_root.glob('*') if p.is_dir()) if runs_root.exists() else []\n"
            "print('existing runs:', existing[-5:])\n"
            "processed = ROOT / 'data' / 'processed'\n"
            "print('processed datasets:', [p.name for p in processed.glob('*')] if processed.exists() else [])\n"
        ),
        md(
            "## 5. Run the pipeline\n\n"
            "The runner refuses to execute a stage whose dependencies are missing. "
            "Resume semantics: completed+valid stages are not recomputed."
        ),
        code(
            "RESUME = True  # safe re-run\n"
            "runner = PipelineRunner(cfg)\n"
            "print('run_id', runner.run_id)\n"
            "print('run_dir', runner.run_dir)\n"
            "status = runner.run(resume=RESUME)\n"
            "print('finished', status['run_id'])\n"
        ),
        md("## 6. Checkpoint / stage table"),
        code(
            "st = load_state(runner.run_dir)\n"
            "for name, rec in st.stages.items():\n"
            "    err = f'  ({rec.error})' if rec.error else ''\n"
            "    print(f'{name:24} {rec.state}{err}')\n"
        ),
        md("## 7. Results: RQ1–RQ4"),
        code(
            "def loadj(rel):\n"
            "    p = runner.run_dir / rel\n"
            "    return json.loads(p.read_text()) if p.is_file() else None\n"
            "\n"
            "rq1 = loadj('rq1/rq1.json')\n"
            "rq2 = loadj('rq2/rq2.json')\n"
            "rq3 = loadj('rq3/rq3.json')\n"
            "rq4 = loadj('rq4/rq4.json')\n"
            "shap = loadj('shapley/summary.json')\n"
            "\n"
            "if shap:\n"
            "    for ds, rows in shap.items():\n"
            "        print('==', ds, 'Game A ==')\n"
            "        for row in rows:\n"
            "            print(' seed', row['seed'], 'phi', row['phi'], 'v(L)', row['v'].get('(1, 2, 3)'))\n"
            "            print(' flag any', row['flag']['any_pair_flagged'], 'undefined', row['flag']['undefined'])\n"
            "            print(' Game B trigger', row['game_b_trigger']['triggered'])\n"
            "\n"
            "if rq1:\n"
            "    for ds, rec in rq1.items():\n"
            "        print('RQ1', ds, 'v_L_mean', rec.get('v_L_mean'), 'spearman', rec.get('v_C_spearman_mean_pairwise'))\n"
            "        print('  modal order', rec.get('phi_modal_order'), 'cosine', rec.get('phi_mean_pairwise_cosine'))\n"
            "        print('  note', rec.get('note'))\n"
            "\n"
            "if rq3:\n"
            "    for ds, rec in rq3.items():\n"
            "        print('RQ3', ds, 'mean_delta', rec.get('mean_delta_if_all_eligible'), rec.get('result_state'))\n"
            "\n"
            "if rq4:\n"
            "    for ds, rec in rq4.items():\n"
            "        agg = rec['aggregate']\n"
            "        print('RQ4', ds, agg.get('mean_test_ndcg'), 'holm', agg.get('holm'))\n"
        ),
        md("## 8. Figures"),
        code(
            "from IPython.display import Image, display, Markdown\n"
            "figdir = runner.run_dir / 'figures'\n"
            "figs = sorted(figdir.glob('*.png')) if figdir.exists() else []\n"
            "print(len(figs), 'figures')\n"
            "for p in figs:\n"
            "    display(Markdown(f'**{p.name}**'))\n"
            "    display(Image(filename=str(p)))\n"
        ),
        md("## 9. Warnings and compliance (honest — no fabricated PASS)"),
        code(
            "from collections import Counter\n"
            "summary = loadj('summary.json')\n"
            "print('run status', summary.get('status') if summary else None)\n"
            "print('warnings:')\n"
            "for w in (summary or {}).get('warnings', []) or ['_none_']:\n"
            "    print(' -', w)\n"
            "comp = loadj('spec_compliance.json')\n"
            "if comp:\n"
            "    counts = Counter(v.get('status') for v in comp.values() if isinstance(v, dict) and 'status' in v)\n"
            "    print('compliance counts', dict(counts))\n"
            "    print()\n"
            "    for k, v in sorted(comp.items()):\n"
            "        if not isinstance(v, dict) or 'status' not in v:\n"
            "            continue\n"
            "        print(f\"{v['status']:16} {k:22} {v.get('name', '')}\")\n"
        ),
        md(
            "## 10. Resume after interruption\n\n"
            "Re-run the next cell to continue the same `run_id`. Completed stages stay completed."
        ),
        code(
            "status2 = runner.run(resume=True)\n"
            "print('resume status ok, run_id', status2['run_id'])\n"
        ),
    ]
    dest = ROOT / "notebooks" / "run_all.ipynb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, dest)
    print("wrote", dest, "cells", len(nb.cells))


if __name__ == "__main__":
    main()
