[README.md](https://github.com/user-attachments/files/34823457/README.md)
# How close does a PINN estimate of the fractional order and the diffusivity come to the Cramér–Rao bound?

Code and data for the paper *How Close Does a PINN Estimate of the Fractional Order and the Diffusivity Come to the Cramér–Rao Bound? A paired benchmark against nonlinear least squares, with an a posteriori error bound and documented negative results* (G. Progri, University of Elbasan "Aleksandër Xhuvani", 2026). The manuscript is in `paper/`.

## What the study does

We identify the order α and the diffusivity D of the time-fractional diffusion equation ∂ₜ^α u = D u_xx on (0,1), with zero Dirichlet boundary conditions and a known initial condition, from noisy point measurements. The problem has an exact solution, u(x,t) = Σ cₙ E_α(−D n²π² t^α) √2 sin(nπx), so the best attainable accuracy (the Cramér–Rao bound, CRB) and the best classical estimator (nonlinear least squares, NLS, with the exact solver) are both available. On identical data we compare:

* **NLS** with the exact spectral solver (the reference);
* a **spectral-hybrid PINN**, in which a small network learns only the modal time coefficients aₙ(t^α) and the physics enters through an L1 discretization of the Caputo derivative;
* a **full-network fPINN-style baseline**.

We also compute an a posteriori **Gauss–Newton error bound** for the parameter estimate (it needs the exact solver) and a bound from the network's **Volterra defect** alone (vacuous with our training), and we localize the small deterministic offset of the PINN (Section 5.9 of the paper).

True parameters: θ* = (α, D) = (0.6, 0.5); initial condition with three modes c = (1, 0.5, 0.25). Measurements at m ∈ {10, 20, 40} random points, x ∈ [0.1, 0.9], t ∈ [0.05, 1], Gaussian noise σ ∈ {0.01, 0.03}.

## Main results (see the paper for confidence intervals and caveats)

| Result | Value |
|---|---|
| NLS / CRB | within 15% in every configuration (within 7% for m ≥ 20) |
| Spectral PINN / NLS, σ = 0.01 | 1.1–1.4 (90 validation runs, 0 failures) |
| Spectral PINN / NLS, σ = 0.03, m = 20, 3 starts fixed in advance | 0.93 [0.82, 1.01] for α (30 fresh seeds) |
| Deterministic offset of the PINN (noise-free data) | α −0.0042 ± 0.0003, D +0.0023 ± 0.0001 |
| Gauss–Newton bound, observed coverage | 100% of 130 runs (σ = 0.01, m = 40, χ² variant), bound 2–3× the error |
| Bound from the network alone | vacuous (defects 3–300× above the ~10⁻³ needed) |
| fPINN baseline (only the physics weight tuned) | α error about 5× the spectral PINN's (12 trials) |

## Repository layout

```
code/
  frac_identification_baseline.py  exact spectral solver (Mittag-Leffler by quadrature), Jacobian, NLS
  pinn_spectral.py                 spectral-hybrid PINN (SpecPINN, train, L1 residual, Volterra defect)
  pinn_alpha_D.py                  first full-MLP PINN with L1 Caputo (early failure, Section 5.2)
  fpinn_baseline.py                fPINN-style full-network baseline
  fpinn_diag.py                    forward-problem diagnostic for the fPINN (Table 7)
  experiment_B.py                  experiment driver: NLS + CRB + PINN per trial, CSV with resume
  compare_runs.py                  paired bootstrap comparison, grouped by (sigma, m)
  merge_csv.py                     merge CSV files
  certificate_C.py                 Gauss–Newton bound with the exact solver (Section 5.6)
  network_certificate.py           bound from the Volterra defect of the network (Section 5.7)
  bias_emulators.py                Section 5.9: same discrete loss without the network (NumPy/SciPy)
  analyze_bias.py                  Section 5.9: reproduces Tables 9 and 10 from the CSV files
results/
  nls_crb_validation/  ablations/  multistart_sigma003/  volterra/  fpinn/  bias_analysis/
paper/
  paper_PINN_fractional_CRB_FINAL.docx
```

CSV columns: `sigma, m, trial, nls_a, nls_D, crb_a, crb_D, pinn_a, pinn_D, pinn_pde_rms, pinn_misfit, pinn_time, pinn_start` and, in files produced by the current version, `code_version` (short git commit or SHA-256 of the core source files).

## Use of the trials (seeds)

The data of a trial are generated with the seed `1000*trial + m`, so NLS, the CRB and every PINN variant see exactly the same data.

| Purpose | Trials (σ, m) | Files |
|---|---|---|
| Tuning and ablations (Table 2) | 0–19 (0.01, 40) | `results/ablations/abl_*.csv` |
| Diagnosis of the bias (Section 5.9) | 0–19 noisy, 0–9 noise-free (0.01, 40) | `results/bias_analysis/*.csv` |
| Validation, NLS vs CRB (Tables 1, 3) | 100–129 (0.01; 10, 20, 40) | `results/nls_crb_validation/val_m*.csv` |
| Sensitivity to the start (Table 4) | 100–129 (0.03; 20, 40) | `results/multistart_sigma003/val_s003_*.csv` |
| Volterra loss (Section 5.5) | 100–119 (0.01, 40) | `results/volterra/val_volterra.csv` |
| Confirmation, starts fixed in advance | 200–229 (0.03, 20) | `results/multistart_sigma003/fresh_*.csv` |
| fPINN baseline (Tables 7, 8) | 200–229; fair run 200–211 (0.03, 20) | `results/fpinn/*.csv` |

The training configuration was chosen on trials 0–19. The first multistart set (0.4, 0.5, 0.7) was chosen after inspecting trials 100–129 at σ = 0.03 (assembled from the three single-start files `val_s003_m20.csv`, `_a04`, `_a07`); for that reason it was confirmed on trials 200–229 with the starts (0.3, 0.5, 0.8) fixed in advance.

## Installation

Python ≥ 3.10. Training runs on the CPU in float64 (the networks are tiny).

```
pip install -r requirements.txt
```

## Reproducing the results

All commands are run from `code/`. On Windows, set `OMP_NUM_THREADS=1` and run several processes in parallel with `start cmd /k ...`; on Linux/macOS use `export OMP_NUM_THREADS=1` and `&`. Times are for one CPU core per process (spectral PINN ≈ 165 s per start, fPINN ≈ 1700 s per start).

```
:: Table 1 and validation sigma = 0.01 (Table 3)
python experiment_B.py --offset 100 --trials 30 --ms 10 --wfinal 1e6 --skip 0.05 --out val_m10.csv
python experiment_B.py --offset 100 --trials 30 --ms 20 --wfinal 1e6 --skip 0.05 --out val_m20.csv
python experiment_B.py --offset 100 --trials 30 --ms 40 --wfinal 1e6 --skip 0.05 --out val_m40.csv

:: NLS and CRB only (no PyTorch needed)
python experiment_B.py --no-pinn --offset 100 --trials 30 --ms 10 20 40 --out nls_only.csv

:: Ablations (Table 2), trials 0-19, single start
python experiment_B.py --trials 20 --ms 40 --out abl_base.csv
python experiment_B.py --trials 20 --ms 40 --wfinal 1e6 --skip 0.05 --out abl_w1e6_s005.csv

:: Sensitivity to the start and confirmation with multistart (Table 4)
python experiment_B.py --offset 100 --trials 30 --ms 20 --sigmas 0.03 --alpha0 0.7 --wfinal 1e6 --skip 0.05 --out val_s003_m20_a07.csv
python experiment_B.py --offset 200 --trials 30 --ms 20 --sigmas 0.03 --starts 0.3 0.5 0.8 --wfinal 1e6 --skip 0.05 --out fresh_all.csv

:: Volterra loss (Section 5.5)
python experiment_B.py --offset 100 --trials 20 --ms 40 --res volterra --wfinal 1e4 --epochs 3000 --out val_volterra.csv

:: fPINN: forward diagnostic (Table 7) and fair comparison (Table 8)
python fpinn_diag.py --feature talpha --wfinal 1e3 --m 20 --sigma 0.03
python experiment_B.py --model fpinn --feature talpha --offset 200 --trials 12 --ms 20 --sigmas 0.03 --starts 0.3 0.5 0.8 --wfinal 1e3 --skip 0.05 --out fp1e3_all.csv

:: Bias diagnosis (Section 5.9)
python experiment_B.py --trials 20 --ms 40 --wfinal 1e6 --skip 0.02 --out bias_s002.csv
python experiment_B.py --trials 20 --ms 40 --wfinal 1e6 --skip 0.05 --M 400 --out bias_s005_M400.csv
python experiment_B.py --noise-scale 0 --trials 10 --ms 40 --skip 0.02 --wfinal 1e5 --out nf_w1e5.csv
python experiment_B.py --noise-scale 0 --trials 10 --ms 40 --skip 0.02 --wfinal 1e5 --width 128 --depth 4 --lbfgs 3000 --out nf_both.csv
python experiment_B.py --noise-scale 0 --trials 10 --ms 40 --skip 0.02 --wfinal 1e5 --alpha0 0.6 --D0 0.5 --out nf_init_true.csv
python bias_emulators.py --rep grid --trials 0 1 2
python bias_emulators.py --rep cheb --trials 0 1 2
python bias_emulators.py --rep cheb --trials 100 101 102 --noise 1 --cert

:: Analysis of existing files
python compare_runs.py ../results/multistart_sigma003/fresh_all.csv ../results/fpinn/fp1e3_all.csv
python experiment_B.py --summarize ../results/nls_crb_validation/val_m40.csv
python certificate_C.py ../results/nls_crb_validation/val_m40.csv
cd .. && python code/analyze_bias.py
```

Note: `experiment_B.py` skips trials already present in the output CSV (resume). Delete an output file before re-running a configuration into the same name.

## Limitations

One θ*, one initial condition with three modes, zero source and known σ. The PINN uses the sine eigenbasis and therefore works under favourable conditions; nothing here is shown for variable coefficients, sources or problems without an exact solver. The coverage of the Gauss–Newton bound is observed, not guaranteed. The fPINN baseline was tuned in one hyperparameter only. See Section 7 of the paper.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).
