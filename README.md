# naPINN: Prediction Videos

This page provides qualitative prediction videos for **naPINN**
(**Noise-Adaptive Physics-Informed Neural Networks**), a framework for robustly recovering physical dynamics from corrupted measurement data.

naPINN is designed for measurement-driven inverse PDE problems where observations may contain complex non-Gaussian noise and gross outliers. The model estimates residual reliability during training and uses a trainable reliability gate to downweight unreliable measurements, enabling more robust reconstruction of the underlying physical solution.

## Video Results

The following videos show model prediction results on three benchmark PDE systems.

### 1. 2D Allen–Cahn Equation

The Allen–Cahn benchmark evaluates whether naPINN can recover scalar field dynamics from sparse and corrupted measurements.

[Watch video](VIDEO_LINK_1)

---

### 2. 2D Burgers’ Equation

The Burgers benchmark evaluates reconstruction of nonlinear convection–diffusion dynamics with two coupled state variables.

[Watch video](VIDEO_LINK_2)

---

### 3. 2D λ–ω Reaction–Diffusion System

The λ–ω reaction–diffusion benchmark evaluates whether naPINN can recover complex spatiotemporal pattern formation under corrupted measurements.

[Watch video](VIDEO_LINK_3)

---

## Overview

Physics-Informed Neural Networks (PINNs) are effective for solving inverse problems and discovering governing equations from observational data. However, standard PINNs can be highly sensitive to corrupted measurements because unreliable data points may dominate the data-fitting loss.

naPINN addresses this limitation by introducing:

* **Residual-based noise distribution estimation**
* **Trainable reliability gating**
* **Adaptive downweighting of unreliable measurements**
* **Rejection-cost regularization to prevent excessive data rejection**

Through this mechanism, naPINN can suppress corrupted observations while preserving valid measurement data, leading to more accurate reconstruction of physical dynamics.

## Benchmarks

The qualitative videos correspond to the three PDE benchmarks used in the study:

| Benchmark                 | Description                                         |
| ------------------------- | --------------------------------------------------- |
| 2D Allen–Cahn             | Phase separation and interface dynamics             |
| 2D Burgers                | Nonlinear convection–diffusion dynamics             |
| 2D λ–ω Reaction–Diffusion | Spatiotemporal reaction–diffusion pattern formation |

## Experimental Setting

The videos visualize model predictions under sparse measurement conditions with corrupted observations. Measurements are contaminated by complex non-Gaussian noise and gross outliers, while the model is required to reconstruct the clean physical solution.

In the experiments, naPINN is compared with standard and robust PINN baselines, including Vanilla PINN, B-PINN, LAD-PINN, and OrPINN. Across the evaluated benchmarks, naPINN shows improved robustness under increasing outlier ratios and produces reconstructions that more closely match the reference physical dynamics.

## Citation

If you find this work useful, please cite:

**naPINN: Noise-Adaptive Physics-Informed Neural Networks for Recovering Physics from Corrupted Measurement**

```bibtex
@inproceedings{napinn2026,
  title={naPINN: Noise-Adaptive Physics-Informed Neural Networks for Recovering Physics from Corrupted Measurement},
  author={Anonymous},
  booktitle={Submitted to NeurIPS},
  year={2026}
}
```

## Contact

For questions or further information, please contact:

**[Your Name]**
**[Your Email]**

Presented at **[Conference Name]**.
