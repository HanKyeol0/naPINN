"""Pressure-latent coordinate PINN for real RealPDEBench PIV measurements.

The legacy Cylinder artifact stores SI-labelled arrays and derives scales from
``Re, nu, D``.  Generic artifacts instead store released-unit coordinates and
explicit characteristic length/velocity scales.  Both paths predict
nondimensional ``(u, v, p)`` from coordinates normalized to ``[-1, 1]^3``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pinnlab.experiments.base import BaseExperiment, grad_sum, make_leaf


class RealPDEBenchCylinder(BaseExperiment):
    """RealPDEBench fluid-PIV nominal Navier--Stokes experiment.

    The historical class name is retained for checkpoint/config compatibility.
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.device = torch.device(device)
        data_path = Path(cfg["data"]["path"])
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Prepared RealPDEBench artifact not found: {data_path}. "
                "Run scripts/rebuttal/prepare_realpdebench_cylinder.py first."
            )
        with np.load(data_path, allow_pickle=False) as raw:
            generic_required = {
                "x_coord",
                "y_coord",
                "t_coord",
                "u_observed",
                "v_observed",
                "fluid_mask",
                "train_sensor_flat_indices",
                "heldout_flat_indices",
                "metadata_json",
            }
            legacy_required = {
                "x_m",
                "y_m",
                "t_s",
                "u_mps",
                "v_mps",
                "fluid_mask",
                "train_sensor_flat_indices",
                "heldout_flat_indices",
                "metadata_json",
            }
            if generic_required.issubset(raw.files):
                keys = {
                    "x": "x_coord",
                    "y": "y_coord",
                    "t": "t_coord",
                    "u": "u_observed",
                    "v": "v_observed",
                    "u_clean": "u_clean",
                    "v_clean": "v_clean",
                }
                required = generic_required
            else:
                keys = {
                    "x": "x_m",
                    "y": "y_m",
                    "t": "t_s",
                    "u": "u_mps",
                    "v": "v_mps",
                    "u_clean": "u_clean_mps",
                    "v_clean": "v_clean_mps",
                }
                required = legacy_required
            missing = required.difference(raw.files)
            if missing:
                raise ValueError(f"Prepared artifact lacks arrays: {sorted(missing)}")
            self.metadata = json.loads(str(raw["metadata_json"].item()))
            self.x_coord = raw[keys["x"]].astype(np.float64, copy=True)
            self.y_coord = raw[keys["y"]].astype(np.float64, copy=True)
            self.t_coord = raw[keys["t"]].astype(np.float64, copy=True)
            self.u_observed = raw[keys["u"]].astype(np.float32, copy=True)
            self.v_observed = raw[keys["v"]].astype(np.float32, copy=True)
            self.u_clean = raw[
                keys["u_clean"] if keys["u_clean"] in raw.files else keys["u"]
            ].astype(np.float32, copy=True)
            self.v_clean = raw[
                keys["v_clean"] if keys["v_clean"] in raw.files else keys["v"]
            ].astype(np.float32, copy=True)
            self.fluid_mask = raw["fluid_mask"].astype(bool, copy=True)
            self.train_sensor_indices = raw[
                "train_sensor_flat_indices"
            ].astype(np.int64, copy=True)
            self.heldout_indices = raw["heldout_flat_indices"].astype(
                np.int64, copy=True
            )
            if "training_corruption_mask" in raw.files:
                self.training_corruption_mask = raw[
                    "training_corruption_mask"
                ].astype(bool, copy=True)
            else:
                self.training_corruption_mask = None

        if self.u_observed.shape != self.v_observed.shape:
            raise ValueError("u and v shapes differ")
        if (
            self.u_clean.shape != self.u_observed.shape
            or self.v_clean.shape != self.v_observed.shape
        ):
            raise ValueError("Clean and observed velocity shapes differ")
        if (
            self.u_observed.shape[1:] != self.x_coord.shape
            or self.x_coord.shape != self.y_coord.shape
        ):
            raise ValueError("Velocity and coordinate grid shapes differ")
        if self.u_observed.shape[0] != self.t_coord.size:
            raise ValueError("Velocity and time dimensions differ")
        overlap = np.intersect1d(
            self.train_sensor_indices, self.heldout_indices, assume_unique=True
        )
        if overlap.size:
            raise ValueError("Training sensors overlap held-out evaluation points")
        valid_indices = np.flatnonzero(self.fluid_mask.reshape(-1))
        if not np.array_equal(
            np.sort(
                np.concatenate(
                    (self.train_sensor_indices, self.heldout_indices)
                )
            ),
            valid_indices,
        ):
            raise ValueError(
                "Training and held-out indices do not partition fluid_mask"
            )
        if not np.array_equal(
            self.u_observed.reshape(self.t_coord.size, -1)[
                :, self.heldout_indices
            ],
            self.u_clean.reshape(self.t_coord.size, -1)[
                :, self.heldout_indices
            ],
        ) or not np.array_equal(
            self.v_observed.reshape(self.t_coord.size, -1)[
                :, self.heldout_indices
            ],
            self.v_clean.reshape(self.t_coord.size, -1)[
                :, self.heldout_indices
            ],
        ):
            raise ValueError("Observed artifact corrupts held-out PIV labels")
        if not all(
            np.isfinite(array).all()
            for array in (
                self.x_coord.reshape(-1)[valid_indices],
                self.y_coord.reshape(-1)[valid_indices],
                self.u_observed.reshape(self.t_coord.size, -1)[
                    :, valid_indices
                ],
                self.v_observed.reshape(self.t_coord.size, -1)[
                    :, valid_indices
                ],
            )
        ):
            raise ValueError(
                "Selected training/held-out domain contains non-finite values"
            )

        self.reynolds = float(self.metadata["reynolds_number"])
        if "characteristic_length_source_units" in self.metadata:
            self.length_scale = float(
                self.metadata["characteristic_length_source_units"]
            )
            self.velocity_scale = float(
                self.metadata["characteristic_velocity_source_units"]
            )
            self.scale_mode = "explicit_released_units"
        else:
            self.length_scale = float(self.metadata["cylinder_diameter_m"])
            kinematic_viscosity = float(
                self.metadata["water_kinematic_viscosity_m2ps"]
            )
            self.velocity_scale = (
                self.reynolds * kinematic_viscosity / self.length_scale
            )
            recorded_velocity = float(
                self.metadata.get(
                    "characteristic_velocity_mps", self.velocity_scale
                )
            )
            if not np.isclose(
                recorded_velocity, self.velocity_scale, rtol=1e-10
            ):
                raise ValueError(
                    "Prepared artifact has inconsistent velocity scale"
                )
            self.scale_mode = "legacy_cylinder_si"
        if (
            not np.isfinite(self.length_scale)
            or self.length_scale <= 0.0
            or not np.isfinite(self.velocity_scale)
            or self.velocity_scale <= 0.0
        ):
            raise ValueError("Characteristic scales must be finite and positive")
        # Compatibility aliases used by the existing runner's metadata path.
        self.diameter_m = self.length_scale
        self.velocity_scale_mps = self.velocity_scale

        pde_cfg = cfg.get("pde", {})
        self.learn_reynolds = bool(pde_cfg.get("learn_reynolds", False))
        if self.learn_reynolds:
            initial_reynolds = float(pde_cfg.get("initial_reynolds", 8000.0))
            if initial_reynolds <= 0:
                raise ValueError("pde.initial_reynolds must be positive")
            self.log_effective_reynolds = torch.nn.Parameter(
                torch.tensor(
                    np.log(initial_reynolds),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        else:
            self.log_effective_reynolds = None

        self.x_star = self.x_coord / self.length_scale
        self.y_star = self.y_coord / self.length_scale
        self.t_star = (
            self.t_coord * self.velocity_scale / self.length_scale
        )
        self.u_star = self.u_observed / self.velocity_scale
        self.v_star = self.v_observed / self.velocity_scale
        self.u_clean_star = self.u_clean / self.velocity_scale
        self.v_clean_star = self.v_clean / self.velocity_scale

        lower = np.asarray(
            [self.x_star.min(), self.y_star.min(), self.t_star.min()],
            dtype=np.float32,
        )
        upper = np.asarray(
            [self.x_star.max(), self.y_star.max(), self.t_star.max()],
            dtype=np.float32,
        )
        if np.any(upper <= lower):
            raise ValueError(f"Degenerate coordinate bounds: {lower}, {upper}")
        self.lower = torch.tensor(lower, device=self.device)
        self.upper = torch.tensor(upper, device=self.device)
        self.derivative_scale = 2.0 / (self.upper - self.lower)

        mask_cfg = self.metadata.get(
            "geometry_exclusion", self.metadata.get("cylinder_mask")
        )
        if mask_cfg is None:
            raise ValueError("Prepared artifact lacks geometry provenance")
        self.geometry_kind = str(
            mask_cfg.get("kind", "official_reference_disk")
        )
        center_row = int(round(float(mask_cfg["center_y_px"])))
        center_column = int(round(float(mask_cfg["center_x_px"])))
        if not (
            0 <= center_row < self.x_star.shape[0]
            and 0 <= center_column < self.x_star.shape[1]
        ):
            raise ValueError("Geometry reference centre lies outside the grid")
        self.geometry_center_star = torch.tensor(
            [
                self.x_star[center_row, center_column],
                self.y_star[center_row, center_column],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.geometry_half_extent_star = 0.5

        (
            self.X_data,
            self.y_data,
            self.y_data_clean,
            self.data_corruption_labels,
        ) = self._make_training_measurements()
        self.n_data_batch = int(cfg["batch"]["n_data"])
        if self.n_data_batch <= 0:
            raise ValueError("batch.n_data must be positive")

        eval_cfg = cfg.get("eval", {})
        self.eval_batch_size = int(eval_cfg.get("batch_size", 65536))
        self.pde_eval_points = int(eval_cfg.get("pde_points", 4096))
        self.pde_eval_seed = int(eval_cfg.get("pde_seed", 20260725))
        eval_generator = torch.Generator(device=self.device)
        eval_generator.manual_seed(self.pde_eval_seed)
        self.X_f_eval = self._sample_collocation(
            self.pde_eval_points, eval_generator
        )

    def _normalize_physical_star(self, coordinates: torch.Tensor) -> torch.Tensor:
        return 2.0 * (coordinates - self.lower) / (self.upper - self.lower) - 1.0

    def _make_training_measurements(self):
        sensor_x = self.x_star.reshape(-1)[self.train_sensor_indices]
        sensor_y = self.y_star.reshape(-1)[self.train_sensor_indices]
        time_grid, sensor_grid = np.meshgrid(
            np.arange(self.t_star.size),
            np.arange(self.train_sensor_indices.size),
            indexing="ij",
        )
        x = sensor_x[sensor_grid].reshape(-1)
        y = sensor_y[sensor_grid].reshape(-1)
        t = self.t_star[time_grid].reshape(-1)
        coordinates = torch.tensor(
            np.stack((x, y, t), axis=1),
            dtype=torch.float32,
            device=self.device,
        )

        u = self.u_star.reshape(self.t_star.size, -1)[
            :, self.train_sensor_indices
        ]
        v = self.v_star.reshape(self.t_star.size, -1)[
            :, self.train_sensor_indices
        ]
        values = torch.tensor(
            np.stack((u, v), axis=-1).reshape(-1, 2),
            dtype=torch.float32,
            device=self.device,
        )
        u_clean = self.u_clean_star.reshape(self.t_star.size, -1)[
            :, self.train_sensor_indices
        ]
        v_clean = self.v_clean_star.reshape(self.t_star.size, -1)[
            :, self.train_sensor_indices
        ]
        clean_values = torch.tensor(
            np.stack((u_clean, v_clean), axis=-1).reshape(-1, 2),
            dtype=torch.float32,
            device=self.device,
        )
        expected_label_shape = (
            self.t_star.size,
            self.train_sensor_indices.size,
            2,
        )
        if self.training_corruption_mask is None:
            labels = np.zeros(expected_label_shape, dtype=bool)
        else:
            if self.training_corruption_mask.shape != expected_label_shape:
                raise ValueError(
                    "training_corruption_mask shape "
                    f"{self.training_corruption_mask.shape} does not match "
                    f"{expected_label_shape}"
                )
            labels = self.training_corruption_mask
        label_tensor = torch.tensor(
            labels.reshape(-1, 2),
            dtype=torch.bool,
            device=self.device,
        )
        return (
            self._normalize_physical_star(coordinates),
            values,
            clean_values,
            label_tensor,
        )

    def _sample_collocation(
        self, n_points: int, generator: torch.Generator
    ) -> torch.Tensor:
        if n_points <= 0:
            raise ValueError("Number of collocation points must be positive")
        accepted = []
        n_accepted = 0
        while n_accepted < n_points:
            candidates = -1.0 + 2.0 * torch.rand(
                (max(2 * (n_points - n_accepted), 64), 3),
                generator=generator,
                device=self.device,
            )
            star = self.lower + 0.5 * (candidates + 1.0) * (
                self.upper - self.lower
            )
            offset_x = star[:, 0] - self.geometry_center_star[0]
            offset_y = star[:, 1] - self.geometry_center_star[1]
            if self.geometry_kind == "official_reference_square_envelope":
                excluded = (
                    (offset_x.abs() <= self.geometry_half_extent_star)
                    & (offset_y.abs() <= self.geometry_half_extent_star)
                )
            else:
                excluded = (
                    offset_x.square() + offset_y.square()
                    <= self.geometry_half_extent_star**2
                )
            fluid = candidates[~excluded]
            accepted.append(fluid)
            n_accepted += fluid.shape[0]
        return torch.cat(accepted, dim=0)[:n_points]

    def sample_batch(
        self, n_f: int, generator: torch.Generator | None = None
    ):
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(torch.initial_seed())
        X_f = self._sample_collocation(n_f, generator)
        n_data = self.X_data.shape[0]
        indices = torch.randint(
            0,
            n_data,
            (min(self.n_data_batch, n_data),),
            generator=generator,
            device=self.device,
        )
        return {
            "X_f": X_f,
            "X_d": self.X_data[indices],
            "y_d": self.y_data[indices],
        }

    def measurement_residual(self, model, batch):
        """Return scalar-interface-ready velocity residuals before flattening."""
        return batch["y_d"] - model(batch["X_d"])[:, :2]

    def pde_residuals(self, model, X_normalized: torch.Tensor):
        X = make_leaf(X_normalized)
        output = model(X)
        if output.shape[1] != 3:
            raise ValueError("Cylinder PIV model must output (u, v, p)")
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        sx, sy, st = self.derivative_scale

        du_dq = grad_sum(u, X)
        dv_dq = grad_sum(v, X)
        dp_dq = grad_sum(p, X)
        u_x, u_y, u_t = (
            sx * du_dq[:, 0:1],
            sy * du_dq[:, 1:2],
            st * du_dq[:, 2:3],
        )
        v_x, v_y, v_t = (
            sx * dv_dq[:, 0:1],
            sy * dv_dq[:, 1:2],
            st * dv_dq[:, 2:3],
        )
        p_x, p_y = sx * dp_dq[:, 0:1], sy * dp_dq[:, 1:2]

        d_u_x_dq = grad_sum(u_x, X)
        d_u_y_dq = grad_sum(u_y, X)
        d_v_x_dq = grad_sum(v_x, X)
        d_v_y_dq = grad_sum(v_y, X)
        u_xx = sx * d_u_x_dq[:, 0:1]
        u_yy = sy * d_u_y_dq[:, 1:2]
        v_xx = sx * d_v_x_dq[:, 0:1]
        v_yy = sy * d_v_y_dq[:, 1:2]

        inverse_reynolds = self.inverse_effective_reynolds()
        momentum_u = (
            u_t
            + u * u_x
            + v * u_y
            + p_x
            - inverse_reynolds * (u_xx + u_yy)
        )
        momentum_v = (
            v_t
            + u * v_x
            + v * v_y
            + p_y
            - inverse_reynolds * (v_xx + v_yy)
        )
        continuity = u_x + v_y
        return momentum_u, momentum_v, continuity

    def effective_reynolds(self) -> torch.Tensor:
        if self.log_effective_reynolds is None:
            return torch.tensor(
                self.reynolds, dtype=torch.float32, device=self.device
            )
        return torch.exp(self.log_effective_reynolds)

    def inverse_effective_reynolds(self) -> torch.Tensor:
        if self.log_effective_reynolds is None:
            return torch.tensor(
                1.0 / self.reynolds, dtype=torch.float32, device=self.device
            )
        return torch.exp(-self.log_effective_reynolds)

    def extra_params(self):
        if self.log_effective_reynolds is None:
            return []
        return [self.log_effective_reynolds]

    def state_dict(self):
        if self.log_effective_reynolds is None:
            return {}
        return {
            "log_effective_reynolds": self.log_effective_reynolds.detach()
        }

    def load_state_dict(self, state_dict):
        if (
            self.log_effective_reynolds is not None
            and "log_effective_reynolds" in state_dict
        ):
            with torch.no_grad():
                self.log_effective_reynolds.copy_(
                    state_dict["log_effective_reynolds"].to(self.device)
                )

    def pde_residual_loss(self, model, batch):
        residuals = self.pde_residuals(model, batch["X_f"])
        return sum(residual.square().mean() for residual in residuals)

    def data_loss(self, model, batch, phase=1):
        residual = self.measurement_residual(model, batch)
        kind = str(self.cfg.get("method", {}).get("kind", "mse")).lower()
        if kind == "lad":
            return residual.abs().mean()
        return residual.square().mean()

    def eval_on_grid(self, model, grid_cfg=None):
        model.eval()
        flat_x = self.x_star.reshape(-1)[self.heldout_indices]
        flat_y = self.y_star.reshape(-1)[self.heldout_indices]
        u_true_all = self.u_clean_star.reshape(self.t_star.size, -1)[
            :, self.heldout_indices
        ]
        v_true_all = self.v_clean_star.reshape(self.t_star.size, -1)[
            :, self.heldout_indices
        ]

        absolute_error_sum = 0.0
        squared_error_sum = 0.0
        true_magnitude_sum = 0.0
        true_squared_sum = 0.0
        count = 0
        with torch.no_grad():
            for time_index, time_value in enumerate(self.t_star):
                for start in range(0, self.heldout_indices.size, self.eval_batch_size):
                    stop = min(start + self.eval_batch_size, self.heldout_indices.size)
                    coordinates = torch.tensor(
                        np.stack(
                            (
                                flat_x[start:stop],
                                flat_y[start:stop],
                                np.full(stop - start, time_value),
                            ),
                            axis=1,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    prediction = model(
                        self._normalize_physical_star(coordinates)
                    )[:, :2]
                    true = torch.tensor(
                        np.stack(
                            (
                                u_true_all[time_index, start:stop],
                                v_true_all[time_index, start:stop],
                            ),
                            axis=1,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    error_magnitude = torch.linalg.vector_norm(
                        prediction - true, dim=1
                    )
                    true_magnitude = torch.linalg.vector_norm(true, dim=1)
                    absolute_error_sum += float(error_magnitude.sum().cpu())
                    squared_error_sum += float(
                        (prediction - true).square().sum().cpu()
                    )
                    true_magnitude_sum += float(true_magnitude.sum().cpu())
                    true_squared_sum += float(true.square().sum().cpu())
                    count += stop - start

        rmae = (absolute_error_sum / count) / (true_magnitude_sum / count)
        rmse = np.sqrt(squared_error_sum / count)
        rms_true = np.sqrt(true_squared_sum / count)
        return {
            "rMAE": float(rmae),
            "rMSE": float(rmse / rms_true),
            "n_heldout_spatial": int(self.heldout_indices.size),
            "n_heldout_measurements": int(count),
        }

    def evaluate_physics(self, model):
        model.eval()
        momentum_u, momentum_v, continuity = self.pde_residuals(
            model, self.X_f_eval
        )
        momentum_sq = 0.5 * (
            momentum_u.square().mean() + momentum_v.square().mean()
        )
        return {
            "pde_momentum_rms": float(momentum_sq.sqrt().detach().cpu()),
            "continuity_rms": float(
                continuity.square().mean().sqrt().detach().cpu()
            ),
            "n_pde_eval_points": int(self.X_f_eval.shape[0]),
        }

    def plot_final(self, model, grid_cfg, out_dir):
        return None
