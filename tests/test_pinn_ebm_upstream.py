"""Focused checks for the separately labeled PINN-EBM variants."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/rebuttal/run_pinn_ebm_upstream.py"
SPEC = importlib.util.spec_from_file_location("pinn_ebm_upstream_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class UpstreamVariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.active_path = (
            ROOT / "configs/rebuttal/pinn_ebm_upstream_active.yaml"
        )
        cls.paper_path = ROOT / "configs/rebuttal/pinn_ebm_paper_spec.yaml"
        cls.active = RUNNER.load_yaml(cls.active_path)
        cls.paper = RUNNER.load_yaml(cls.paper_path)

    def test_configs_are_valid_and_scientifically_distinct(self) -> None:
        RUNNER.validate_config(self.active, self.active_path)
        RUNNER.validate_config(self.paper, self.paper_path)
        self.assertIsNone(self.active["source"]["patch"])
        self.assertIsNotNone(self.paper["source"]["patch"])
        self.assertEqual(
            self.active["expected_active_source"]["pinn"]["hidden_layers"], 4
        )
        self.assertEqual(
            self.paper["expected_active_source"]["pinn"]["hidden_layers"], 8
        )
        for config in (self.active, self.paper):
            execution = config["execution"]
            self.assertEqual(execution["seed"], 0)
            self.assertEqual(execution["nrun"], 5)
            self.assertEqual(execution["model_indices"], [0, 3, 2])
            self.assertEqual(execution["pinn_updates"], 100000)
            self.assertEqual(execution["ebm_init_updates"], 2000)
            self.assertIn("not independent", execution["seed_semantics"].lower())

    def test_generated_input_completes_only_disclosed_missing_choice(self) -> None:
        rendered = RUNNER.render_upstream_input(self.active)
        namespace = {
            "set_par": lambda pars, key, value: pars.setdefault(key, value),
            "get_ds": lambda pars: object(),
        }
        exec(rendered, namespace)
        pars = namespace["pars"]
        self.assertEqual(pars["prop_noise"], 0)
        self.assertEqual(pars["x_opt"], 101)
        self.assertEqual(pars["n_opt"], "3G")
        self.assertEqual(pars["jmodel_vec"], [0, 3, 2])
        self.assertEqual(pars["lf_fac2"], 50)
        self.assertEqual(pars["lf_fac2_alt"], 1)

    def test_staged_sources_match_commit_and_patch_scope(self) -> None:
        output_root = ROOT / "outputs/rebuttal/pinn_ebm_upstream"
        for config in (self.active, self.paper):
            source = output_root / "_sources" / config["variant_id"]
            self.assertTrue(source.is_dir(), source)
            metadata = RUNNER.expected_source_status(config, source)
            self.assertEqual(metadata["commit"], config["source"]["commit"])
            RUNNER.audit_source_semantics(config, source)

    def test_dataset_is_the_frozen_raissi_artifact(self) -> None:
        dataset = ROOT / "outputs/rebuttal/pinn_ebm_upstream/data/cylinder_nektar_wake.mat"
        metadata = RUNNER.verify_dataset(dataset, self.active["dataset"])
        self.assertEqual(metadata["sha256"], self.active["dataset"]["sha256"])
        self.assertEqual(metadata["size_bytes"], 24081984)

    def test_compute_matched_weight_grid_is_complete_and_no_gate(self) -> None:
        manifest = yaml.safe_load(
            (ROOT / "configs/rebuttal/pinn_ebm_variant_manifest.yaml").read_text()
        )
        grid = manifest["variants"]["C_compute_matched_current_task"][
            "pde_weight_grid"
        ]
        self.assertEqual(set(grid), {1, 10, 50})
        for weight, relative_path in grid.items():
            config = yaml.safe_load((ROOT / relative_path).read_text())
            self.assertEqual(float(config["training"]["joint_pde_weight"]), weight)
            self.assertFalse(config["gate"]["enabled"])
            self.assertEqual(config["method"]["kind"], "pinn_ebm")


if __name__ == "__main__":
    unittest.main()
