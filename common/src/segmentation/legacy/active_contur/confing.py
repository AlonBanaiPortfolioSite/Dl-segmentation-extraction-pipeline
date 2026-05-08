"""
Active Contour Configuration
=============================
Dataclass-based configuration for active contour refinement experiments,
with YAML serialisation for reproducibility.

.. note::
    This module was used for mid-pipeline contour refinement during
    retraining iterations.  It is **not part of the final segmentation
    pipeline** but is retained for reproducibility.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import yaml


@dataclass
class FilterConfig:
    """Parameters for image denoising prior to active contour fitting."""

    filter_type: Literal["gaussian", "bilateral"]

    # Gaussian
    sigma: float = 3.0

    # Bilateral
    d: int = 7
    sigma_color: int = 30
    sigma_space: int = 5


@dataclass
class ActiveContourConfig:
    """Snake energy term weights."""

    alpha: float = 0.015   # continuity (elasticity)
    beta: float = 10.0     # smoothness (rigidity)
    gamma: float = 0.001   # time-step


@dataclass
class OutputConfig:
    """Flags controlling which artefacts are written to disk."""

    save_denoised_image: bool = True
    save_init_contour: bool = True
    save_final_contour: bool = True
    save_summary_stats: bool = True


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    On construction the ``output_folder`` is auto-generated with a version
    counter so that repeated runs do not overwrite each other.
    """

    experiment_name: str
    base_folder: str

    filter: FilterConfig
    active_contour: ActiveContourConfig
    output: OutputConfig

    file_format: str = "tiff"   # "tiff" or "nifti"
    axis_order: str = "zyx"     # "zyx" or "xyz"

    output_folder: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.output_folder is None:
            if self.filter.filter_type == "gaussian":
                filter_str = f"gauss{self.filter.sigma}"
            else:
                filter_str = f"bilat{self.filter.d}"

            ac = self.active_contour
            ac_str = f"a{ac.alpha}_b{ac.beta}_g{ac.gamma}"
            base_name = f"{self.experiment_name}_{filter_str}_{ac_str}"

            counter = 1
            while True:
                folder_name = f"{base_name}_v{counter:03d}"
                full_path = os.path.join(self.base_folder, folder_name)
                if not os.path.exists(full_path):
                    break
                counter += 1

            self.output_folder = full_path

    # ------------------------------------------------------------------
    # YAML persistence
    # ------------------------------------------------------------------

    def save_to_yaml(self, filepath: Optional[str] = None) -> str:
        """Serialise the configuration to a YAML file."""
        if filepath is None:
            os.makedirs(self.output_folder, exist_ok=True)
            filepath = os.path.join(self.output_folder, "config.yaml")

        config_dict = {
            "experiment_name": self.experiment_name,
            "base_folder": self.base_folder,
            "timestamp": self.timestamp,
            "output_folder": self.output_folder,
            "filter": {
                "filter_type": self.filter.filter_type,
                "sigma": self.filter.sigma,
                "d": self.filter.d,
                "sigma_color": self.filter.sigma_color,
                "sigma_space": self.filter.sigma_space,
            },
            "active_contour": {
                "alpha": self.active_contour.alpha,
                "beta": self.active_contour.beta,
                "gamma": self.active_contour.gamma,
            },
            "output": {
                "save_denoised_image": self.output.save_denoised_image,
                "save_init_contour": self.output.save_init_contour,
                "save_final_contour": self.output.save_final_contour,
                "save_summary_stats": self.output.save_summary_stats,
            },
        }

        with open(filepath, "w") as fh:
            yaml.dump(config_dict, fh, default_flow_style=False, sort_keys=False)

        print(f"Config saved to: {filepath}")
        return filepath

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "ExperimentConfig":
        """Deserialise an ``ExperimentConfig`` from a YAML file."""
        with open(filepath, "r") as fh:
            d = yaml.safe_load(fh)

        return cls(
            experiment_name=d["experiment_name"],
            base_folder=d["base_folder"],
            filter=FilterConfig(**d["filter"]),
            active_contour=ActiveContourConfig(**d["active_contour"]),
            output=OutputConfig(**d["output"]),
            timestamp=d.get("timestamp"),
            output_folder=d.get("output_folder"),
        )