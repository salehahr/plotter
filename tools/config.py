from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Union

import pydantic
import yaml

from tools import data

if TYPE_CHECKING:
    import pandas as pd
    import wandb

    from tools.data_types import ParamsDict

    WandbRuns = Union[wandb.apis.public.Runs, List[wandb.apis.public.Run]]


def get_config(filepath: str) -> Config:
    return Config(filepath)


class Config(pydantic.BaseModel):
    # general
    entity: str
    rel_data_path: str

    # user input
    project: str
    run_id: Optional[Union[str, List[str]]]
    sweep_id: Optional[str]

    folder_name: str

    # derived - for api
    project_path: Optional[str]
    multiple_runs: Optional[bool] = False

    # derived - for csv files
    folder_path: Optional[str]
    run_name: Optional[str]

    all_runs_csv: Optional[str]
    training_csv: Optional[str]
    skips_csv: Optional[str]
    sweeps_csv: Optional[str]

    def __init__(self, filepath: str):
        with open("configs/general.yaml") as f:
            gen_data = yaml.load(f, Loader=yaml.FullLoader)

        with open(filepath) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            config_data.update(gen_data)

        super(Config, self).__init__(**config_data)

        self.project_path = f"{self.entity}/{self.project}"
        self.folder_path = self._get_folder_path()

        if self.run_id is None or isinstance(self.run_id, list):
            self.multiple_runs = True

        self.all_runs_csv = self.gen_filepath("runs.csv")
        self.skips_csv = self.gen_filepath("skips.csv")
        self.training_csv = self.gen_filepath("training.csv")
        self.sweeps_csv = self.gen_filepath("sweeps.csv")

    def get_runs(self, api: wandb.Api, all: bool = True, **kwargs) -> WandbRuns:
        if all:
            return api.runs(f"{self.project_path}", **kwargs)
        else:
            return [api.run(f"{self.project_path}/{id}") for id in self.run_id]

    def get_runs_df(self, api: wandb.Api, **kwargs) -> pd.DataFrame:
        runs = self.get_runs(api, **kwargs)
        return data.get_runs_df(runs)

    def get_sweep(self, api: wandb.Api, sweep_id: str) -> wandb.apis.public.Sweep:
        return api.sweep(f"{self.project_path}/{sweep_id}")

    def get_sweeps_df(self, api: wandb.Api, sweep_id: str) -> pd.DataFrame:
        sweep = self.get_sweep(api, sweep_id)
        df = data.get_runs_df(sweep.runs).drop("train_params", axis=1)

        cols_to_flatten = ["model_arch_params", "summary"]
        return data.flatten(df, cols_to_flatten)

    def get_run(self, api: wandb.Api) -> wandb.apis.public.Run:
        assert self.run_id is not None
        return api.run(f"{self.project_path}/{self.run_id}")

    def gen_filepath(self, filename: Optional[str] = None) -> str:
        # parse filename
        if self.run_name is None:
            assert filename is not None
        else:
            filename = (
                f"{self.run_name}_{filename}" if filename is not None else self.run_name
            )

        if not filename.endswith("csv"):
            filename += ".csv"

        return os.path.join(self.folder_path, filename)

    def save_project_summary(self, api: wandb.Api) -> None:
        # # all runs
        runs_df = self.get_runs_df(api)
        runs_df.to_csv(self.all_runs_csv)

        # skips
        skips_df = self.get_runs_df(api, filters={"tags": "skip"})
        data.save_flat(skips_df, self.skips_csv)

        # training
        training_runs_df = self.get_runs_df(api, filters={"tags": "train_hyp"})
        data.save_flat(training_runs_df, self.training_csv)

        # sweeps
        sweeps_df = self.get_sweeps_df(api, self.sweep_id)
        sweeps_df.to_csv(self.sweeps_csv)

    def update_config(
        self,
        api: wandb.Api,
        update_vals: ParamsDict,
        runs: Optional[List[str]] = None,
    ) -> None:
        runs = self._parse_runs(api, runs)

        for run in runs:
            for k, v in update_vals.items():
                run.config[k] = v
            run.update()

    def update_metric(
        self, api: wandb.Api, metric: str, runs: Optional[List[str]] = None
    ):
        metric_summary = f"best_{metric}"

        runs = self._parse_runs(api, runs)
        for run in runs:
            history = [h[metric] for h in run.history(keys=[metric], pandas=False)]

            if not history:
                print(f"Did not update metric for run {run.path[-1]}")
                continue

            best_val = max(data.filter.smooth(history))
            run.summary[metric_summary] = best_val
            run.summary.update()

    def _get_folder_path(self) -> str:
        folder_path = os.path.join(self.rel_data_path, self.folder_name)
        folder_path = os.path.abspath(os.path.join(os.getcwd(), folder_path))

        # check existing
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return folder_path

    def _parse_runs(
        self, api: wandb.Api, runs: Optional[List[str]] = None
    ) -> WandbRuns:
        if runs is None:
            return self.get_runs(api)
        else:
            return [api.run(f"{self.project_path}/{r}") for r in runs]
