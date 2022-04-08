from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import pydantic
import yaml

if TYPE_CHECKING:
    import pandas as pd
    import wandb


def get_config(filepath: str) -> Config:
    return Config(filepath)


class Config(pydantic.BaseModel):
    # user input
    entity: str
    project: str
    run_id: Optional[str]

    rel_data_path: str
    folder_name: str
    run_name: str

    # derived
    project_path: Optional[str]
    folder_path: Optional[str]

    def __init__(self, filepath: str):
        with open(filepath) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        super(Config, self).__init__(**data)

        self.project_path = f"{self.entity}/{self.project}"
        self.folder_path = self._get_folder_path()

    def get_runs(self, api: wandb.Api, **kwargs) -> wandb.apis.public.Runs:
        return api.runs(f"{self.project_path}", **kwargs)

    def get_run(self, api: wandb.Api) -> wandb.apis.public.Run:
        assert self.run_id is not None
        return api.run(f"{self.project_path}/{self.run_id}")

    def get_metrics(self, api: wandb.Api) -> pd.DataFrame:
        run = self.get_run(api)
        return run.history()

    def _get_folder_path(self) -> str:
        folder_path = os.path.join(self.rel_data_path, self.folder_name)
        folder_path = os.path.abspath(os.path.join(os.getcwd(), folder_path))

        # check existing
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        return folder_path