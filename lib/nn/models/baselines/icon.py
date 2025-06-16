import os
from torchmetrics import MetricCollection
from tqdm import tqdm
import xarray as xr
import pandas as pd
import torch
import numpy as np
from tsl.nn import models as tsl_models
import lib

class ICONData:

    def __init__(self, pw_dataset, stations=None, root=None):

        self.stations = pw_dataset.nodes if stations is None else stations

        self.ds_u = pw_dataset.icon_data["ew_wind"]
        self.ds_v = pw_dataset.icon_data["nw_wind"]
        assert "u_wind" in self.ds_u and "v_wind" in self.ds_v 
        assert "u_wind" not in self.ds_v and "v_wind" not in self.ds_u 

        self.ds_u.compute()
        self.ds_v.compute()
        
        if self.stations is not None:  # filter stations
            self.ds_u = self.ds_u.sel(nat_abbr=self.stations)
            self.ds_v = self.ds_v.sel(nat_abbr=self.stations)

        # consider reference time steps available for both u_wind and v_wind
        self.reftimes = np.intersect1d(self.ds_u.reftime.values, self.ds_v.reftime.values)
        self.reftimes = np.unique(self.reftimes)
        assert np.isin(self.reftimes, self.ds_u.reftime).all()
        assert np.isin(self.reftimes, self.ds_v.reftime).all()

    def get_ensamble(self, reftime, horizon):
        from einops import rearrange
        slice_u = self.ds_u.sel(reftime=reftime, lead=slice(np.timedelta64(1, 'h'), np.timedelta64(horizon, 'h')))
        slice_v = self.ds_v.sel(reftime=reftime, lead=slice(np.timedelta64(1, 'h'), np.timedelta64(horizon, 'h')))
        y_hat = torch.cat([
            torch.from_numpy(slice_u.to_array().values),
            torch.from_numpy(slice_v.to_array().values)
        ], dim=0)
        y_hat = rearrange(y_hat, "f b h n s -> s b h n f")
        return y_hat
    
    def test_set_idx(self, torch_dataset):
        assert torch_dataset.delay == 0

        rt_dti = pd.DatetimeIndex(self.reftimes, tz="UTC")
        ds_dti = pd.DatetimeIndex(torch_dataset.data_timestamps()["window"][:, -1])
        intersect = ds_dti.isin(rt_dti)
        test_ds_indices = intersect.nonzero()[0]
        test_dt = ds_dti[intersect]

        return test_ds_indices, test_dt
    
    def test_set_eval(self, torch_dataset, metrics, predictor=None, batch_size=32, device="cpu"):
        test_ds_idx, icon_reftimes = self.test_set_idx(torch_dataset)
        
        metrics = MetricCollection(
            {n: m for n, m in metrics.items() if isinstance(m, lib.metrics.SampleMetric)},
            prefix="nwp/test_")
        metrics.to(device)
        metrics.reset()

        predictor.in_testing_step = True  # sets the correct number of test samples
        predictor.to(device)

        test_ds_idx = np.array(test_ds_idx)
        icon_reftimes = icon_reftimes.tz_localize(None)
        for i in tqdm(range(0, len(test_ds_idx), batch_size), desc=f'Eval on NWP test set'):
            data = torch_dataset[test_ds_idx[i: i + batch_size]]
            data.to(device)
            if isinstance(predictor.model, ICONDummyModel):
                y_hat = self.get_ensamble(reftime=icon_reftimes[i: i + batch_size], horizon=torch_dataset.horizon)
                y_hat=y_hat.to(device)
            else:
                y_hat = predictor.predict_batch(data, preprocess=False, postprocess=True)

            metrics.update(y=data.y, y_hat=y_hat, mask=data.mask)

        predictor.in_testing_step = False
        return metrics


class ICONDummyModel(tsl_models.BaseModel):
    def forward(self):
        raise NotImplementedError("This a dummy model")
