import os
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import scalers
from tsl.experiment import Experiment
from tsl.metrics import torch_metrics
from tsl.nn import models as tsl_models

import lib
from lib.datasets import PeakWeather
from lib.nn import models
import lib.metrics
from lib.nn.models import ICONData, ICONDummyModel
from lib.nn.predictors import Predictor, SamplingPredictor


def get_model_class(model_str):
    # Forecasting models  ###############################################
    if model_str == 'tts_imp':
        model = models.TimeThenGraphIsoModel
    elif model_str == 'rnn_glob':
        model = models.GlobalLocalRNNModel
    elif model_str == 'rnn_emb':
        model = models.GlobalLocalRNNModel
    elif model_str == 'pers_st':
        model = models.PersistenceModel
    elif model_str == 'pers_day':
        model = models.PersistenceModel
    elif model_str == 'icon':
        model = ICONDummyModel
    elif model_str == 'rnn':
        model = tsl_models.RNNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model

def scale_scaled_weather_vars(dataset, estimation_slice):
    channels = dataset.get_frame('u', return_pattern=False)
    mask = dataset.get_frame('u_mask', return_pattern=False)

    def min_max(ch, msk, sl):
        # mi, ma = ch[msk.bool()].min(), ch[msk.bool()].max()
        mi, ma = ch[sl][msk[sl]].min(), ch[sl][msk[sl]].max()
        return (ch - mi) / (ma - mi)

    def std_scale(ch, msk, sl):
        # mu_ = (ch * msk).sum() / msk.sum()
        # sigma2_ = ((ch - mu_)**2 * msk).sum() / msk.sum()
        mu_ = (ch[sl] * msk[sl]).sum() / msk[sl].sum()
        sigma2_ = ((ch[sl] - mu_)**2 * msk[sl]).sum() / msk[sl].sum()
        return (ch - mu_) / np.clip(np.sqrt(sigma2_), a_min=1e-2, a_max=None)

    channel_scalers = {
        'wind_direction': min_max, 
        'wind_speed': min_max, 
        'wind_u': std_scale, 
        'wind_v': std_scale, 
        'wind_gust': std_scale, 
        'pressure': std_scale, 
        'precipitation': min_max, 
        'sunshine': min_max, 
        'temperature': std_scale, 
        'humidity': min_max
    }

    for i, ch in enumerate(dataset.covariates_id):
        channels[..., i] = channel_scalers[ch](channels[..., i], mask[..., i], estimation_slice)

    return channels, mask

def run(cfg: DictConfig):

    ########################################
    # Get Dataset                          #
    ########################################
    dataset = PeakWeather(**cfg.dataset.hparams,
                          extended_nwp_vars = ["ew_wind", "nw_wind"] if cfg.nwp_test_set else None)
    # Get connectivity
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)
    # Get mask
    mask = dataset.get_mask()

    # Get covariates
    u = []
    if cfg.dataset.covariates.year:
        u.append(dataset.datetime_encoded('year').values)
    if cfg.dataset.covariates.day:
        u.append(dataset.datetime_encoded('day').values)
    if cfg.dataset.covariates.weekday:
        u.append(dataset.datetime_onehot('weekday').values)
    if cfg.dataset.covariates.mask:
        u.append(mask.astype(np.float32))
    if 'u' in dataset.covariates:
        # other weather vars as covariates
        other_channels, other_mask = scale_scaled_weather_vars(dataset, slice(0, 365*24))  # small enough (1y) to not exceed the training set
        u.append(other_channels) 
        u.append(other_mask)

    # Concatenate covariates
    assert len(u)
    ndim = max(u_.ndim for u_ in u)
    u = np.concatenate([np.repeat(u_[:, None], dataset.n_nodes, 1)
                        if u_.ndim < ndim else u_
                        for u_ in u], axis=-1)

    # Get static information
    covs = dict(u=u)
    if cfg.dataset.covariates.v:
        v = dataset.stations_table[[*cfg.dataset.static_attributes]]
        v = (v - v.mean(0)) / v.std(0)
        covs["v"] = v

    torch_dataset = SpatioTemporalDataset(dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covs,
                                          connectivity=adj,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)
    assert (dataset.dataframe().loc[:, dataset.nodes].columns == dataset.dataframe().columns).all()

    # Scale input features
    scaler_cfg = cfg.get('scaler')
    if scaler_cfg is not None:
        scale_axis = (0,) if scaler_cfg.axis == 'node' else (0, 1)
        scaler_cls = getattr(scalers, f'{scaler_cfg.method}Scaler')
        transform = dict(target=scaler_cls(axis=scale_axis))
    else:
        transform = None

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        mask_scaling=True,
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    print(f"Split sizes\n\tTrain: {len(dm.trainset)}\n"
          f"\tValidation: {len(dm.valset)}\n"
          f"\tTest: {len(dm.testset)}")

    print("Sample:")
    print(dm.torch_dataset[0])

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in covs else 0
    d_exog += torch_dataset.input_map.v.shape[-1] if 'v' in covs else 0
    
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    if cfg.loss_fn == "mae":
        loss_fn = torch_metrics.MaskedMAE()
    elif cfg.loss_fn == "ens":
        loss_fn = lib.metrics.EnergyScore()
    else:
        raise ValueError(f"Loss function <{cfg.loss_fn}> not available.")

    mae_at = [1, 3, 6, 12, 18, 24]
    point_metrics = {'mae': torch_metrics.MaskedMAE(),
                      **{f'mae_{h:d}h': torch_metrics.MaskedMAE(at=h-1) for h in mae_at if h <= cfg.horizon},
                     'mse': torch_metrics.MaskedMSE(),
                     'dir_mae': lib.metrics.DirectionMAE(zerowind=cfg.dataset.zerowind), 
                      **{f'dir_mae_{h:d}h': lib.metrics.DirectionMAE(at=h-1, zerowind=cfg.dataset.zerowind) for h in mae_at if h <= cfg.horizon},
                     'speed_mae': lib.metrics.SpeedMAE(),
                      **{f'speed_mae_{h:d}h': lib.metrics.SpeedMAE(at=h-1) for h in mae_at if h <= cfg.horizon}}
    
    sample_metrics = {'smae': lib.metrics.SampleMAE(),
                      **{f'smae_{h:d}h': lib.metrics.SampleMAE(at=h-1) for h in mae_at if h <= cfg.horizon},
                      'smse': lib.metrics.SampleMSE(),
                      'dir_smae': lib.metrics.SampleDirectionMAE(zerowind=cfg.dataset.zerowind),
                      **{f'dir_smae_{h:d}h': lib.metrics.SampleDirectionMAE(at=h-1, zerowind=cfg.dataset.zerowind) for h in mae_at if h <= cfg.horizon},
                      'dir_ens': lib.metrics.DirectionEnergyScore(zerowind=cfg.dataset.zerowind),
                      **{f'dir_ens_{h:d}h': lib.metrics.DirectionEnergyScore(at=h-1, zerowind=cfg.dataset.zerowind) for h in mae_at if h <= cfg.horizon},
                      'speed_smae': lib.metrics.SampleSpeedMAE(),
                      **{f'speed_smae_{h:d}h': lib.metrics.SampleSpeedMAE(at=h-1) for h in mae_at if h <= cfg.horizon},
                      'speed_ens': lib.metrics.SpeedEnergyScore(),
                      **{f'speed_ens_{h:d}h': lib.metrics.SpeedEnergyScore(at=h-1) for h in mae_at if h <= cfg.horizon},
                      'ens': lib.metrics.EnergyScore(),
                      **{f'ens_{h:d}h': lib.metrics.EnergyScore(at=h-1) for h in mae_at if h <= cfg.horizon},}

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # select the appropriate predictor    
    if isinstance(loss_fn, lib.metrics.SampleMetric):
        predictor_class = SamplingPredictor
        assert not point_metrics.keys() & sample_metrics.keys()
        log_metrics = dict(**point_metrics, **sample_metrics)
        predictor_kwargs = dict(**cfg.sampling)
        monitored_metric = 'val_smae'
    else:
        predictor_class = Predictor
        log_metrics = point_metrics
        predictor_kwargs = dict()
        monitored_metric = 'val_mae'

    # setup predictor
    predictor = predictor_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
        **predictor_kwargs
    )

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor=monitored_metric,
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor=monitored_metric,
        mode='min',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Will place the logs in ./mlruns
    exp_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=cfg.mlflow_tracking_uri)

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback, lr_monitor])

    load_model_path = cfg.get('load_model_path')
    if not isinstance(predictor.model, (models.PersistenceModel, ICONDummyModel)):
        if load_model_path is not None:
            predictor.load_model(load_model_path)
        else:
            trainer.fit(predictor,
                        train_dataloaders=dm.train_dataloader(),
                        val_dataloaders=dm.val_dataloader())
            predictor.load_model(checkpoint_callback.best_model_path)

    predictor.freeze()

    if not isinstance(predictor.model, (models.PersistenceModel, ICONDummyModel)) and load_model_path is None:
        result = checkpoint_callback.best_model_score.item()
    else:
        result = dict()

    ########################################
    # testing                              #
    ########################################
    if cfg.nwp_test_set and isinstance(predictor, SamplingPredictor):

        icon = ICONData(pw_dataset=dataset) 

        metrics = icon.test_set_eval(
            torch_dataset=torch_dataset, 
            metrics=sample_metrics, 
            predictor=predictor, 
            batch_size=cfg.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("Model metrics on NWP test set:")
        for k, v in metrics.compute().items():
            logger.info(f" - {k}: {v:.5f}")

    if not isinstance(predictor.model, ICONDummyModel):
        trainer.test(predictor, dataloaders=dm.test_dataloader())

    return result


if __name__ == '__main__':
    exp = Experiment(run_fn=run, config_path='../config/',
                     config_name='default')
    res = exp.run()
    logger.info(res)
