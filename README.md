# PeakWeather 🌦️🌤️⛈️  a use case in wind forecasting

This repository provides the code for replicating the wind forecasting experiments on the recently released [PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) dataset which has been presented in paper

> __[PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning](https://arxiv.org/abs/2506.13652)__,  
> _[Daniele Zambon](https://dzambon.github.io)¹, [Michele Cattaneo](https://github.com/MicheleCattaneo)², [Ivan Marisca](https://marshka.github.io)¹, [Jonas Bhend](https://github.com/jonasbhend)², [Daniele Nerini](https://github.com/dnerini)², [Cesare Alippi](https://alippi.faculty.polimi.it/)¹³._   
> ¹ USI, IDSIA (Lugano, Switzerland), ² MeteoSwiss  (Zurich, Switzerland), ³ PoliMi (Milan, Italy).

[PeakWeather](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) is a high-resolution spatiotemporal dataset of validated ground-based meteorological observations from Switzerland, and is associated with a [Python library](https://github.com/MeteoSwiss/PeakWeather) to download, load, and prepare the data.
It includes

- **Meteorological observations** from 302 stations across Switzerland, every **10 minutes** from **Jan 2017 to Mar 2025**
- **Topographic features** extracted from a 50m DEM for each station
- **Operational NWP forecasts** (ICON-CH1-EPS) co-located with observation points

The dataset supports a variety of tasks, including spatiotemporal forecasting, virtual sensing, and graph structure learning. Here, wind forecasting is considered.


## Getting Started

### Requirements

To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```shell
conda env create -f conda_env.yml
conda activate peakweather-env
```

> [!TIP]
> Please, note that Conda packages for `pytorch_geometric` are currently not available for M1/M2/M3 macs; see [PyG documentation](https://pytorch-geometric.readthedocs.io/en/2.4.0/notes/installation.html). To install it, you can comment out PyG-related entries in `conda_env.yml` and install with pip, e.g.,
> ```shell
> pip install torch_geometric==2.4 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
> ```


### Model training and testing

To train and evaluate a model for wind forecasting, run

```bash
python -m experiments.run_wind_prediction dataset=wind_1d model=<MODEL_NAME> 
```

Available models are listed in `experiments/run_wind_prediction.py` script; examples include `tts_imp` (STGNN), `pers_st` (PM-st), and `icon` (ICON).


### Logging and tracking 

To track the training, we rely on MLflow.
To run it locally, execute the command below and open a browser at `http://127.0.0.1:<PORT>/`:

```bash
mlflow ui --port <PORT>
```

To run the experiment using your hosted MLflow server, specify the tracking URI with the overwrite: `++mlflow_tracking_uri=<YOUR_URI>` when starting the experiment.

After training, the script automatically evaluates the model's predictive performance on the test set using predefined metrics.

### Configuration files

The `config` directory stores all the configuration files used to run the
experiment using [Hydra](https://hydra.cc/).


## Citation

If you use this code in your work, please cite our paper:

```bibtex
@misc{zambon2025peakweather,
  title={PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning}, 
  author={Zambon, Daniele and Cattaneo, Michele and Marisca, Ivan and Bhend, Jonas and Nerini, Daniele and Alippi, Cesare},
  year={2025},
  eprint={2506.13652},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.13652}, 
}
```
