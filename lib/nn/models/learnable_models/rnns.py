from typing import List, Literal, Union

from lib.nn.models.learnable_models.time_then_graph_isotropic import TimeThenGraphIsoModel


class GlobalLocalRNNModel(TimeThenGraphIsoModel):
    def __init__(self, input_size: int, horizon: int, n_nodes: int = None,
                 output_size: int = None,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 emb_size: int = 0,
                 add_embedding_before: Union[str, List[str]] = 'encoding',
                 use_local_weights: Union[str, List[str]] = None,
                 time_layers: int = 1,
                #  root_weight: bool = True,
                #  norm: str = 'none',
                #  add_backward: bool = False,
                #  cached: bool = False,
                 activation: str = 'elu',
                 noise_mode: Literal["lin", "multi", "add", "none"] = "lin",
                #  time_skip_connect: bool = False
                 ):
        super(GlobalLocalRNNModel, self).__init__(
            input_size=input_size, 
            horizon=horizon, 
            n_nodes=n_nodes, 
            output_size=output_size, 
            exog_size=exog_size, 
            hidden_size=hidden_size, 
            emb_size=emb_size, 
            add_embedding_before=add_embedding_before, 
            use_local_weights=use_local_weights, 
            time_layers=time_layers, 
            graph_layers=0, 
            root_weight=False, 
            norm='none', 
            add_backward=False, 
            cached=False, 
            activation=activation, 
            noise_mode=noise_mode, 
            time_skip_connect=False)
