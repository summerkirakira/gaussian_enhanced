from pydantic import BaseModel


class BasicConfig(BaseModel):

    class TrainCfg(BaseModel):
        densify_interval: int
        densify_from: int
        densify_grad_threshold: float
        densify_until_iteration: int
        feature_lr: float
        iterations: int
        lambda_dssim: float
        opacity_lr: float
        opacity_reset_interval: int
        percent_dense: float
        position_lr_final: float
        position_lr_initial: float
        position_lr_max_steps: int
        position_lr_delay_mult: float
        random_background: bool
        rotation_lr: float
        scaling_lr: float

    class Pipe(BaseModel):
        compute_cov3D_python: bool
        convert_SHs_python: bool
        debug: bool

    class Dataset(BaseModel):
        data_device: str = 'cuda'
        eval: bool = False
        images: str = 'images'
        resolution: int = -1
        sh_degree: int = 3
        source_path: str = './data/ship'
        white_background: bool = False
        model_path: str = './outputs/ship'

    train: TrainCfg
    pipe: Pipe
    project: str
    dataset: Dataset
    name: str
    ip: str
    port: int
