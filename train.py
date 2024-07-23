import hydra
from config import BasicConfig
from models.gaussian_module import GaussianModule
from utils.data_utils import get_dataset
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning as L
from pathlib import Path
from datetime import datetime
from utils.data_utils import upload_results
from omegaconf import DictConfig, OmegaConf
from gaussian_renderer import render, network_gui


def modify_config(cfg):
    datasets = []
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    for name, config in cfg_dict['dataset'].items():
        config["name"] = name
        datasets.append(config)
    cfg['dataset'] = datasets
    return cfg


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg):
    cfg = modify_config(cfg)
    cfg = BasicConfig.model_validate(cfg)

    logger = WandbLogger(
        project=cfg.project,
        name=f"{cfg.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    network_gui.init(cfg.ip, cfg.port)

    for dataset in cfg.dataset:
        model_path = Path(dataset.model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        source_path = Path(dataset.source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path {source_path} does not exist.")
        dataset.source_path = str(source_path.absolute())

        model = GaussianModule(cfg, dataset)
        train_dataset = get_dataset(model.scene)
        test_dataset = get_dataset(model.scene, is_train=False)

        trainer = L.Trainer(
            max_steps=cfg.train.iterations,
            logger=logger,
            enable_checkpointing=False
        )
        trainer.fit(model, train_dataset)
        trainer.test(model, test_dataset)

    upload_results(logger)


if __name__ == "__main__":
    main()
