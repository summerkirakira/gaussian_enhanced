import hydra

from config import BasicConfig
from models.gaussian_module import GaussianModule
from utils.data_utils import GaussianDataModule
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
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

        checkpoint_callback = ModelCheckpoint(
            monitor=f'{dataset.name}_total_loss',
            dirpath='./checkpoints',
            filename=f'{dataset.name}/'+'{epoch:02d}-{' + f'{dataset.name}_total_loss' + ':.4f}',
            save_top_k=1,
            every_n_train_steps=cfg.train.checkpoint_interval,
            mode='min'
        )

        model_path = Path(dataset.model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        source_path = Path(dataset.source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path {source_path} does not exist.")
        dataset.source_path = str(source_path.absolute())

        if not cfg.is_test:
            model = GaussianModule(cfg, dataset)
        else:
            model_path = Path('checkpoints') / f'{dataset.name}'
            if not model_path.exists():
                raise FileNotFoundError(f"Model path {model_path} does not exist.")
            checkpoint_path = str(next(model_path.glob('*.ckpt')))
            model = GaussianModule.load_from_checkpoint(
                checkpoint_path,
                config=cfg,
                dataset=dataset
            )
        datamodule = GaussianDataModule(model.scene)

        trainer = L.Trainer(
            max_steps=cfg.train.iterations,
            logger=logger,
            callbacks=[checkpoint_callback],
        )
        datamodule.test_dataloader()
        if not cfg.is_test:
            trainer.fit(model, datamodule.train_dataloader())
        model.eval()
        trainer.test(model, datamodule.test_dataloader())

    upload_results(logger)


if __name__ == "__main__":
    main()
