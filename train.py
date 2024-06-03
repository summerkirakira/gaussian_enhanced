import hydra
from config import BasicConfig
from models.gaussian_module import GaussianModule
from utils.data_utils import get_dataset
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning as L
from pathlib import Path


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg):
    cfg = BasicConfig.model_validate(cfg)

    model_path = Path(cfg.dataset.model_path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    source_path = Path(cfg.dataset.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path {source_path} does not exist.")
    cfg.dataset.source_path = str(source_path.absolute())

    model = GaussianModule(cfg)
    train_dataset = get_dataset(model.scene)
    test_dataset = get_dataset(model.scene, is_train=False)

    logger = WandbLogger(
        project=cfg.project,
        name=cfg.name
    )

    trainer = L.Trainer(
        max_steps=cfg.train.iterations,
        logger=logger,
        enable_checkpointing=True
    )
    trainer.fit(model, train_dataset)
    trainer.test(model, test_dataset)


if __name__ == "__main__":
    main()
