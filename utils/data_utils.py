from torch.utils.data import Dataset
from scene import Scene
from pydantic import BaseModel
from lightning.pytorch.loggers.wandb import WandbLogger
import wandb
from PIL import Image
import random


class TrainResults(BaseModel):
    class Result(BaseModel):
        name: str
        ll1: float
        lssim: float
        psnr: float
        mse: float
        lpips: float
        image: str

    data: dict[str, list[Result]]


_results = TrainResults(data={})


class GaussianDataset(Dataset):
    def __init__(self, scene: Scene, is_train=True):
        self.is_train = is_train
        self.cameras = scene.getTrainCameras()

        if not is_train:
            self.cameras = random.sample(self.cameras, 10)

        # self.cameras = scene.getTrainCameras()[:int(len(scene.getTrainCameras()) * 0.8)] if is_train else scene.getTrainCameras()[int(len(scene.getTrainCameras()) * 0.8):]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx], 'dummy_target'


def get_dataset(scene: Scene, is_train=True):
    return GaussianDataset(scene, is_train)


def add_result(dataset, result: TrainResults.Result):
    global _results
    if dataset not in _results.data:
        _results.data[dataset] = []
    else:
        _results.data[dataset].append(result)


def upload_results(logger: WandbLogger):
    global _results
    for dataset, results in _results.data.items():

        table = wandb.Table(columns=["Name", "L1", "SSIM", "PSNR", "MSE", "LPIPS"])

        average = {"ll1": 0, "lssim": 0, "psnr": 0, "mse": 0, "lpips": 0}
        for index, result in enumerate(results):
            # image = Image.open(result.image)
            table.add_data(f"Index: {index} / name: {result.name}", result.ll1, result.lssim, result.psnr, result.mse, result.lpips)
            average["ll1"] += result.ll1
            average["lssim"] += result.lssim
            average["psnr"] += result.psnr
            average["mse"] += result.mse
            average["lpips"] += result.lpips

        average["ll1"] /= len(results)
        average["lssim"] /= len(results)
        average["psnr"] /= len(results)
        average["mse"] /= len(results)
        average["lpips"] /= len(results)

        table.add_data("Average", average["ll1"], average["lssim"], average["psnr"], average["mse"], average["lpips"])

        logger.experiment.log({f"Dataset {dataset} Results": table})

