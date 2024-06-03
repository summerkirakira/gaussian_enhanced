from torch.utils.data import Dataset
from scene import Scene


class GaussianDataset(Dataset):
    def __init__(self, scene: Scene, is_train=True):
        self.is_train = is_train
        self.cameras = scene.getTrainCameras()[:int(len(scene.getTrainCameras()) * 0.8)] if is_train else scene.getTrainCameras()[int(len(scene.getTrainCameras()) * 0.8):]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx], 'dummy_target'


def get_dataset(scene: Scene, is_train=True):
    return GaussianDataset(scene, is_train)
