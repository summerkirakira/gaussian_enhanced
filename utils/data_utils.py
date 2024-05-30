from torch.utils.data import Dataset
from scene import Scene


class GaussianDataset(Dataset):
    def __init__(self, scene: Scene):
        self.scene = scene

    def __len__(self):
        return len(self.scene.getTrainCameras())

    def __getitem__(self, idx):
        return self.scene.getTrainCameras()[idx], 'dummy_target'


def get_dataset(scene: Scene):
    return GaussianDataset(scene)
