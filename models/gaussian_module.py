from typing import Union, Optional, Callable, Any
from utils.loss_utils import l1_loss, ssim, lpips_loss
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from gaussian_renderer import render, network_gui
from config import BasicConfig
from scene import GaussianModel, Scene
from scene.cameras import Camera
from typing import Tuple
import torch
from torch import Tensor
from utils.image_utils import psnr, mse
import wandb
from PIL import Image
import pathlib
from utils.data_utils import TrainResults, add_result
import numpy as np
import uuid
from pathlib import Path


class GaussianModule(L.LightningModule):
    def __init__(self, config: BasicConfig, dataset: BasicConfig.Dataset):
        super().__init__()

        model_path = Path(dataset.model_path)
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians, shuffle=True)
        self.background = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.background, dtype=torch.float32, device=dataset.data_device)
        self.dataset = dataset

        self.viewspace_point_tensor = None
        self.visibility_filter = None
        self.radii = None

        self.test_outputs = []
        self.cache_path = pathlib.Path("cache") / dataset.name / uuid.uuid4().hex
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True, exist_ok=True)

    def log_image(self, image: Tensor, gt_image: Tensor, name: str = "image"):
        concatenated_img = torch.cat([image, gt_image], dim=2)
        wandb.log({name: [wandb.Image(concatenated_img)]})

    def save_cache_image(self, image: Tensor, gt_image: Tensor, name="image") -> pathlib.Path:
        concatenated_img = torch.cat([image, gt_image], dim=2)
        concatenated_img = concatenated_img.detach().cpu().numpy() * 255
        concatenated_img = np.transpose(concatenated_img, (1, 2, 0))
        image = Image.fromarray(concatenated_img.astype(np.uint8))
        path = self.cache_path / f"{name.split('.')[0]}.jpg"
        image.save(path)
        return path

    def test_step(self, batch: Tuple[Camera, str], batch_idx):
        camera = batch[0]
        bg = torch.rand((3), device="cuda") if self.config.train.random_background else self.background
        render_pkg = render(camera, self.gaussians, self.config.pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = camera.original_image.cuda()

        image = torch.clamp(image, min=0, max=1.0)
        gt_image = torch.clamp(gt_image, min=0, max=1.0)

        Ll1 = l1_loss(image, gt_image)
        # self.log("test_l1_loss", Ll1)
        Lssim = ssim(image, gt_image)
        # self.log("test_ssim_loss", Lssim)

        test_psnr = psnr(image, gt_image).mean()
        test_mse = mse(image, gt_image).mean()
        test_lpips = lpips_loss(image, gt_image)

        image_path = self.save_cache_image(image, gt_image, name=camera.image_name)
        result = TrainResults.Result(name=camera.image_name, ll1=Ll1.item(), lssim=Lssim.item(), psnr=test_psnr.item(), mse=test_mse.item(), lpips=test_lpips.item(), image=str(image_path))
        add_result(self.dataset.name, result)

        self.log_image(image, gt_image, name=f"{self.dataset.name} Test Render Image")

        self.test_outputs.append({"test_ll1": Ll1, "test_lssim": Lssim, "test_psnr": test_psnr, "test_mse": test_mse, "test_lpips": test_lpips})

    def on_test_epoch_end(self):
        outputs = self.test_outputs
        avg_ll1 = torch.stack([x['test_ll1'] for x in outputs]).mean()
        avg_lssim = torch.stack([x['test_lssim'] for x in outputs]).mean()

        avg_psnr = torch.stack([x['test_psnr'] for x in outputs]).mean()
        avg_mse = torch.stack([x['test_mse'] for x in outputs]).mean()
        avg_lpips = torch.stack([x['test_lpips'] for x in outputs]).mean()

        self.log(f"{self.dataset.name}_avg_test_ll1", avg_ll1)
        self.log(f"{self.dataset.name}_avg_test_lssim", avg_lssim)
        self.log(f"{self.dataset.name}_avg_test_psnr", avg_psnr)
        self.log(f"{self.dataset.name}_avg_test_mse", avg_mse)
        self.log(f"{self.dataset.name}_avg_test_lpips", avg_lpips)


    def training_step(self, batch: Tuple[Camera, str], batch_idx):
        self.update_lr()
        camera = batch[0]
        bg = torch.rand((3), device="cuda") if self.config.train.random_background else self.background
        render_pkg = render(camera, self.gaussians, self.config.pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        self.viewspace_point_tensor = viewspace_point_tensor
        self.visibility_filter = visibility_filter
        self.radii = radii

        gt_image = camera.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        self.log(f"{self.dataset.name}_l1_loss", Ll1)
        Lssim = ssim(image, gt_image)
        self.log(f"{self.dataset.name}_ssim_loss", Lssim)
        loss = (1.0 - self.config.train.lambda_dssim) * Ll1 + self.config.train.lambda_dssim * (1.0 - Lssim)
        self.log(f"{self.dataset.name}_total_loss", loss)

        if self.global_step % 1000 == 0:
            self.log_image(image, gt_image, name=f"Rendered Image[{self.dataset.name}]")
        return loss

    def configure_optimizers(self):
        optimizer = self.gaussians.training_setup(self.config.train)
        return optimizer

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        optimizer.step(optimizer_closure)
        optimizer.zero_grad(set_to_none=True)

    def on_after_backward(self) -> None:
        with torch.no_grad():
            if self.global_step < self.config.train.densify_until_iteration:
                self.densify(self.viewspace_point_tensor, self.visibility_filter, self.radii)

        self.training_report()
        self.connect_gui()

    def training_report(self):
        self.log(f"{self.dataset.name}_total_points", self.gaussians.get_xyz.shape[0])

    def connect_gui(self):
        iteration = self.global_step
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, self.config.pipe.convert_SHs_python, self.config.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.config.pipe, self.background, scaling_modifer)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, self.dataset.source_path)
                if do_training and ((iteration < int(self.config.train.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def densify(self, viewspace_point_tensor, visibility_filter, radii):
        # Keep track of max radii in image-space for pruning
        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                  radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if self.global_step > self.config.train.densify_from and self.global_step % self.config.train.densify_interval == 0:
            size_threshold = 20 if self.global_step > self.config.train.opacity_reset_interval else None
            self.gaussians.densify_and_prune(self.config.train.densify_grad_threshold, 0.005, self.scene.cameras_extent,
                                             size_threshold)

        if self.global_step % self.config.train.opacity_reset_interval == 0 or (
                self.dataset.white_background and self.global_step == self.config.train.densify_from):
            if self.global_step == 0:
                return
            self.gaussians.reset_opacity()

    def update_lr(self):
        lr = self.gaussians.update_learning_rate(self.global_step)
        if self.global_step % 1000 == 0:
            self.gaussians.oneupSHdegree()
        self.log(f"{self.dataset.name}_learning_rate", lr)
