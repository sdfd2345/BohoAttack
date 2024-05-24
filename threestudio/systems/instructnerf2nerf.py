import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *
import torchvision.transforms as transforms 

from yoloattack import YoloDetectorLoss


@threestudio.register("instructnerf2nerf-system")
class Instructnerf2nerf(BaseLift3DSystem):
        
    @dataclass
    class Config(BaseLift3DSystem.Config):
        per_editing_step: int = 10
        start_editing_step: int = 10000

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.edit_frames = {}
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.person_detector_loss = YoloDetectorLoss(classname = 'chair').eval().to(get_device())
        self.automatic_optimization = False
        

    def configure_optimizers(self):
        # only update colors, donot change density
        for p in self.renderer.base_renderer.geometry.density_network.parameters():
            p.requires_grad_(True)
            
        for p in self.renderer.base_renderer.geometry.feature_network.parameters():
            p.requires_grad_(True)
            
        optimizer = torch.optim.Adam(self.renderer.base_renderer.geometry.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1e-9)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            #     "monitor": "loss",
            #     "frequency": 5
            # },
        }
        
        return   

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        self.zero_grad()
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        origin_gt_rgb = batch["gt_rgb"]
        mask = batch["mask"]
        B, H, W, C = origin_gt_rgb.shape
        if batch_index in self.edit_frames:
            gt_rgb = self.edit_frames[batch_index].to(batch["gt_rgb"].device)
            gt_rgb = torch.nn.functional.interpolate(
                gt_rgb.permute(0, 3, 1, 2), (H, W), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1)
            batch["gt_rgb"] = gt_rgb
        else:
            gt_rgb = origin_gt_rgb
        out = self(batch)
        if (
            self.cfg.per_editing_step > 0
            and self.global_step > self.cfg.start_editing_step
        ):
            prompt_utils = self.prompt_processor()
            if (
                not batch_index in self.edit_frames
                or self.global_step % self.cfg.per_editing_step == 0
            ):
                self.renderer.eval()
                full_out = self(batch)
                self.renderer.train()
                input_rgb = full_out["comp_rgb"] * mask +  origin_gt_rgb.detach() * (1-mask)
                with torch.no_grad():
                    result = self.guidance(
                        input_rgb, origin_gt_rgb, prompt_utils
                    )
                out_rgb =  result["edit_images"].detach() * mask +  origin_gt_rgb.detach() * (1-mask)
                self.edit_frames[batch_index] = out_rgb.cpu()

        loss = 0.0
        guidance_out = {
            "loss_l1": torch.nn.functional.l1_loss(out["comp_rgb"], gt_rgb),
            "loss_lpips": self.perceptual_loss(
                out["comp_rgb"].permute(0, 3, 1, 2).contiguous(),
                gt_rgb.permute(0, 3, 1, 2).contiguous(),
            ).sum(),
        }
        
        if self.global_step > self.cfg.start_editing_step:
            loss_adv = self.person_detector_loss(out["comp_rgb"], gt_rgb, mask)
            loss = loss + loss_adv * self.C(self.cfg.loss.lambda_adv)
            self.log("train/loss_adv", loss_adv)

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log("train/loss", loss)
        self.manual_backward(loss)
        opt.step()
        return None
        #return {"loss":loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if torch.is_tensor(batch["index"]):
            batch_index = batch["index"].item()
        else:
            batch_index = batch["index"]
        if batch_index in self.edit_frames:
            B, H, W, C = batch["gt_rgb"].shape
            rgb = torch.nn.functional.interpolate(
                self.edit_frames[batch_index].permute(0, 3, 1, 2), (H, W)
            ).permute(0, 2, 3, 1)[0]
        else:
            rgb = batch["gt_rgb"][0]
        
        if "mask" in batch:
            boxes, yolo_img_gt = self.person_detector_loss.get_boxes_and_img(batch["gt_rgb"],batch["gt_rgb"],batch["mask"])
        else:
            boxes, yolo_img_gt = self.person_detector_loss.get_boxes_and_img(batch["gt_rgb"],batch["gt_rgb"])
        transform = transforms.Compose([ 
            transforms.PILToTensor() 
        ]) 
        yolo_img_gt = transform(yolo_img_gt).float().unsqueeze(0).permute(0,2,3,1)/255
        
        
        if "mask" in batch:
            boxes, yolo_img = self.person_detector_loss.get_boxes_and_img(out["comp_rgb"],batch["gt_rgb"],batch["mask"])
        else:
            boxes, yolo_img = self.person_detector_loss.get_boxes_and_img(out["comp_rgb"],batch["gt_rgb"])                                                                                                                                                                                                                            
        transform = transforms.Compose([ 
            transforms.PILToTensor() 
        ]) 
        yolo_img = transform(yolo_img).float().unsqueeze(0).permute(0,2,3,1)/255
        savepath = self.save_image_grid(
            f"val-it{self.true_global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": rgb,
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            
            + [
                {
                    "type": "rgb",
                    "img": yolo_img_gt[0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ]
            + [
                {
                    "type": "rgb",
                    "img": yolo_img[0],
                    "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )
        print("save image : " + savepath)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
    # def on_train_epoch_end(self):
    #     super().on_train_epoch_end()
    #     sch = self.lr_schedulers()
    #     # If the selected scheduler is a ReduceLROnPlateau scheduler.
    #     if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #         sch.step(self.trainer.callback_metrics["loss"])