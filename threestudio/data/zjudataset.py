import torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset                                                                                                                                                                                                                                                                                                                                                                      
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from dataclasses import dataclass
from threestudio.utils.ray import *
import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *
import pytorch_lightning as pl
def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def rodrigues_mat_to_rot(R):
    eps =1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.)/ 2.
    #sinacostrc2 = np.sqrt(1 - trc2 * trc2)
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega

def rodrigues_rot_to_mat(r):
    wx,wy,wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta*theta)
    c = np.sin(theta) / theta
    R = np.zeros([3,3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def pose_spherical(theta, phi, radius, coord_trans=None):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    if coord_trans is not None:
        c2w = c2w @ torch.Tensor(coord_trans)
    return c2w

@dataclass
class ZJUDataModuleConfig:
    dataroot: str = ""
    train_downsample_resolution: int = 4
    eval_downsample_resolution: int = 4
    train_data_interval: int = 1
    eval_data_interval: int = 10
    batch_size: int = 1
    eval_batch_size: int = 1
    camera_layout: str = "around"
    camera_distance: float = -1
    eval_interpolation: Optional[Tuple[int, int, int]] = None  # (0, 1, 30)

class ZJUIterableDataset(IterableDataset):
    def __init__(self, cfg: Any, split = 'train') -> None:
        super().__init__()
        self.cfg: ZJUDataModuleConfig = cfg
        self.root_dir = self.cfg.dataroot
        self.split = split
        self.downsample = self.cfg.train_downsample_resolution
        self.img_size = 1024
        self.img_wh = (int(self.img_size/self.downsample),int(self.img_size/self.downsample))
        self.frame_h = self.img_wh[0]
        self.frame_w = self.img_wh[1]
        self.define_transforms()
        self.near_far = [10., 1000.]
        self.near = self.near_far[0]
        self.far = self.near_far[1]
        self.read_meta()
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        self.camera_model = self.meta['camera_model']
        w, h = self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        self.image_paths = []
        self.poses = []
        self.frames_img = []
        self.all_masks = []
        self.all_depth = []
        self.rays_o = []
        self.rays_d = []
        self.masks_img = []
        self.frames_proj = []
        self.frames_position = []

        
        idxs = list(range(0, len(self.meta['frames']), self.cfg.train_data_interval))
        self.n_frames = len(idxs)
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            self.intrinsics = torch.eye(3).float()
            self.intrinsic[0, 0] = frame["fl_x"] / self.downsample
            self.intrinsic[1, 1] = frame["fl_y"] / self.downsample
            self.intrinsic[0, 2] = frame["cx"] / self.downsample
            self.intrinsic[1, 2] = frame["cy"] / self.downsample

            if self.camera_model is None or self.camera_model == 'ZJU':
                pose = np.array(frame['transform_matrix']) @ self.ZJU2opencv
            c2w : Float[Tensor, "4 4"] = torch.as_tensor(
                pose, dtype=torch.float32
            )
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.permute(1, 2, 0)  # (h, w, 4) RGBA
            img = img[:, :, :3]
            # img = img[:, :, :3] * img[:, :, -1:] + (1 - img[:, :,  -1:])  # blend A to RGB
            self.frames_img += [img]
            
            maskpath = image_path.replace(self.split, self.split+"_mask")
            mask = cv2.imread(maskpath)
            mask = cv2.resize(mask, (w, h)).copy()
            mask = torch.FloatTensor(mask)/255
            self.masks_img.append(mask)

            rays_o, rays_d = get_rays(self.directions, c2w, keepdim= True)  # both (h*w, 3)
            self.rays_o.append(rays_o)
            self.rays_d.append(rays_d)
            
            near = self.near_far[0]
            far = self.near_far[1]
            
            self.frames_position.append(camera_position)

        self.poses = torch.stack(self.poses)
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]
        # build render path
        self.render_path = torch.stack([pose_spherical(angle, -30.0, 4.0, self.ZJU2opencv) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        self.masks_img : Float[Tensor, "B H W"] = torch.stack(self.masks_img, dim=0)
        # self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        
        self.rays_o = torch.stack(self.rays_o, 0)  # (len(self.meta['frames]),h*w, 3)
        self.rays_d = torch.stack(self.rays_d, 0)
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(self.frames_img, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(self.frames_position, dim=0)
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )



    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        index = torch.randint(0, self.n_frames, (1,)).item()
        return {
            "index": index,
            "rays_o": self.rays_o[index : index + 1],
            "rays_d": self.rays_d[index : index + 1],
            # "mvp_mtx": self.proj_mat[index : index + 1],
            # "c2w": self.poses[index : index + 1],
            "light_positions": self.light_positions[index : index + 1],
            "camera_positions": self.frames_position[index : index + 1],
            "gt_rgb": self.frames_img[index : index + 1],
            "mask": self.masks_img[index: index + 1],
            "height": self.img_wh[0],
            "width": self.img_wh[1],
        }

class ZJUDataset(Dataset):
    def __init__(self, cfg: Any, split = 'test') -> None:
        self.cfg: ZJUDataModuleConfig = cfg
        self.root_dir = self.cfg.dataroot
        self.split = split
        self.is_stack = False
        self.downsample = self.cfg.eval_downsample_resolution
        self.img_wh = (int(800/self.downsample),int(800/self.downsample))
        self.frame_h = self.img_wh[0]
        self.frame_w = self.img_wh[1]
        self.define_transforms()

        self.ZJU2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.white_bg = True
        self.near_far = [2.0,6.0]
        self.read_meta()

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.frames_img = []
        self.all_masks = []
        self.all_depth = []
        self.rays_o = []
        self.rays_d = []
        self.masks_img = []
        self.frames_proj = []
        self.frames_position = []

        idxs = list(range(0, len(self.meta['frames']), self.cfg.eval_data_interval))
        self.n_frames = len(idxs)
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.ZJU2opencv
            c2w : Float[Tensor, "4 4"] = torch.as_tensor(
                pose, dtype=torch.float32
            )
            camera_position: Float[Tensor, "3"] = c2w[:3, 3:].reshape(-1)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.permute(1, 2, 0)  # (h, w, 4) RGBA
            img = img[:, :, :3]
            # img = img[:, :, :3] * img[:, :, -1:] + (1 - img[:, :,  -1:])  # blend A to RGB
            self.frames_img += [img]
            
            maskpath = image_path.replace(self.split, self.split+"_mask")
            mask = cv2.imread(maskpath)
            mask = cv2.resize(mask, (w, h)).copy()
            mask = torch.FloatTensor(mask)/255
            self.masks_img.append(mask)

            rays_o, rays_d = get_rays(self.directions, c2w, keepdim= True)  # both (h*w, 3)
            self.rays_o.append(rays_o)
            self.rays_d.append(rays_d)
            
            near = self.near_far[0]
            far = self.near_far[1]
            
            self.frames_position.append(camera_position)

        self.poses = torch.stack(self.poses)
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]
        # build render path
        self.render_path = torch.stack([pose_spherical(angle, -30.0, 4.0, self.ZJU2opencv) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        self.masks_img : Float[Tensor, "B H W"] = torch.stack(self.masks_img, dim=0)
        # self.frames_proj: Float[Tensor, "B 4 4"] = torch.stack(frames_proj, dim=0)
        
        self.rays_o = torch.stack(self.rays_o, 0)  # (len(self.meta['frames]), h, w, 3)
        self.rays_d = torch.stack(self.rays_d, 0)
        self.frames_img: Float[Tensor, "B H W 3"] = torch.stack(self.frames_img, dim=0)
        self.frames_position: Float[Tensor, "B 3"] = torch.stack(self.frames_position, dim=0)
        self.light_positions: Float[Tensor, "B 3"] = torch.zeros_like(
            self.frames_position
        )

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.frames_img)

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            # "mvp_mtx": self.proj_mat[index],
            # "c2w": self.poses[index],
            "light_positions": self.light_positions[index],
            "camera_positions": self.frames_position[index],
            "gt_rgb": self.frames_img[index],
            "mask": self.masks_img[index],
            "height": self.img_wh[0],
            "width": self.img_wh[1],
        }

    def __len__(self):
        return self.frames_img.shape[0]

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.img_wh[0], "width": self.img_wh[1]})
        return batch

@register("ZJU-datamodule")
class ZJUDataModule(pl.LightningDataModule):
    cfg: ZJUDataModuleConfig
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ZJUDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ZJUIterableDataset(self.cfg)
        if stage in [None, "test", "predict"]:
            self.test_dataset = ZJUDataset(self.cfg, split = "test")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ZJUDataset(self.cfg, split = "val")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=1,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )