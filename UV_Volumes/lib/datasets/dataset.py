import torch.utils.data as data
import numpy as np
import os
import imageio
import cv2
import sys
sys.path.append("/home/yjli/AIGC/Adversarial_camou/UV_Volumes")
from ..config import cfg
from ..utils import data_utils
import sys
import pickle
import json
'''
dataroot = '/home/yjli/AIGC/humannerf/dataset/zju_mocap/CoreView_377'
human = 'CoreView_377'
ann_file = '/home/yjli/AIGC/humannerf/dataset/zju_mocap/CoreView_377/annots.npy'
split = 'test'
'''


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):

        super(Dataset, self).__init__()

        self.data_root = data_root
        dp_root = os.path.join(self.data_root, cfg.densepose)
        if not os.path.exists(dp_root):
            print("The densepose directory '{}' is not accessible.".format(dp_root))
            sys.exit()
    
        self.human = human
        self.split = split
        
        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame

        if self.human == "Peter" or "wild" in self.human or "CoreView" not in self.human:
            # Specify the path to your pickle file
            file_path = os.path.join(data_root, "metadata.json")
            # Load the pickle file
            with open(file_path, "r") as file:
                data = json.load(file)

            self.num_cams = 1
            if split == 'train': 
                self.ims_names =  sorted(os.listdir(os.path.join(data_root, "Camera_B1"))[i:i + ni * i_intv][::i_intv],
                    key=lambda x: int(os.path.splitext(x)[0]))
            else:
                self.ims_names =  sorted(os.listdir(os.path.join(data_root, "Camera_B1"))[i + ni * i_intv: -1],
                    key=lambda x: int(os.path.splitext(x)[0]))
            
            self.cams = {}
            self.cams['K'] = []
            self.cams['D'] = []
            self.cams['T'] = []
            self.cams['R'] = []
            for index in  range(len(self.ims_names)):
                self.cams['K'].append(data[self.ims_names[index]]['K'])
                self.cams['D'].append(data[self.ims_names[index]]['D'])
                self.cams['T'].append(data[self.ims_names[index]]['T'])
                self.cams['R'].append(data[self.ims_names[index]]['R'])
            # TT = np.array(self.cams['T'])
            # print(np.max(TT)) # 49
            self.cam_inds = range(len(self.ims_names))
            
            self.ims = [os.path.join("Camera_B1", x) for x in self.ims_names] 
      
        else:
            if len(cfg.test_view) == 0:
                test_view = [i for i in range(num_cams) if i not in cfg.training_view]
            else:
                test_view = cfg.test_view
            view = cfg.training_view if split == 'train' else test_view
            if len(view) == 0:
                view = [0]
            annots = np.load(ann_file, allow_pickle=True).item()
            self.cams = annots['cams']
            num_cams = len(self.cams['K'])
            self.ims = np.array([
                np.array(ims_data['ims'])[view]
                for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
            ]).ravel()

            self.cam_inds = np.array([
                np.arange(len(ims_data['ims']))[view]
                for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
                ]).ravel()
            self.num_cams = len(view)

        self.nrays = cfg.N_rand

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, cfg.mask,
                                self.ims[index])[:-4] + '.png'
        msk_cihp = imageio.v2.imread(msk_path)

        if cfg.mask == 'mask_cihp':
            msk = (msk_cihp != 0).astype(np.uint8)
        else:
            msk = (msk_cihp / 255 ).astype(np.uint8)
            msk = (msk > cfg.mask_threshold).astype(np.uint8)
        # else:
        #     msk = (msk_cihp / 255. < cfg.mask_threshold).astype(np.uint8)

        return msk

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32) # [6890,3]

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= cfg.box_padding
        max_xyz += cfg.box_padding

        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        poses = params['poses'][0].astype(np.float32)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= cfg.box_padding
        max_xyz += cfg.box_padding

        bounds = np.stack([min_xyz, max_xyz], axis=0)

        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_shape = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_shape = (out_shape | (x - 1)) + 1 # make the output shape as the times of x

        return coord, out_shape, can_bounds, bounds, Rh, Th, poses


    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.v2.imread(img_path).astype(np.float32) / 255.
        img = cv2.resize(img, (cfg.W, cfg.H))
        msk = self.get_mask(index)

        dp_name = self.ims[index].split('.')[0] + '_IUV.png'
        dp_path = os.path.join(self.data_root, cfg.densepose, dp_name)

        if os.path.exists(dp_path):
            dp = cv2.imread(dp_path)
        else:
            dp = np.zeros_like(img).astype(np.uint8)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        
        # cv2.imwrite("undistort.png", img[...,[2,1,0]]*255)
        # cv2.imwrite("undistort_mask.png", msk*255)

        if cfg.ignore_boundary:
            border = cfg.cihp_border
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        if cfg.erode_msk:
            border = cfg.mask_border
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.erode(msk.copy(), kernel)

        
        if "CoreView" in self.human:
            R = np.array(self.cams['R'][cam_ind])
            T = np.array(self.cams['T'][cam_ind]) / 1000.
        else:
            R = np.array(self.cams['R'][cam_ind])
            T = np.array(self.cams['T'][cam_ind])

        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        dp = cv2.resize(dp, (W, H), interpolation=cv2.INTER_NEAREST)
        i_gt = np.eye(25)[dp[:,:,0]][..., (1-int(cfg.use_bg)):].astype(np.float32)
        uv_gt = dp[:, :, 1:].astype(np.float32) / 255.
        iuv = np.concatenate((i_gt, uv_gt), axis=-1)

        if len(msk.shape) > 2:
            msk = msk[:,:,0]

        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313','CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i # image里文件名

        if self.human in ['cmu_panoptic']:
            i = os.path.basename(img_path).split('.')[0]

        ###########
        coord, out_sh, can_bounds, bounds, Rh, Th, poses = self.prepare_input(i)

        if cfg.use_nb_mask_at_box: # when train, this is false; when test, this is true
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, masked_iuv, mask_at_dp, mask_at_body, mask_at_bg, ray_d_center = \
                data_utils.sample_ray_h36m_whole(img, msk, K, R, T, can_bounds, iuv)
        else:
            rgb, ray_o, ray_d, near, far, coord_, mask_at_box, masked_iuv, mask_at_dp, mask_at_body, mask_at_bg, ray_d_center = \
                data_utils.sample_ray_h36m_whole_dilate(img, msk, K, R, T, can_bounds, iuv, self.split)

        
        ret = {
            'img_name': self.ims[index], 
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'iuv': masked_iuv,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'mask_at_dp': mask_at_dp,
            'mask_at_body': mask_at_body,
            'mask_at_bg': mask_at_bg,
            'poses': poses,
            'ratio': np.array(cfg.ratio, dtype=np.float32),
            'ray_d_center': ray_d_center.astype(np.float32)
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = index // self.num_cams
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind,
            'H': H,
            'W': W,
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
    
if __name__ == "__main__":
    # dataroot = '/home/yjli/AIGC/Adversarial_camou/my_dataset/Peter_humannerf'
    # human = 'Peter'
    # ann_file = '/home/yjli/AIGC/Adversarial_camou/my_dataset/Peter_humannerf/annots.npy'
    # split = 'test'
    data_root = cfg.train_dataset.data_root
    human = cfg.train_dataset.human
    ann_file = cfg.train_dataset.ann_file
    split = cfg.train_dataset.split
    dataset = Dataset(data_root, human, ann_file, split)
    item = dataset[0]

