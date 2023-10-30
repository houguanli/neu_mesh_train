import json

import cv2
import torch
import torch.nn.functional as F
import random
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def load_K_Rt_from_P(camera_matrix_map, camera_name):
    M_name = camera_name + "_M"
    K_name = camera_name + "_K"
    M = camera_matrix_map[M_name]
    K = camera_matrix_map[K_name]

    pose = np.array(M)
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    # M as pose(C2W), K as intrinsics
    return pose, intrinsics


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.pic_mode = conf.get_string('pic_mode')
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.' + self.pic_mode)))

        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.' + self.pic_mode)))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = []
        self.scale_mats_np = []
        self.intrinsics_all = []
        self.pose_all = []
        js_path = os.path.join(self.data_dir, self.render_cameras_name)
        print(js_path)
        with open(os.path.join(self.data_dir, self.render_cameras_name), "r") as json_file:
            camera_params_list = json.load(json_file)

        self.frame_count = conf.get_int('frame_count')
        self.camera_count = conf.get_int('camera_count')
        self.json_type = conf.get_string('json_type')

        print(self.frame_count)
        print(self.camera_count)
        for frame_id in range(1, self.frame_count + 1):
            for camera_id in range(1, self.camera_count + 1):
                mat_name = str(frame_id) + "_" + str(camera_id)
                if self.json_type == "separate":
                    idx = frame_id * self.camera_count + camera_id - self.camera_count - 1
                    camera_matrix_map = camera_params_list[idx]  # get current camera_info_map
                    pose, intrinsics = load_K_Rt_from_P(camera_matrix_map=camera_matrix_map, camera_name=mat_name)
                    self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                    self.pose_all.append(torch.from_numpy(pose).float())
                elif self.json_type == "merged":  # this mat can be found directly in the original list
                    pose, intrinsics = load_K_Rt_from_P(camera_matrix_map=camera_params_list, camera_name=mat_name)
                    self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                    self.pose_all.append(torch.from_numpy(pose).float())
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        # Object scale mat: region of interest to **extract mes-h**
        object_bbox_min = np.array([-0.1,  -0.1, -0.1])
        object_bbox_max = np.array([0.2, 0.2, 0.2])
        self.object_bbox_min = object_bbox_min
        self.object_bbox_max = object_bbox_max
        if conf.get_bool('with_sphere'):  # TODO: need to reset here
            self.sphere_center = torch.from_numpy(np.array([0.08, -0.06, 0.04]).astype(np.float32)).cuda()
            self.radius = 0.2
        else:
            self.sphere_center = torch.zeros(3)
            self.radius = 1

        print('Load data: End')

        self.focus_rays_in_mask = conf.get_bool('focus_rays_in_mask')  # this requires whether gen all rays in mask
        self.rays_o_in_masks, self.rays_v_in_masks, self.rays_color_in_masks = None, None, None

        def set_image_w_h(self, w, h):
            self.W = w
            self.H = h
        def gen_all_rays_in_mask(max_rays_in_gpu=2000000):
            # TODO: gen all rays_o and rays_v in mask for faster training
            print("----------------generating all rays within mask-----------------")
            rays_o_in_masks, rays_v_in_masks, rays_color_in_masks = [], [], []
            for index in range(0, self.frame_count):
                rays_mask = (self.masks[index]).detach().cpu().numpy()
                rays_mask = np.where(rays_mask > 0.1, 1, 0).astype(np.bool_)
                rays_color = (self.images[index]).detach().cpu().numpy()
                W, H = self.W, self.H
                tx = np.linspace(0, W - 1, W)
                ty = np.linspace(0, H - 1, H)
                pixels_x, pixels_y = np.meshgrid(tx, ty, indexing='ij')
                p = np.stack([pixels_x, pixels_y, np.ones_like(pixels_y)], axis=-1)  # W, H, 3
                intrinsics_inv = self.intrinsics_all_inv[index].detach().cpu().numpy()
                p = np.matmul(intrinsics_inv[None, None, :3, :3],
                              p[:, :, :, None]).squeeze()  # W, H, 3
                rays_v = p / np.linalg.norm(p, ord=2, axis=-1, keepdims=True)  # W, H, 3
                camera_pose = self.pose_all[index].detach().cpu().numpy()
                rays_v = np.matmul(camera_pose[None, None, :3, :3],
                                   rays_v[:, :, :, None]).squeeze()  # W, H, 3
                rays_o = np.tile(camera_pose[:3, 3], (W, H, 1))  # W, H, 3

                rays_o, rays_v = np.transpose(rays_o, (1, 0, 2)), np.transpose(rays_v, (1, 0, 2))  # H, W, 3

                rays_o, rays_v, rays_color = rays_o[rays_mask].reshape(-1, 3), rays_v[rays_mask].reshape(-1, 3), \
                    rays_color[rays_mask].reshape(-1, 3)  # H*W{mask), 3
                print("hold " + str(rays_o.shape[0]) + "rays for image " + str(index + 1))
                if len(rays_o) > max_rays_in_gpu:
                    # hold as a random sequence in max batch size
                    hold_sequence = np.random.choice(range(len(rays_o)), max_rays_in_gpu)  # pick outer_count in mask
                    rays_o, rays_v, rays_color = rays_o[hold_sequence], rays_v[hold_sequence], rays_color[hold_sequence]
                    # min(H*W{mask),max_rays_in_gpu) , 3

                # need to cut down rays here if its so big

                rays_o_in_masks.append(rays_o)
                rays_v_in_masks.append(rays_v)
                rays_color_in_masks.append(rays_color)
            # import pdb
            # pdb.set_trace()
            print("------------------generate finished--------------------")
            return rays_o_in_masks, rays_v_in_masks, rays_color_in_masks

        if self.focus_rays_in_mask:
            self.rays_o_in_masks, self.rays_v_in_masks, self.rays_color_in_masks = gen_all_rays_in_mask()

        print('Load data: End')
    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at_pose_mat(self, transform_matrix, resolution_level=1, intrinsic_inv=None):
        transform_matrix = torch.from_numpy(transform_matrix.astype(np.float32)).to(self.device)
        if intrinsic_inv is None:
            intrinsic_inv = self.intrinsics_all_inv[0]
        # transform_matrix.cuda()  # add to cuda
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        # we assume that the fx fy in all intrinsic mats are the same, so use the first intrinsics_all_inv to gen rays
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(transform_matrix[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = transform_matrix[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H W 3

    def gen_rays_at_pose(self, rotation, transition, resolution_level=1):
        rotation_mat, _ = cv2.Rodrigues(rotation)
        transform_matrix = np.zeros((4, 4))
        transform_matrix[0:3, 0:3] = rotation_mat
        transform_matrix[0:3, [3]] = transition
        transform_matrix = torch.from_numpy(transform_matrix)
        transform_matrix.cuda()   # add to cuda

        return self.gen_rays_at_pose_mat(transform_matrix, resolution_level)

    def gen_rays_at_pose_and_change(self, transform_matrix, moving_mat, resolution_level=1):
        #  moving mat refers a moving rigid body's current position to original, use its inv
        mov_inv = np.linalg.inv(moving_mat)
        after_tran = mov_inv @ transform_matrix

        return self.gen_rays_at_pose_mat(after_tran, resolution_level)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.to(self.device), rays_v.to(self.device), color, mask[:, :1]],
                         dim=-1).cuda()  # batch_size, 10

    def gen_random_rays_within_mask(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera within the mask, this function assumes that mask has been defined
        be cautious that image samples as (y,x)!
        """
        outer_count = int(batch_size * 0.1)
        inner_count = int(batch_size - outer_count)
        mask_cpu = self.masks[img_idx].cpu()
        fg_index = np.where(mask_cpu > 0.9)  # foreground & background
        fg_yx = np.stack(fg_index, axis=1)  # n x 2
        fg_num = fg_yx.shape[0]
        bg_index = np.where(mask_cpu < 0.1)
        bg_yx = np.stack(bg_index, axis=1)
        bg_num = bg_yx.shape[0]
        if inner_count > fg_num:
            inner_count = fg_num
            outer_count = batch_size - inner_count
        fg_index = np.random.choice(range(fg_num), inner_count)
        bg_index = np.random.choice(range(bg_num), outer_count)
        # import pdb
        # pdb.set_trace()
        pixels_x = np.concatenate((fg_yx[fg_index, 1], bg_yx[bg_index, 1]))
        pixels_y = np.concatenate((fg_yx[fg_index, 0], bg_yx[bg_index, 0]))
        pixels_x, pixels_y = torch.from_numpy(pixels_x), torch.from_numpy(pixels_y)
        pixels_x, pixels_y = pixels_x.to(self.device), pixels_y.to(self.device)
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.to(self.device), rays_v.to(self.device), color, mask[:, :1]],
                         dim=-1).cuda()  # batch_size, 10

    def select_random_rays_in_masks(self, img_idx, batch_size):
        outer_count = int(batch_size * 0.1)
        inner_count = int(batch_size - outer_count)
        rays_o_in_mask_i = self.rays_o_in_masks[img_idx]
        rays_v_in_mask_i = self.rays_v_in_masks[img_idx]
        rays_color_in_mask_i = self.rays_color_in_masks[img_idx]
        fg_count = rays_o_in_mask_i.shape[0]
        fg_index = np.random.choice(range(fg_count), inner_count)  # pick outer_count in mask
        rays_o_in_pick, rays_v_in_pick, true_color_in_pick = rays_o_in_mask_i[fg_index], rays_v_in_mask_i[fg_index], \
            rays_color_in_mask_i[fg_index]  # pick out rays
        mask_in_pick = np.ones_like(rays_o_in_pick[:, 0]).reshape(-1, 1)
        rays_o_in_pick, rays_v_in_pick, true_color_in_pick, mask_in_pick = \
            torch.from_numpy(rays_o_in_pick.astype(np.float32)).to(self.device), \
                torch.from_numpy(rays_v_in_pick.astype(np.float32)).to(self.device), \
                torch.from_numpy(true_color_in_pick.astype(np.float32)).to(self.device), \
                torch.from_numpy(mask_in_pick.astype(np.bool_)).to(self.device)
        # rest from gen_random_at, that is randomly gen
        # data_out = (self.gen_random_rays_at(img_idx, outer_count)).detach().cpu().numpy()  # gen_random_rays_at and make it to numpy
        data_out = self.gen_random_rays_at(img_idx, outer_count)
        rays_o_out, rays_v_out, true_rgb_out, mask_out = data_out[:, :3], data_out[:, 3: 6], data_out[:, 6: 9], data_out[:, 9: 10]
        # import pdb
        # pdb.set_trace()
        rays_o, rays_v, rays_rgb, rays_mask = torch.cat([rays_o_in_pick, rays_o_out], dim=0), \
            torch.cat([rays_v_in_pick, rays_v_out], dim=0), torch.cat([true_color_in_pick, true_rgb_out], dim=0), torch.cat([mask_in_pick, mask_out], dim=0)

        return rays_o, rays_v, rays_rgb, rays_mask

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().to(self.device).numpy()
        pose_1 = self.pose_all[idx_1].detach().to(self.device).numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d, center=torch.zeros(3).cuda(), radius=1.0): # this should be set from the org setting conf or json
        if self.sphere_center is not None:
            center = self.sphere_center
            radius = self.radius
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)  #
        # print("running on center and radius " + str(center) + " " + str(radius))
        # b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        # mid = 0.5 * (-b) / a
        #
        b_2 = torch.sum((rays_o - center) * rays_d, dim=-1, keepdim=True)
        mid = (-b_2) / a

        near = mid - radius
        far = mid + radius
       # import pdb
        # pdb.set_trace()
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
