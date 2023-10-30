
import os
import time
import json
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from tqdm import trange
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.rigid_body import rigid_body_simulator
from models.common import *
from argparse import ArgumentParser
from exp_runner import Runner

def load_cameras_and_images(images_path, masks_path, camera_params_path, frames_count, with_fixed_camera=False, pic_mode="png"): # assmue load from a json file
    print("---------------------Loading image data-------------------------------------")

    with open(camera_params_path, "r") as json_file:
        camera_params_list = json.load(json_file)   
    global_K, global_M = None, None

    if with_fixed_camera:  # in this case, we assume all frames share with the same K & M
        global_K = camera_params_list['K']
        global_M = camera_params_list['M']
        
        
    images, masks, cameras_K, cameras_M = [], [], [], []  # cameras_M should be c2w mat
    for i in range(1, frames_count+1):
        picture_name = f"{i:04}"
        image_I_path = images_path + "/" + picture_name + "." + pic_mode
        image = cv.imread(image_I_path)
        images.append(np.array(image)) 
        mask_I_path = masks_path + "/mask_" + picture_name + "." + pic_mode
        mask = cv.imread(mask_I_path)
        masks.append(np.array(mask))
        if with_fixed_camera:
            cameras_K.append(np.array(global_K))
            cameras_M.append(np.array(global_M))
        else:
            cameras_name = str(i)
            camera_K = camera_params_list[cameras_name + "_K"]
            cameras_K.append(np.array(camera_K))
            camera_M = camera_params_list[cameras_name + "_M"]
            cameras_M.append(np.array(camera_M))
    

    print("---------------------Load image data finished-------------------------------")
    return images, masks, cameras_K, cameras_M  # returns numpy arrays

def generate_rays_with_K_and_M(transform_matrix, intrinsic_mat, W, H, resolution_level=1):  # transform mat should be c2w mat
    transform_matrix = torch.from_numpy(transform_matrix.astype(np.float32)).to('cuda:0')# add to cuda
    intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
    intrinsic_mat_inv = torch.from_numpy(intrinsic_mat_inv.astype(np.float32)).to('cuda:0')
    tx = torch.linspace(0, W - 1, W // resolution_level)
    ty = torch.linspace(0, H - 1, H // resolution_level)
    pixels_x, pixels_y = torch.meshgrid(tx, ty)
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
    p = torch.matmul(intrinsic_mat_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    rays_v = torch.matmul(transform_matrix[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_o = transform_matrix[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3, start from transform
    return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H W 3

def generate_all_rays(imgs, masks, cameras_K, cameras_c2w, W_all, H_all):
    # this function generate rays from given img and camera_K & c2w, also returns rays_gt as reference
    # assume input raw images are 255-uint, this function transformed to 1.0-up float0
    # stack the result into [frames_count, W*H, 3] format, assume all frames has the same resolution with W, H
    frames_count = len(imgs)
    rays_o_all, rays_v_all, rays_gt_all, rays_mask_all = [], [], [], []
    for i in range(0, frames_count):
        rays_gt, rays_mask = imgs[i], masks[i] ## check if is  H, W, 3
        rays_gt = rays_gt / 256.0
        rays_gt = rays_gt.reshape(-1, 3)
        rays_gt = torch.from_numpy(rays_gt.astype(np.float32)).to("cuda:0")
        rays_mask = rays_mask / 255.0 
        rays_mask = np.where(rays_mask > 0, 1, 0).reshape(-1, 3)
        rays_mask = torch.from_numpy(rays_mask.astype(np.bool_)).to("cuda:0")        
        rays_o, rays_v = generate_rays_with_K_and_M(cameras_c2w[i], cameras_K[i], W_all, H_all) ## check if is  H, W, 3
        rays_o = rays_o.reshape(-1, 3)
        rays_v = rays_v.reshape(-1, 3)
        rays_o_all.append(rays_o)
        rays_v_all.append(rays_v)
        rays_gt_all.append(rays_gt)
        rays_mask_all.append(rays_mask)
    # returns rays_o_all, rays_v_all, rays_gt_all, rays_mask_all formulate by frames
    return rays_o_all, rays_v_all, rays_gt_all, rays_mask_all

class GenshinStart(torch.nn.Module):
    def __init__(self, setting_json_path):
        super(GenshinStart, self).__init__()
        self.flag = 0
        self.device = 'cuda:0'
        with open(setting_json_path, "r") as json_file:
            motion_data = json.load(json_file)
        static_mesh = motion_data["static_mesh_path"]
        option = {'frames': motion_data["frame_counts"],
                  'frame_dt': motion_data["frame_dt"], 
                  'ke': 0.1,
                  'mu': 0.8,
                  'transform': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  'linear_damping': 0.999,
                  'angular_damping': 0.998}
        self.physical_simulator = rigid_body_simulator(static_mesh, option)
        self.physical_simulator.set_init_quat(np.array(motion_data['R0'], dtype=np.float32))
        self.physical_simulator.set_init_translation(np.array(motion_data['T0'], dtype=np.float32))
        self.max_frames = 1
        self.translation, self.quaternion= [], []
        self.static_object_conf_path =    motion_data["neus_object_conf_path"]
        self.static_object_name =     motion_data['neus_static_object_name']
        self.static_object_continue =     motion_data['neus_static_object_continue']

        self.static_background_conf_path = motion_data["neus_background_conf_path"]       
        self.static_background_name = motion_data['neus_static_background_name']
        self.static_background_continue = motion_data['neus_static_background_continue']
        # in this step, use 'train' mode as default
        self.runner_object = \
            Runner.get_runner(self.static_object_conf_path, self.static_object_name, self.static_object_continue) 
        # self.runner_background = \
        #     Runner.get_runner(self.static_background_conf_path, self.static_background_name, self.static_background_continue)
        
        with torch.no_grad():
            self.init_mu = torch.zeros([1], requires_grad=True, device=self.device)
            self.init_ke = torch.zeros([1], requires_grad=True, device=self.device)
            self.init_translation = torch.zeros([3], requires_grad=True, device=self.device)
            self.init_quaternion = torch.zeros([4], requires_grad=True, device=self.device)
            self.init_v = torch.zeros([3], requires_grad=True, device=self.device)
            self.init_omega = torch.zeros([3], requires_grad=True, device=self.device)
        # TODO: need to be completed， should be torch tensor here
        self.batch_size = motion_data["batch_size"]
        self.frame_counts = motion_data["frame_counts"]
        self.images_path = motion_data["images_path"]
        self.masks_path = motion_data["masks_path"]
        self.camera_setting_path = motion_data["cameras_setting_path"]
        self.with_fixed_camera = motion_data["with_fixed_camera"]
        images, masks, cameras_K, cameras_M = load_cameras_and_images(self.images_path, self.masks_path, self.camera_setting_path, self.frame_counts, with_fixed_camera=self.with_fixed_camera)
        
        self.cameras_K, self.cameras_M = cameras_K, cameras_M
        self.W, self.H = images[0].shape[1], images[0].shape[0]
        # images, masks, cameras_K, cameras_M = images[9:18], masks[9:18], cameras_K[9:18], cameras_M[9:18]  # TO DO: temp debug
        # self.frame_counts = 5
        with torch.no_grad():
            self.rays_o_all, self.rays_v_all, self.rays_gt_all, self.rays_mask_all = generate_all_rays(images, masks, cameras_K, cameras_M, self.W, self.H)

    def get_transform_matrix(self, translation, quaternion):
        w, x, y, z = quaternion
        w, x, y, z = quaternion
        w, x, y, z = quaternion

        transform_matrix = torch.tensor([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w), translation[0]],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w), translation[1]],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2), translation[2]],
            [0, 0, 0, 1.0]
        ], device=self.device, requires_grad=True, dtype=torch.float32)
        transform_matrix_inv = torch.inverse(transform_matrix)   # make an inverse
        transform_matrix_inv.requires_grad_(True)
        return transform_matrix, transform_matrix_inv

    def forward(self, max_f:int):       
        pbar = trange(1, max_f) 
        pbar.set_description('\033[5;41mForward\033[0m')
        global_loss = 0
        self.physical_simulator.clear()
        self.physical_simulator.clear_gradients()
        with torch.no_grad():
            self.physical_simulator.set_init_v(v=self.init_v)
            self.physical_simulator.set_collision_coeff(mu=self.init_mu, ke=self.init_ke)
        print_info(f'init v: {self.init_v}')
        for i in pbar:
            print_blink(f'frame id : {i}')               

            orgin_mat_c2w = torch.from_numpy(self.cameras_M[i].astype(np.float32)).to(self.device)
            # orgin_mat_K_inv = torch.from_numpy(np.linalg.inv(self.cameras_K[i].astype(np.float32))).to(self.device)
            # translation = torch.nn.Parameter(torch.Tensor([0, 0, 0]), requires_grad=True).to(self.device)
            # quaternion = torch.nn.Parameter(torch.Tensor([1, 0, 0, 0]), requires_grad=True).to(self.device)
            translation, quaternion = self.physical_simulator.forward(i)
            translation.requires_grad_(True)
            quaternion.requires_grad_(True)
            self.translation.append(translation)
            self.quaternion.append(quaternion)
            print_info(f'frame:{i}, translation: {translation}, quaternion: {quaternion}')
            # import pdb; pdb.set_trace()
            # camera_pos = torch.zeros((4,4), device=self.device, requires_grad=True)
            # transform_matrix, transform_matrix_inv = self.get_transform_matrix(translation=translation, quaternion=quaternion)
            # camera_pos = torch.matmul(transform_matrix_inv, orgin_mat_c2w)
            rays_gt, rays_mask, rays_o, rays_d = self.rays_gt_all[i], self.rays_mask_all[i], self.rays_o_all[i], self.rays_v_all[i]
            # gW, gH = self.global_W, self.global_H
            # tx = torch.linspace(0, gW - 1, self.W)
            # ty = torch.linspace(0, gH - 1, self.H)
            # pixels_x, pixels_y = torch.meshgrid(tx, ty)
            # p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
            # # we assume that the fx fy in all intrinsic mats are the same, so use the first intrinsics_all_inv to gen rays
            # p = torch.matmul(orgin_mat_K_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
            # rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
            # rays_d = torch.matmul(camera_pos[None, None, :3, :3], rays_d[:, :, :, None]).squeeze()  # W, H, 3
            # rays_o = camera_pos[None, None, :3, 3].expand(rays_d.shape)  # W, H, 3
            # rays_o = rays_o.transpose(0, 1)
            # rays_d = rays_d.transpose(0, 1)  # H W 3
            # rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
            # generate rays_o rays_d trmperarly here
            rays_mask = torch.ones_like(rays_mask)  # full img render
            rays_o, rays_d, rays_gt = rays_o[rays_mask].reshape(-1, 3), rays_d[rays_mask].reshape(-1, 3), rays_gt[rays_mask].reshape(-1, 3)  # reshape is used for after mask, it become [len*3]
            rays_sum = len(rays_o)
            debug_rgb = []
            for rays_o_batch, rays_d_batch, rays_gt_batch in zip(rays_o.split(self.batch_size), rays_d.split(self.batch_size), rays_gt.split(self.batch_size)):
                near, far = self.runner_object.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = None
                # this render out contains grad & img loss, find out its reaction with phy simualtion
                render_out = self.runner_object.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch, near=near, far=far, 
                                                                        R=quaternion, T=translation, camera_c2w=orgin_mat_c2w,
                                                                        cos_anneal_ratio=self.runner_object.get_cos_anneal_ratio(),background_rgb=background_rgb)
                color_fine = render_out["color_fine"]
                color_error = (color_fine - rays_gt_batch)
                # print("render at o & d " + str(rays_o_batch) + "\n" + str(rays_d_batch[0]) + "\n" + str(rays_d_batch[1])) 

                debug_rgb.append(color_fine.clone().detach().cpu().numpy())
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error),
                                            reduction='sum') / rays_sum / max_f  # normalize
                global_loss += color_fine_loss.clone().detach()
                color_fine_loss.backward()  # img_loss for refine R & T
                # import pdb; pdb.set_trace()

                torch.cuda.synchronize()
                del render_out
            ### img_debug should has same shape as rays_gt
            debug_rgb = (np.concatenate(debug_rgb, axis=0).reshape(-1, 3) * 256).clip(0, 255).astype(np.uint8) 
            W, H, cnt = self.W, self.H, 0
            rays_mask = (rays_mask.detach().cpu().numpy()).reshape(H, W, 3)
            debug_img = np.zeros_like(rays_mask).astype(np.float32)
            for index in range(0, H):
                for j in range(0, W):  
                    if rays_mask[index][j][0]:
                        debug_img[index][j][0] = debug_rgb[cnt][0]
                        debug_img[index][j][1] = debug_rgb[cnt][1]
                        debug_img[index][j][2] = debug_rgb[cnt][2]
                        cnt = cnt + 1
            # debug_img2 = np.ones((W*H, 3)).astype(np.uint8) 
            # debug_img2[:rays_sum, : ] = debug_rgb    
            # debug_img2 = debug_img2.reshape(H, W, 3)      
            print_blink("saving debug image at " + str(i) + " index")
            cv.imwrite("./debug" + str(i) + ".png", debug_img)
            # print_blink("saving debug image2 at " + str(i) + " index")
            # cv.imwrite("./debug" + str(i) + "_.png", debug_img2)
            pbar.set_description(f"[Forward] loss: {global_loss.item()}")
        return global_loss

    def backward(self, max_f:np.int32):
        pbar = trange(1, max_f)
        pbar.set_description('\033[5;30m[Backward]\033[0m')
        for i in pbar:
            f = max_f - i - 1
            with torch.no_grad():
                translation_grad = self.translation[f].grad
                quaternion_grad = self.quaternion[f].grad
                print_info(f'translation grad: {translation_grad}, quaternion grad: {quaternion_grad}')
            if f > 0:
                self.physical_simulator.set_motion_grad(f, translation_grad, quaternion_grad)
                self.physical_simulator.backward(f)
            else:
                v_grad, omega_grad, ke_grad, mu_grad, translation_grad, quaternion_grad = \
                    self.physical_simulator.backward(f)
                self.init_v.backward(retain_graph=True, gradient=v_grad)
                self.init_omega.backward(retain_graph=True, gradient=omega_grad)
                self.init_ke.backward(retain_graph=True, gradient=ke_grad)
                self.init_mu.backward(retain_graph=True, gradient=mu_grad)
                self.init_translation.backward(retain_graph=True, gradient=translation_grad)
                self.init_quaternion.backward(retain_graph=True, gradient=quaternion_grad)

def get_optimizer(mode, genshinStart):
    
    optimizer = None
    if mode == "train_static":
        optimizer = torch.optim.Adam(
            [
                {"params": getattr(), 'lr': 1e-1}
            ]
        )
    elif mode == "train_velocity":
            optimizer = torch.optim.LBFGS(
            [
                {"params": getattr(genshinStart, 'init_v'), 'lr': 1e-1}
            ]
        )
    elif mode == "train_dynamic":
        optimizer = torch.optim.Adam(
            [
                # {"params": getattr(genshinStart,'init_translation'), 'lr': 1e-1},
                # {'params': getattr(genshinStart,'init_quaternion'), 'lr':1e-1},
                {'params':getattr(genshinStart, 'init_mu'), 'lr': 1e-2},
                {'params':getattr(genshinStart, 'init_ke'), 'lr': 1e-1},
                # {"params": getattr(genshinStart, 'init_v'), 'lr': 1e-1}
            ]
            ,
            amsgrad=False
        )

    return optimizer

def train_static(self):
    static = 0 
    # train static object -> export as a obj mesh

    # train static background -> export as a obj mesh

    # also need to train R0 & T0?
    
def train_velocity(self):
    velocity = 1

def train_dynamic(max_f, iters, genshinStart, optimizer, device):
    def train_forward(optimizer):
        optimizer.zero_grad()
        loss = torch.tensor(np.nan, device=device)
        while loss.isnan():
            loss = genshinStart.forward(max_f)
        return loss

    optimizer = get_optimizer('train_dynamic',genshinStart)
    for i in range(iters):
        loss = train_forward(optimizer=optimizer)
        if loss.norm() < 1e-6:
            break
        genshinStart.backward(max_f)
        optimizer.step()
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    print_blink('Genshin Nerf, start!!!')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_dtype(torch.float32)

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = ArgumentParser()
    parser.add_argument('--conf', type=str, default='./dynamic_test/base.json')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    genshinStart = GenshinStart(args.conf)
    optimizer = get_optimizer('train_dynamic', genshinStart=genshinStart)
    if args.mode == "train":
        train_static()
        train_velocity()
        train_dynamic()
    else:
        train_dynamic(5, iters=1000, genshinStart=genshinStart, optimizer=optimizer, device='cuda:0')

    
# python genshin_start.py --mode debug --conf ./dynamic_test/genshin_start.json --case bird --is_continue 
# D:\gitwork\genshinnerf> python genshin_start_copy.py --mode debug --conf ./dynamic_test/genshin_start.json --case bird --is_continue