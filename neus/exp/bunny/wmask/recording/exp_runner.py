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
from pyhocon import ConfigFactory
from models.dataset_json import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from models.rigid_body import rigid_body_simulator


def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')


def print_ok(*message):
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    print('\033[93m', *message, '\033[0m')


def print_info(*message):
    print('\033[96m', *message, '\033[0m')


def calc_new_pose(setting_path):
    # TODO: read pose and movement from one json file and calc new pose
    # returns new camera pose calculated by that json file
    with open(setting_path, "r") as json_file:
        all_json_data = json.load(json_file)

    # q, t, original_mat = None, None, None
    t, q, original_mat = all_json_data['translation'], all_json_data['rotation'], all_json_data["1_1_M"]
    if q is None:
        print("error at reading setting " + setting_path)
        exit(-1)

    w, x, y, z = q
    rotate_mat = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    transform_matrix = np.zeros((4, 4))
    transform_matrix[0:3, 0:3] = rotate_mat
    transform_matrix[0:3, 3] = t
    transform_matrix[3, 3] = 1.0

    return transform_matrix @ original_mat


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def train_dynamic(self):
        # TODO use render_dynamic to pass img_loss
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'dynamic_logs'))
        return

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def render_novel_image_at(self, camera_pose, resolution_level, intrinsic_inv=None):
        rays_o, rays_d = self.dataset.gen_rays_at_pose_mat(camera_pose, resolution_level=resolution_level,intrinsic_inv=intrinsic_inv)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        # import pdb
        # pdb.set_trace()
        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            print("SCALE MAP")
            print(self.dataset.scale_mats_np[0][0, 0])
            print(self.dataset.scale_mats_np[0][:3, 3][None])

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def save_render_pic_at(self, setting_json_path):
        camera_pose = calc_new_pose(args.render_at_pose_path)
        img = self.render_novel_image_at(camera_pose, 2)
        set_dir, file_name_with_extension = os.path.dirname(setting_json_path), os.path.basename(setting_json_path)
        file_name_with_extension = os.path.basename(setting_json_path)
        case_name, file_extension = os.path.splitext(file_name_with_extension)
        render_path = set_dir + "/" + case_name + ".png"
        print("Saving render img at " + render_path)
        cv.imwrite(render_path, img)
        return

    def render_motion(self, setting_json_path):
        with open(setting_json_path, "r") as json_file:
            motion_data = json.load(json_file)
        if motion_data["frames"] is None:
            print_error("must provide a sequence of motion information")
            exit()
        frames = motion_data["frames"]
        print_info(f"{frames} frames will be rendered.")
        motion_transforms = motion_data["results"]
        original_mat = motion_data["1_1_M"]
        if original_mat == None:
            print_error("static camera information must be provided")
        for i in tqdm(range(1)):
            motion_transform = motion_transforms[i]
            assert i == motion_transform["frame_id"], "invalid frame sequence"
            t, q = motion_transform['translation'], motion_transform['rotation'],
            q = [0.9515, 0.1449, 0.2685, 0.0381]
            t = [0000, 0.0000, 0.8659]

            q = [0.9515, 0.1449, 0.2685, 0.0381]
            t = [0.0000, 0.0000, 0.8671]

            w, x, y, z = q
            rotate_mat = np.array([
                [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
            ])
            transform_matrix = np.zeros((4, 4))
            transform_matrix[0:3, 0:3] = rotate_mat
            transform_matrix[0:3, 3] = t
            transform_matrix[3, 3] = 1.0
            inverse_matrix = np.linalg.inv(transform_matrix)
            camera_pose = np.array(original_mat)

            img = self.render_novel_image_at(camera_pose, 2)
            # img loss
            set_dir, file_name_with_extension = os.path.dirname(setting_json_path), os.path.basename(setting_json_path)
            file_name_with_extension = os.path.basename(setting_json_path)
            case_name, file_extension = os.path.splitext(file_name_with_extension)
            render_path = f"{set_dir}/test_render_motion{i:04d}.png"
            print("Saving render img at " + render_path)
            cv.imwrite(render_path, img)
            print_info(f"finish rendering frame:{i}")

        print_ok(f"{frames} images has been rendered!")

    def train_dynamic_single_frame(self, setting_json_path):
        with open(setting_json_path, "r") as json_file:
            motion_data = json.load(json_file)
        static_mesh = motion_data["static_mesh_path"]

        optimizer = torch.optim.Adam(
            [
                #  {'params':pnerf.nerf.density.parameters(), 'lr': 1e-1}
            ],
            amsgrad=False
        )

        # in the future, it need to be replaced as a set of camera poses from real-world data
        original_mat = motion_data["camera_poses_mat"]
        if original_mat is None:
            print_error("static camera information must be provided")
            original_mat = np.eye((4, 4))
            original_mat[2, 3] = -5
        else:
            original_mat = np.array(original_mat)

        option = {'frames': 2,
                  'ke': 0.1,
                  'mu': 0.8,
                  'transform': [0.0, 0.0, 0.9985088109970093, 0.0, 0.0, 0.0],
                  'linear_damping': 0.999,
                  'angular_damping': 0.998}
        dynamic_observation = rigid_body_simulator(static_mesh, option)
        dynamic_observation.set_init_quat(
            np.array([0.9515485167503357, 0.14487811923027039, 0.2685358226299286, 0.03813457489013672]))
        dynamic_observation.set_init_translation(np.array([0.0, 0.0, 0.9985088109970093]))
        dynamic_observation.clear()
        translation, quat = dynamic_observation.forward()

        ## TODO: the following code needs to be batchfied as :
        # for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
        #     break
        # load the ground truth img
        # use cv2.IMREAD_COLOR to read image
        image = cv.imread('./dynamic_test/transform0001.png')
        image_mask = cv.imread('./dynamic_test/transform0001_mask.png')
        resolution_level = 1
        image_rgb = image / 256.0
        image_mask = (np.array(image_mask) / 255.0)
        # image_rgb = (cv.resize(image_rgb,
        #                        (image_rgb.shape[0] // resolution_level, image_rgb.shape[1] // resolution_level))).clip(
        #     0, 255)  # W, H, 3

        image_rgb = torch.from_numpy((image_rgb).astype(np.float32)).to(self.device)

        if image_mask is None:
            rays_mask = torch.ones_like(image_rgb.reshape(-1, 3))
        else:
            rays_mask = torch.from_numpy(np.where(image_mask > 0, 1, 0)).to(self.device).bool()
        self.dataset.set_image_w_h(image_rgb.shape[1], image_rgb.shape[0])  # change W & H here, index 1 is W, 0 is H
        rays_o, rays_d = self.dataset.gen_rays_at_pose_mat(original_mat,
                                                           resolution_level=resolution_level)  # the shape here is H, W, 3
        # import pdb
        # pdb.set_trace()
        rays_o = rays_o[rays_mask].reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d[rays_mask].reshape(-1, 3).split(self.batch_size)  # similar as pacnerf, cut down the rays
        rays_gt = image_rgb[rays_mask].reshape(-1, 3).split(self.batch_size)

        # now in batch, (H*W-mask_0)/batch, 3
        translation.requires_grad_(True)
        quat.requires_grad_(True)
        out_rgb_fine, color_fine_loss = [], None
        for rays_o_batch, rays_d_batch, rays_gt_batch, rays_mask_batch in zip(rays_o, rays_d, rays_gt, rays_mask):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            # this render out contains grad & img loss, find out its reaction with phy simualtion
            render_out = self.renderer.render_dynamic(rays_o=rays_o_batch, rays_d=rays_d_batch, near=near, far=far,
                                                      T=translation, R=quat
                                                      , cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                      background_rgb=background_rgb)
            color_fine = render_out["color_fine"]
            color_error = (color_fine - rays_gt_batch)
            mask_sum = self.batch_size
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error),
                                        reduction='sum') / mask_sum  # normalize
            color_fine_loss.backward(retain_graph=True)  # img_loss for refine R & T
            # print_info(f'translation_grad:{T_grad}, rotation_grad: {R_grad}')
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            del render_out

        print(translation.grad.shape)

        # img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape(
        #     [image_rgb.shape[0], image_rgb.shape[1], 3]) * 256).clip(0, 255).astype(np.uint8)
        # cv.imwrite('dynamic_train.png', img_fine)
        print_ok('dynamic train has done!')
        return


    def render_novel_image_with_RTKM(self):
        q = [1, 0, 0, -0]
        t = [0.000, 0.0000, 0.11]

        w, x, y, z = q
        rotate_mat = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ])
        transform_matrix = np.zeros((4, 4))
        transform_matrix[0:3, 0:3] = rotate_mat
        transform_matrix[0:3, 3] = t
        transform_matrix[3, 3] = 1.0
        inverse_matrix = np.linalg.inv(transform_matrix)
        original_mat = np.array(
            [[0.99913844, - 0.02643227, - 0.03199565,  0.03332534],
            [0.03194597, - 0.00229971,  0.99948695, - 0.22578363],
            [-0.02649229, - 0.99964796, - 0.00145332, 0.07182068],
            [0.,          0.,          0.,          1.]]
        )
        intrinsic_mat = np.array(
            [[196.04002654133333, 0, 256.14846416266664], [0, 195.57227938666668, 147.136028024], [0, 0, 1]]
        )
        intrinsic_inv = torch.from_numpy(np.linalg.inv(intrinsic_mat).astype(np.float32)).cuda()
        # original_mat = np.eye(4)
        # original_mat[3, :3] = [0.1, 0.1, 0.1]
        # original_mat[3, 3] = 0.2
        camera_pose = np.array(original_mat)
        transform_matrix = inverse_matrix @ camera_pose
        self.dataset.W = 512
        self.dataset.H = 288
        # transform_matrix =transform_matrix.astype(np.float32).cuda()
        img = self.render_novel_image_at(transform_matrix, resolution_level=1, intrinsic_inv=intrinsic_inv)
        # img loss
        # set_dir, file_name_with_extension = os.path.dirname(setting_json_path), os.path.basename(setting_json_path)
        # file_name_with_extension = os.path.basename(setting_json_path)
        # case_name, file_extension = os.path.splitext(file_name_with_extension)
        render_path = os.path.join(self.base_exp_dir, "test.png")
        print("Saving render img at " + render_path)
        cv.imwrite(render_path, img)

    def get_runner(neus_conf_path, case_name, is_continue):
        return Runner(neus_conf_path, mode="train", case=case_name, is_continue=is_continue)


if __name__ == '__main__':
    print('Genshin Nerf, start!!!')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.cuda.set_device(args.gpu)

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--render_at_pose_path', type=str, default="./confs/base_movement.json")
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'render_at':
        runner.save_render_pic_at(args.render_at_pose_path)
    elif args.mode == 'render_motion':
        runner.render_motion(args.render_at_pose_path)
    elif args.mode == 'train_dynamic':
        runner.train_dynamic_single_frame(args.render_at_pose_path)
    elif args.mode == 'render_rtkm':
        runner.render_novel_image_with_RTKM()
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    

#  example cmd in rebuilding:
"""
conda activate neus
cd D:/gitwork/NeuS
D:
python exp_runner.py --mode render_at --conf ./confs/wmask.conf --case bird --is_continue --render_at_pose_path D:/gitwork/genshinnerf/dynamic_test/test_render.json


python exp_runner.py --mode train_dynamic --conf ./confs/wmask.conf --case bird --is_continue --render_at_pose_path D:/gitwork/genshinnerf/dynamic_test/train_dynamic_setting.json
python exp_runner.py --mode render_motion --conf ./confs/wmask.conf --case bird --is_continue --render_at_pose_path D:/gitwork/genshinnerf/dynamic_test/transform.json

python exp_runner.py --mode validate_mesh --conf ./confs/wmask.conf --case bird --is_continue
python exp_runner.py --mode train --conf ./confs/womask.conf --case bird_ss --is_continue
python exp_runner.py --mode train --conf ./confs/wmask_js.conf --case sim_ball --is_continue
python exp_runner.py --mode train --conf ./confs/womask_js_bk.conf --case r_bk --is_continue
python exp_runner.py --mode train --conf ./confs/womask_js_bk_single.conf --case real_world_normal --is_continue
python exp_runner.py --mode train --conf ./confs/wmask_js_bk_single.conf --case real_world_normal
python exp_runner.py --mode train --conf ./confs/womask_js_bk_single_sparse.conf --case real_world_sparse
python exp_runner.py --mode train --conf ./confs/womask_js_bk_single_multi_qrs.conf --case real_world_multi_qrs
python exp_runner.py --mode train --conf ./confs/wmask_js_bk_single_multi_qrs.conf --case real_world_multi_qrs

python exp_runner.py --mode train --conf ./confs/wmask_single_blender.conf --case blender_static_test
python exp_runner.py --mode train --conf ./confs/wmask_single_blender.conf --case blender_high_res
python exp_runner.py --mode train --conf ./confs/womask_blender_high_res.conf --case blender_high_res

python exp_runner.py --mode train --conf ./confs/wmask_js_bk_single_multi_qrs.conf --case rws_object
python exp_runner.py --mode train --conf ./confs/wmask_js_bk_single_multi_qrs.conf --case rws_obstacle --is_continue
python exp_runner.py --mode train --conf ./confs/wmask_js_bk_single_multi_qrs.conf --case rws_object2 


"""
