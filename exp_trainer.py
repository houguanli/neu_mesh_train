"""
TODO:
1 load ckpt for neus, and ext geo
2 use qem to simplify mesh derive from mc-calculation
3 train locally using current data
    3.1 create mesh_grid for neu mesh
    3.2 train those paras
    3.3 render the results when deform the mesh
    #?# ::: how to communicate with the phy simulator?
"""
import json

# those are self-written code
from neus.exp_runner import Runner
from neus.models.dataset import Dataset

from QEM import QEM
import mesh_grid


class NeuMeshRunner:
    def __init__(self, conf_path="./conf.json"):
        # args = None
        with open(conf_path, 'r') as f:
            args = json.load(f)
        neus_conf_path = "./neus/confs/wmask.conf"
        export_mc_mesh_path = "./test/bunny.obj"
        self.neus_pretrained = Runner(conf_path=neus_conf_path, mode=train, is_continue=True)
        self.neus_pretrained.validate_mesh(world_space=False, resolution=512,
                                           specified_path=export_mc_mesh_path)  # use high resolution for init mesh
        simplified_mesh_path = export_mc_mesh_path[:-5] + "_sim.obj"
        self.qem_runner = QEM.QEM_driver(export_mc_mesh_path, threshold=0.3, simplify_ratio=0.05,
                                         output_filepath=simplified_mesh_path)
        self.qem_runner.run_QEM()  # simplify the original mesh into 5%
        self.simplified_mesh = o3d.io.read_triangle_mesh(simplified_mesh_path)
        self.mesh = mesh_grid.MeshGrid(simplified_mesh, 0, "frnn")
        self.batch_size = args["batch_size"]
        self.model_config = {
            "speed_factor": args.training.setdefault("speed_factor", 1.0),
            "D_density": args.setdefault("D_density", 3),
            "D_color": args.setdefault("D_color", 4),
            "W": args.setdefault("W", 256),
            "geometry_dim": args.get("geometry_dim", 32),
            "color_dim": args.setdefault("color_dim", 32),
            "multires_view": args.setdefault("multires_view", 4),
            "multires_d": args.setdefault("multires_d", 8),
            "multires_fg": args.setdefault("multires_fg", 2),
            "multires_ft": args.setdefault("multires_ft", 2),
            "enable_nablas_input": args.setdefault("enable_nablas_input", False),
            "learn_indicator_weight": args.get("learn_indicator_weight", False)
        }
        self.loss_weights = {
            "img": args.training.loss_weights.setdefault("img", 0.0),
            "mask": args.training.loss_weights.setdefault("mask", 0.0),
            "eikonal": args.training.loss_weights.setdefault("eikonal", 0.0),
            "distill_density": args.training.loss_weights.setdefault("distill_density", 0.0),
            "distill_color": args.training.loss_weights.setdefault("distill_color", 0.0),
            "indicator_reg": args.training.loss_weights.setdefault("indicator_reg", 0.1),
        }

        if loss_weights["eikonal"] > 0:
            render_kwargs_train["calc_normal"] = True
        self.neu_mash_model = NeuMesh(mesh, **model_config)
        self.neus_teacher_model = neus_pretrained
        # use images directly from neus

        self.images = self.neus_pretrained.dataset.images
        self.masks = self.neus_pretrained.dataset.masks
        self.neus_dataset = self.neus_pretrained.dataset
        self.iter_step = 0
        self.end_step = 30000
        sself.learning_rate = args.training.setdefault("lr", 1e-5)
        params_to_train = []
        # TODO: finish the paras need to train
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

    def train(self):
        """
        simple train step:
        gen rays from gt(the image)
        Returns:

        """
        for iter_i in tqdm(range(self.end_step)):
            self.iter_step = iter_i  # iter upd
            data = self.neus_dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.neus_dataset.near_far_from_sphere(rays_o, rays_d)
            # those code use sample points simply
            # TODO: generate sample pts and get its color, sdf of teacher_neus & neu_mesh
            dist, pts = get_train_pts(rays_o, rays_d, near, far)
            teacher_render_out = self.neus_pretrained.renderer.half_render(pts,
                                                                           background_rgb=background_rgb,
                                                                           cos_anneal_ratio=self.get_cos_anneal_ratio())  # this method is modified to return all color in pts
            # clac color & sdf loss between teacher model a
            neu_mesh_out = self.neu_mash_model.forward()

            color_loss = None
            teacher_loss_s, teacher_loss_c = None, None  # calc for pts
            ek_loss = None

            loss = None
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # TODO: finish functions in key training slots.
            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

        return

    def render_from_deform_mesh(self):
        return


if __name__ == '__main__':
    print('Genshin , start!!!')
    test_runner = NeuMeshRunner()
    test_runner.train()

"""
TODO: new train step for phy simualtion
pass img loss to get the grad of the detected points in phy simu
phy simu update the position of the 
## imp: the camera pose is fixed, so the new train rays, (sampling points, maybe) can also fixed

"""
