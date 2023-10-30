import taichi as ti
from argparse import ArgumentParser
import numpy as np
import os
import trimesh
import torch
import math
import random
from pathlib import Path

ti.init(arch=ti.cpu, flatten_if=True, debug=True)

# quaternion helper functions
@ti.func
def quat_mul(a, b)->ti.Vector:
    return ti.Vector([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                      a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
                      a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]])

@ti.func
def quat_mul_scalar(a, b)->ti.Vector:
    return ti.Vector([a[0] * b, a[1] * b, a[2] * b, a[3] * b])

@ti.func
def quat_add(a, b)->ti.Vector:
    return ti.Vector([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])

@ti.func
def quat_subtraction(a, b)->ti.Vector:
    return ti.Vector([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])

@ti.func
def quat_normal(a)->ti.f32:
    return ti.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3])

@ti.func
def quat_conj(a)->ti.Vector:
    return ti.Vector([a[0], -a[1], -a[2], -a[3]])

@ti.func
def quat_rotate_vec(q, v)->ti.Vector:
    p = ti.Vector([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, p), quat_conj(q))[1:]

@ti.func
def quat_to_matrix(q)->ti.Matrix:
    q = q.normalized()
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ti.Matrix([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                      [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                      [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])

@ti.func
def quat_inverse(q)->ti.Vector:
    # the inverse of a quaternion is its conjugate divided by its norm
    norm_squared = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    return ti.Vector([q[0], -q[1], -q[2], -q[3]]) / norm_squared

@ti.func
def Get_Cross_Matrix(a)->ti.Matrix:
    return ti.Matrix([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    # A = ti.Matrix.zero(dt=ti.f32, n=4, m=4)
    # A[0, 0] = 0
    # A[0, 1] = -a[2]
    # A[0, 2] = a[1]
    # A[1, 0] = a[2]
    # A[1, 1] = 0
    # A[1, 2] = -a[0]
    # A[2, 0] = -a[1]
    # A[2, 1] = a[0]
    # A[2, 2] = 0
    # A[3, 3] = 1
    # return A

# the euler angle is in degree, we first conver it to radian
def form_euler(euler_angle):
    # convert euler angle to quaternion
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    phi = math.radians(euler_angle[0] / 2)
    theta = math.radians(euler_angle[1] / 2)
    psi = math.radians(euler_angle[2] / 2)

    w = math.cos(phi) * math.cos(theta) * math.cos(psi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
    x = math.sin(phi) * math.cos(theta) * math.cos(psi) - math.cos(phi) * math.sin(theta) * math.sin(psi)
    y = math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.cos(theta) * math.sin(psi)
    z = math.cos(phi) * math.cos(theta) * math.sin(psi) - math.sin(phi) * math.sin(theta) * math.cos(psi)

    return [w, x, y, z]

@ti.func
def vec_to_quat(vec):
    return ti.Vector([0.0, vec[0], vec[1], vec[2]])

@ti.func
def quat_to_vec(quat):
    return ti.Vector([quat[1], quat[2], quat[3]])

@ti.func
def get_current_position(initial_vertex_position, quaternion, translation, initial_mass_center)->ti.Vector:
    # Step 1: Calculate initial offset
    initial_offset = initial_vertex_position - initial_mass_center

    # Step 2: Apply rotation using quaternion
    rotated_offset = quat_mul(quat_mul(quaternion, vec_to_quat(initial_offset)), quat_inverse(quaternion))

    # Step 3: Apply translation
    current_position = quat_to_vec(rotated_offset) + translation

    return current_position
    
@ti.data_oriented
class rigid_body_simulator:

    current_frame = 0
    T = ti.Matrix.field(4, 4, dtype=ti.f32, shape=1)
    frame_dt = 1.0 / 60.0
    substep = 10
    dt = frame_dt / substep

    train_iters = 100
    learning_rate = 0.1
    TARGET = ti.Vector([random.random(), random.random()*0.5, random.random()], dt=ti.f32)

    def __init__(self, mesh_file_name, options):
        assert options is not None
        assert mesh_file_name is not None, 'mesh_file_name is None. You need to privide a mesh file name.'
            # floor
        floor_vertices =  np.array([[-5.0, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, 5.0], [5.0, 0.0, -5.0]])
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]]).flatten()
        self.floor_vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.floor_faces = ti.field(dtype=ti.i32, shape=6)
        self.floor_vertices.from_numpy(floor_vertices)
        self.floor_faces.from_numpy(floor_faces)
        # load mesh
        self.mesh = trimesh.load(mesh_file_name)
        vertices = np.array(self.mesh.vertices, dtype=np.float32) - self.mesh.center_mass
        faces = np.array(self.mesh.faces, dtype=np.int32)
        self.mass_center = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.mass_center[None] = ti.Vector(self.mesh.center_mass, dt=ti.f32)

        if options['frames'] is not None:
            self.frames = options['frames']
        
        if options['transform'] is not None:
            self.initial_translation = ti.Vector(options['transform'][0:3], dt=ti.f32)
            self.initial_quat = ti.Vector(form_euler(options['transform'][3:6]), dt=ti.f32)
        else:
            self.initial_quat = ti.Vector([1.0, 0.0, 0.0, 0.0])
            self.initial_translation = ti.Vector([0.0, 0.0, 0.0])

        print('mass_center', self.mass_center[None] + self.initial_translation)

        self.x = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        ti.root.dense(ti.i, self.mesh.vertices.shape[0]).place(self.x, self.x.grad)
        self.J = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.v_out = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.omega_out = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.num_collision = ti.field(dtype=ti.i32)
        self.sum_position = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.contact_normal = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad = True)
        ti.root.place(self.J, self.J.grad, \
                      self.v_out, self.v_out.grad, \
                      self.omega_out, self.omega_out.grad, \
                      self.num_collision, 
                      self.sum_position, self.sum_position.grad)

        # translation 
        self.mass = ti.field(dtype=ti.f32, shape=())
        self.mass[None] = 0.0
        self.inv_mass = 0.0
        # self.force = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        # self.torque = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)

        self.sdf_value = ti.field(dtype=ti.f32)
        self.sdf_grad = ti.Vector3.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, self.mesh.faces.shape[0]).place(self.sdf_value, self.sdf_grad)
        self.contact_sdf = ti.field(dtype=ti.f32, shape=())
        self.contact_normal = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.v = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.init_v = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
        self.init_omega = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
        # rotation
        self.omega = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.inertia_referance = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.inertial = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        # the indices is constant for all frames, so we don't need to store it in the frame loop, but only in the init_state function
        # and we won't need the grad of indices, so we don't need to set needs_grad=True
        self.indices = ti.field(dtype=ti.i32)
         # transformation information for rigid body
        self.quat = ti.Vector.field(4, dtype=ti.f32, needs_grad=True) 
        self.translation = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.v_in = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.omega_in = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.v_out = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.omega_out = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.J = ti.Vector.field(3, dtype=ti.f32, needs_grad=True)
        self.inertial = ti.Matrix.field(3, 3, dtype=ti.f32, needs_grad=True)
        # place data
        ti.root.dense(ti.j, 3 * self.mesh.faces.shape[0]).place(self.indices)
        particles = ti.root.dense(ti.i, self.frames * self.substep)
        # we only store these variables in the the mass center
        particles.place(self.quat, self.quat.grad, self.translation, self.translation.grad,
                        self.omega, self.omega.grad, self.v, self.v.grad, \
                        self.v_in, self.v_in.grad, self.omega_in, self.omega_in.grad, \
                        self.v_out, self.v_out.grad, self.omega_out, self.omega_out.grad, \
                        self.J, self.J.grad, self.inertial, self.inertial.grad)
                        
        # conbvert mesh to taichi data structure
        self.init_state(vertices, faces)
        self.inv_mass = 1.0 / self.mass[None]

        # rigid body parameters
        self.gravity = ti.Vector([0.0, 0.0, -9.8])
        # params need to be optimized
        self.ke = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.mu = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.linear_damping = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.angular_damping = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        assert options['ke'] is not None, 'ke is None. You need to privide a ke value.'
        self.ke[None] = options['ke']
        assert options['mu'] is not None, 'mu is None. You need to privide a mu value.'
        self.mu[None] = options['mu']
        assert options['linear_damping'] is not None, 'linear_damping is None. You need to privide a linear_damping value.'
        self.linear_damping[None] = options['linear_damping']
        assert options['angular_damping'] is not None, 'angular_damping is None. You need to privide a angular_damping value.'
        self.angular_damping[None] = options['angular_damping']

        self.loss = ti.field(dtype=ti.f32, needs_grad=True)
        ti.root.place(self.loss, self.loss.grad)

        # set up ggui
        #create a window
        self.window = ti.ui.Window(name='Rigid body dynamics', res=(1280, 720), fps_limit=60, pos=(150,150))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(1,2,3)
        self.camera.lookat(0,0,0)
        self.camera.up(0,1,0)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.scene.set_camera(self.camera)
        # simulation flag
        self.pause = True
        self.device = 'cuda:0'

    def set_init_translation(self, translation:ti.types.ndarray()):
        for i in range(3):
            self.initial_translation[i] = translation[i]
    
    def set_init_quat(self, quat:ti.types.ndarray()):
        for i in range(4):
            self.initial_quat[i] = quat[i]

    def run(self):
        i = 0
        while self.window.running:
            if self.window.is_pressed(ti.ui.LEFT, 'b'):
                self.pause = not self.pause
            if not self.pause:
        # for i in range(self.substep * self.frames - 1):
                self.clear()
                self.step(i)
                ti.sync()
                i += 1
            if i % self.substep == 0:
                # print('x shape', self.x.shape)
                self.get_transform_matrix(i)
                self.render()
                if i > self.frames * self.substep - 1:
                    self.pause = True

    @ti.kernel
    def get_transform_matrix(self, f:ti.i32):
        R = quat_to_matrix(self.quat[f])
        T = ti.Matrix.identity(ti.f32, 4)
        T[0, 3] = self.translation[f][0]
        T[1, 3] = self.translation[f][1]
        T[2, 3] = self.translation[f][2]
        T[0:3, 0:3] = R
        self.T[0] = T

    def render(self, frame=0):
                 
        self.camera.track_user_inputs(self.window, movement_speed=0.05, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(1,2,3), color=(1, 1, 1))
        # draw the floor
        self.scene.mesh(self.floor_vertices, self.floor_faces, color=(0.5, 0.5, 0.5),show_wireframe=True)
        self.scene.mesh_instance(self.x, self.indices, color=(0.5, 0.5, 0.5),show_wireframe=True, transforms=self.T)
        # self.scene.mesh(self.x, self.indices, color=(0.5, 0.5, 0.5),show_wireframe=True)
        self.canvas.scene(self.scene)
        self.window.show()

    
    @ti.kernel
    def init_state(self, vertices:ti.types.ndarray(), faces:ti.types.ndarray()):
        for i in range(vertices.shape[0]):
            self.x[i] = ti.Vector([vertices[i, 0], vertices[i, 1], vertices[i, 2]], dt=ti.f32)
        for i in range(faces.shape[0]):
            for j in ti.static(range(3)):
                self.indices[i * 3 + j] = faces[i, j]

        # set initial transformation
        self.quat[0] = self.initial_quat
        self.translation[0] = ti.Vector([0.0, 0.0, 0.0]) + self.initial_translation

        # set initial velocity
        self.init_v[None] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.init_omega[None] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32) 
        # set initial rotation
        self.inertia_referance[None] = ti.Matrix.zero(ti.f32, 3, 3)
        # calculate ref inertia (frame 0)
        mass = 1.0
        for i in range(vertices.shape):
            ti.atomic_add(self.mass[None], mass)
            r = self.x[i]
            # inertia = \sum_{i=1}^{n} m_i (r_i^T r_i I - r_i r_i^T)  https://en.wikipedia.org/wiki/List_of_moments_of_inertia
            # as r_i is a col vector, r_i^T is a row vector, so r_i^T r_i is a scalar (actually is dot product)
            I_i = mass * (r.dot(r) * ti.Matrix.identity(ti.f32, 3) - r.outer_product(r))
            ti.atomic_add(self.inertia_referance[None], I_i)

    @ti.kernel
    def set_v(self):
        self.v[0] = self.init_v[None]
        self.omega[0] = self.init_omega[None]

    @ti.kernel
    def pre_compute(self, f:ti.i32):
        # collision Impulse
        self.sum_position[None] = ti.Vector([0.0, 0.0, 0.0])
        self.num_collision[None] = 0

    @ti.kernel
    def collision_detect(self, f:ti.i32):
        for i in range(self.x.shape[0]):
            v_out = (self.v[f] + self.dt * self.gravity) * self.linear_damping[None]
            omega_out = self.omega[f] * self.angular_damping[None]
            ri = self.x[i]
            # xi = self.translation[f] + quat_to_matrix(self.quat[f]) @ ri
            if self.sdf_value[i]:
                vi = v_out + omega_out.cross(quat_to_matrix(self.quat[f]) @ ri)
                if vi.dot(self.sdf_grad[i]) < 0.0:
                    ti.atomic_add(self.num_collision[None], 1)
                    ti.atomic_add(self.sum_position[None], ri)

    @ti.kernel
    def compute_collision_point(self, f:ti.i32, contact_position:ti.types.ndarray()):
        if self.num_collision[None] > 0:
            ri = self.sum_position[None] / self.num_collision[None]
            xi = self.translation[f] + quat_to_matrix(self.quat[f]) @ ri

    @ti.kernel
    def update_state(self, f:ti.i32):
        self.v_in[f] = (self.v[f] + self.dt * self.gravity) * self.linear_damping[None]
        self.omega_in[f] = self.omega[f] * self.angular_damping[None]
        Rri_mat = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dt=ti.f32)
        if self.num_collision[None] > 0:
            ri = self.sum_position[None] / self.num_collision[None]
            Rri = quat_to_matrix(self.quat[f]) @ ri
            vi = self.v_in[f] + self.omega_in[f].cross(quat_to_matrix(self.quat[f]) @ ri)

            # calculate new velocity
            v_i_n = vi.dot(self.contact_normal[None]) * self.contact_normal[None]
            v_i_t = vi - v_i_n
            vi_new = -self.ke[None] * v_i_n + ti.max(1.0 - (self.mu[None] * (1.0 + self.ke[None]) * (v_i_n.norm()/v_i_t.norm())), 0.0) * v_i_t

            # calculate impulse
            I = quat_to_matrix(self.quat[f]) @ self.inertia_referance[None] @ quat_to_matrix(self.quat[f]).transpose()
            Rri_mat = Get_Cross_Matrix(Rri)
            k = ti.Matrix([[self.inv_mass, 0.0, 0.0],\
                           [0.0, self.inv_mass, 0.0],\
                           [0.0, 0.0, self.inv_mass]]) - Rri_mat @ I.inverse() @ Rri_mat
            self.J[f] = k.inverse() @ (vi_new - vi)
            self.inertial[f] = I
            # J = F · Δt = m · Δv,  F = m · Δv / Δt = J / Δt
            # torque = r × F = r × (J / Δt) = (r × J) / Δt
            # Δω = I^(-1) · torque · Δt = I^(-1) · (r × J) / Δt · Δt = I^(-1) · (r × J)
            # update velocity
        self.v_out[f] = ti.select(self.num_collision[None] > 0, self.v_in[f] + self.inv_mass * self.J[f], self.v_in[f])
        self.omega_out[f] = ti.select(self.num_collision[None] > 0, self.omega_in[f] + self.inertial[f].inverse() @ Rri_mat @ self.J[f], self.omega_in[f])
        # self.v_out[f] +=  self.inv_mass * self.J[f]
        # self.omega_out[f] += self.inertial[f].inverse() @ Rri_mat @ self.J[f]
        # elif self.num_collision[None] == 0:
        #     self.v_out[f] = v_out
        #     self.omega_out[f] = omega_out
        wt = self.omega_out[f] * self.dt * 0.5
        dq = quat_mul(ti.Vector([0.0, wt[0], wt[1], wt[2]]), self.quat[f])
        self.translation[f + 1] = self.translation[f] + self.dt * self.v_out[f]
        self.omega[f + 1] = self.omega_out[f]
        self.v[f + 1] = self.v_out[f]
        self.quat[f + 1] = (dq + self.quat[f]).normalized()

    @ti.kernel
    def clear(self):
        self.v.fill(0.0)
        self.omega.fill(0.0)
        self.translation.fill(0.0)
        self.quat.fill(0.0)
        self.inertial.fill(0.0)
        self.J.fill(0.0)
        self.v_in.fill(0.0)
        self.omega_in.fill(0.0)
        self.v_out.fill(0.0)
        self.omega_out.fill(0.0)
        self.translation[0] = self.initial_translation
        self.quat[0] = self.initial_quat

    @ti.kernel
    def clear_gradients_at(self, frame:ti.i32):
        self.x.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.v.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.omege.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.quat.grad[frame] = ti.Vector([0.0, 0.0, 0.0, 0.0], dt=ti.f32)
        self.translation.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.inertial.grad[frame] = ti.Matrix.zero(dt=ti.f32, n=3, m=3)
        self.J.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.v_in.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.omega_in.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.v_out.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.omega_out.grad[frame] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)

    @ti.kernel
    def clear_gradients(self):
        # TODO 
        self.init_v.grad[None] = 0.0
        self.init_omega.grad[None] =0.0
        self.x.grad.fill(0.0)
        self.v.grad.fill(0.0)
        self.omega.grad.fill(0.0)
        self.quat.grad.fill(0.0)
        self.translation.grad.fill(0.0)
        self.inertial.grad.fill(0.0)
        self.J.grad.fill(0.0)
        self.v_in.grad.fill(0.0)
        self.omega_in.grad.fill(0.0)
        self.v_out.grad.fill(0.0)
        self.omega_out.grad.fill(0.0)

    @ti.kernel
    def compute_loss(self):
        delta = self.translation[self.frames * self.substep - 1] - self.TARGET
        self.loss[None] = delta.dot(delta)

    # dLdt: translation grad 
    # dLdq: rotation grad
    @ti.kernel
    def set_motion_grad(self, f:ti.i32, dLdt:ti.types.ndarray(), dLdq:ti.types.ndarray()):
        for i in ti.static(range(3)):
            self.translation.grad[f][i] = dLdt[i]
            self.quat.grad[f][i] = dLdq[i]
        self.quat.grad[f][3] = dLdq[3]
        # print(f'frames:{f} translation grad:{self.translation.grad[f]}')
        # print(f'frames:{f} quat grad:{self.quat.grad[f]}')


    @ti.kernel
    def optimized(self):
        self.init_v[None] -= self.init_v.grad[None] * self.learning_rate
        self.init_omega[None] -= self.init_omega.grad[None] * self.learning_rate
        
    @ti.kernel
    def get_simulation_grad(self, translation_grad:ti.types.ndarray(), quat_grad:ti.types.ndarray()):
        for i in ti.static(range(3)):
            translation_grad[i] = self.translation.grad[0][i]
        for j in ti.static(range(4)):
            quat_grad[j] = self.quat.grad[0][j]

    @ti.kernel
    def get_transform(self, f:ti.i32, translation:ti.types.ndarray(), quat:ti.types.ndarray()):
        for i in range(3):
            translation[i] = self.translation[f][i]
            quat[i] = self.quat[f][i]
        quat[3] = self.quat[f][3]

    @ti.kernel
    def set_sdf_infomation(self, sdf_value:ti.types.ndarray(), sdf_grad:ti.types.ndarray()):
        for i in range(self.x.shape[0]):
            self.sdf_value[i] = sdf_value[i]
            for j in ti.static(range(3)):
                self.sdf_grad[i][j] = sdf_grad[i][j]
    @ti.kernel
    def set_collision_sdf_info(self, sdf_value:ti.f32, sdf_grad:ti.types.ndarray()):
        self.contact_sdf[None] = sdf_value
        for i in ti.static(range(3)):
            self.contact_normal[None][i] = sdf_grad[i]

    def forward(self, frame):
        if frame > 0:
            self.set_v()
            # sdf_value = np.zeros([self.x.shape[0]], dtype=np.float32)
            # sdf_grad = np.zeros([self.x.shape[0], 3], dtype=np.float32)
            # contact_position = torch.zeros([3], dtype=torch.float32, device=self.device)
            # contact_sdf_value = torch.zeros([3], dtype=torch.float32, device=self.device)
            # contact_normal = torch.zeros([3], dtype=torch.float32, device=self.device)
            for i in range(self.substep * (frame-1), self.substep * frame):
                self.pre_compute(i)
                # TODO: get sdf value and grad
                # sdf_value, sdf_value_grad = get_sdf_info(self.x[i])

                # self.set_sdf_infomation(sdf_value=sdf_value, sdf_value_grad=sdf_value_grad)
                self.collision_detect(i)
                
                # TODO: get contact position
                # self.compute_collision_point(f=i, contact_position=contact_position)

                # TODO: get contact sdf and normal
                # contact_sdf_value, contact_normal = get_contact_sdf_info(contact_position)
                # self.set_collision_sdf_info(sdf_value=contact_sdf_value[0], sdf_value_grad=contact_normal)
                
                # collision responses
                self.update_state(i)
                ti.sync()
            # self.compute_loss()n
            translation = np.zeros([3],dtype=np.float32)
            quat = np.zeros([4], dtype=np.float32)
            self.get_transform(f=i, translation=translation, quat=quat)
            return torch.from_numpy(translation).to(self.device), \
                torch.from_numpy(quat).to(self.device)
        else:
            return None, None

    def backward(self, frame:int):
        if frame > 0:
            for i in reversed(range((frame - 1) * self.substep, frame * self.substep)):
                # recalcute sdf information for computing grad
                # self.pre_compute(i)
                # TODO: get sdf value and grad
                # sdf_value, sdf_value_grad = get_sdf_info(self.x[i])

                # self.set_sdf_infomation(sdf_value=sdf_value, sdf_value_grad=sdf_value_grad)
                # self.collision_detect(i)

                # self.compute_collision_point(f=i, contact_position=contact_position)

                # TODO: get contact sdf and normal
                # contact_sdf_value, contact_normal = get_contact_sdf_info(contact_position)
                # self.set_collision_sdf_info(sdf_value=contact_sdf_value[0], sdf_value_grad=contact_normal)
            
                self.update_state.grad(i)
                self.collision_detect.grad(i)
                self.pre_compute.grad(i)
            self.set_v.grad()
        else:
            translation_grad = np.zeros([3],dtype=np.float32)
            quat_grad = np.zeros([4], dtype=np.float32)
            # init_v_grad = np.zeros([3], dtype=np.float32)
            # init_omega_grad = np.zeros([3], dtype=np.float32)
            self.get_simulation_grad(translation_grad=translation_grad, quat_grad=quat_grad) 
            init_v_grad = np.array(self.init_v.grad[None])
            init_omega_grad = np.array(self.init_omega.grad[None])
            ke_grad = np.array(self.ke.grad[None], dtype=np.float32)
            mu_grad = np.array(self.mu.grad[None], dtype=np.float32)
            return torch.from_numpy(init_v_grad).to(self.device), \
                    torch.from_numpy(init_omega_grad).to(self.device), \
                    torch.from_numpy(ke_grad).to(self.device), \
                    torch.from_numpy(mu_grad).to(self.device), \
                    torch.from_numpy(translation_grad).to(self.device), \
                    torch.from_numpy(quat_grad).to(self.device)
    
    @ti.kernel
    def set_init_v(self, v:ti.types.ndarray()):
        for i in ti.static(range(3)):
            self.init_v[None][i] = v[i]

    def train(self):
        loss = []
        for iters in range(self.train_iters):
            self.clear()
            self.clear_gradients()
            # with ti.ad.Tape(loss=self.loss, validation=False):
            l = self.forward()
            self.loss.grad[None] = 1.0
            self.backwards()
            if l < 1e-6:
                break
            loss.append(l)
            self.optimized()
            if iters % 10 == 0:
                print('grad', self.init_v.grad[None], self.init_omega.grad[None])
                print('after optimized:')
                print('v:', self.init_v[None])
                print('omega:', self.init_omega[None])
                print('loss:', loss[-1])
                print(f'TAGET:{self.TARGET}, final position:{self.translation[self.frames * self.substep - 1]}')
        # for i in range(self.substep * self.frames - 1):
        #     if i % self.substep == 0:
        #         self.get_transform_matrix(i)
        #         self.render()
        print('optimized result')
        self.clear()
        self.forward()
        print(f'TAGET:{self.TARGET}, final position:{self.translation[self.frames * self.substep - 1]}')
        for i in range(self.substep * self.frames - 1):
            if i % self.substep == 0:
                self.get_transform_matrix(i)
                self.render()
        import matplotlib.pyplot as plt
        # title
        plt.title('loss')
        plt.plot(loss)
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--bouncing_coefficient', type=float, default=0.5)
    parser.add_argument('--friction_coefficient', type=float, default=0.2)
    parser.add_argument('--linear_damping', type=float, default=0.999)
    parser.add_argument('--angular_damping', type=float, default=0.998)
    parser.add_argument('--translation', type=float, nargs='+', default=[0.0, 0.0, 0.0], help='translation')
    parser.add_argument('--rotation', type=float, nargs='+', default=[0.0, 0.0, 0.0], help='euler angle in degree')

    args = parser.parse_args()
    params = dict(frames=100,
                  ke=args.bouncing_coefficient,
                  mu=args.friction_coefficient,
                  transform=args.translation + args.rotation,
                  linear_damping=args.linear_damping,
                  angular_damping=args.angular_damping)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = Path(current_directory) / 'test.obj'
    
    robot = rigid_body_simulator(file_name,  {'frames': 100 ,
                                    'ke': 0.5,
                                    'mu': 0.2, 
                                    'transform': [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                    'linear_damping': 0.999,
                                    'angular_damping': 0.998})
    robot.train()