import torch
import math
# from botorch.utils.sampling import *


class popcorn_interface:
    def __init__(self, bp=60, bm=2, dim=3):
        self.dim = torch.tensor(dim)
        self.bp = torch.tensor(bp)
        self.bm = torch.tensor(bm)
        self.domain_area = torch.pow(2.0, self.dim)

        '''for compute interface'''
        self.r0 = torch.tensor(0.6)
        self.A = torch.tensor(2)
        self.sigma = torch.tensor(0.2)
        self.factor = self.r0 / torch.sqrt(torch.tensor(5.0))

        self.angles_first = [torch.tensor(2 * k * torch.pi / 5) for k in range(5)]
        self.angles_second = [torch.tensor((2 * k - 11) * torch.pi / 5) for k in range(5, 10)]

        # 预计算点
        self.precomputed_points = self.precompute_points()

    def precompute_points(self):
        points = []
        for k in range(12):
            points.append(self.point_k(k))
        return torch.tensor(points, dtype=torch.float32)
        
        # self.point_ks = []
    def dimension(self):
        d = 3
        return d


    def subdomain_areas(self):#change
        domain_area_minus = torch.tensor(1.93776)
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):#change
        interface_length = torch.tensor(6.73656, dtype=torch.float64)
        return interface_length

    def boundary_length(self):#change
        # boundary_length = 2*self.dim
        boundary_length = 24
        return boundary_length
    '''Differentiate value inside and outside the interface'''


    def point_k(self, k):
        # factor = self.r0 / torch.sqrt(torch.tensor(5.0))
        if 0 <= k <= 4:
            angle = self.angles_first[k]  # 转换为 Tensor
            xk = self.factor * 2 * torch.cos(angle)
            yk = self.factor * 2 * torch.sin(angle)
            zk = self.factor * 1
        elif 5 <= k <= 9:
            angle = self.angles_second[k - 5]  # 从预计算的角度中获取
            xk = self.factor * 2 * torch.cos(angle)
            yk = self.factor * 2 * torch.sin(angle)
            zk = -self.factor * 1
        elif k == 10:
            xk, yk, zk = 0, 0, self.r0
        elif k == 11:
            xk, yk, zk = 0, 0, -self.r0
        else:
            raise ValueError("k must be between 0 and 11")
        return xk, yk, zk

    # def S(self, p):
    #     x = p[..., 0]
    #     y = p[..., 1]
    #     z = p[..., 2]
    #     S = torch.zeros_like(x)
    #     for k in range(12):
    #         xk, yk, zk = self.point_k(k)  # 从 point_k 中获取坐标
    #         square = -(torch.square(x - xk) + torch.square(y - yk) + torch.square(z - zk))
    #         S += self.A * torch.exp(square / torch.square(self.sigma))
    #     return S

    # def S(self, p):
    #     diffs = torch.square(p.unsqueeze(1) - self.precomputed_points.unsqueeze(0))
    #     S = self.A * torch.exp(-torch.sum(diffs, dim=2) / torch.square(self.sigma))
    #     return S.sum(dim=1)

    def S(self, p):
        diffs = torch.square(p.unsqueeze(1) - self.precomputed_points.unsqueeze(0))
        S = self.A * torch.exp(-torch.sum(diffs, dim=2) / torch.square(self.sigma))
        return S.sum(dim=1, keepdim=True)  #  [N_f, 1]

    def phi(self, p):
        # r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = torch.norm(p, dim=1, keepdim=True)
        s = r - self.r0 - self.S(p)
        # s = s.reshape(-1, 1)
        # print('s.shape in phi', s.shape)
        return s > 0


    #meet the condition (in interface), return bp, otherwise return bm
    def diffusion_coefficient_judge(self, p): # #pass domain points
        judge_coefficient = torch.where(self.phi(p), self.bp, self.bm)
        return judge_coefficient

    def diffusion_plus(self, p): #p.shape(m, 2)
        m, n = p.size()
        bp = self.bp * torch.ones(m, 1)
        return bp

    def diffusion_minus(self, p):
        m, n = p.size()
        bm = self.bm * torch.ones(m, 1)
        return bm


    '''exact and rhs'''
    def exact_minus(self, p):
        r2 = torch.sum(torch.square(p), dim=1)
        solution_minus = r2.reshape(-1, 1)
        return solution_minus

    def exact_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        solution_plus = x + y + z
        solution_plus = solution_plus.reshape(-1, 1)
        return solution_plus

    def exact_judge(self, p):
        judge_solution = torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))
        return judge_solution

    def rhs_plus(self, p):
        rhs2 = torch.zeros(p.size(0), 1)
        return rhs2

    def rhs_minus(self, p):
        # m, n = p.size()
        # rh1 = -12 * torch.ones(m, 1)
        rhs1 = -12 * torch.ones(p.size(0), 1)
        return rhs1

    def rhs_judge(self, p): #pass domain points
        rhs_judge = torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))
        return rhs_judge

    #BC conditions u = g
    def BC_Dirichlet(self, p):
        BC_real = self.exact_judge(p) #Pass in BC_points--judge--return exact_minus2
        return BC_real

    '''Jump'''

    def interface_normal(self, p):
        r = torch.norm(p, dim=1, keepdim=True)  # [N, 1]
        diffs = p.unsqueeze(1) - self.precomputed_points.unsqueeze(0)  # [N, 12, 3]: each point in p has to - the 12 points
        exp_terms = torch.exp(-torch.sum(diffs ** 2, dim=2) / torch.square(self.sigma))  # [N, 12]
        S = self.A * exp_terms.sum(dim=1, keepdim=True)  # [N, 1]

        S = S.unsqueeze(-1)  # [N, 1, 1]
        gradients = 2 * S * diffs / torch.square(self.sigma)  # [N, 12, 3]

        normals = (p / r + gradients.sum(dim=1))  # [N, 3]
        normals = normals / torch.norm(normals, dim=1, keepdim=True)  # [N, 3]
        return normals

    def function_jump_p(self, p): #pass interface points
        jump_p = self.exact_plus(p) - self.exact_minus(p)
        return jump_p

    def flux_jump_q(self, p): #pass interface points
        m, n = p.size()
        #verison 2
        unv = self.interface_normal(p)
        grad_minus = 2 * p
        grad_plus = torch.ones(m, 3)
        jump_q = (torch.sum(grad_plus * unv, dim=1) * self.bp -
                  torch.sum(grad_minus * unv, dim=1) * self.bm)
        jump_q = jump_q.reshape(-1, 1)
        return jump_q

    '''Data Generation'''
    def interior_points(self, N_omega):
        dimension = 3
        xr = torch.rand(N_omega, dimension) * 2 - 1 #(N_omega, 2) [0,1]
        return xr

    def boundary_points(self, N_b):
        dimension = 3
        xb = torch.rand(2*dimension*N_b, dimension)*2 - 1 #size: (N_b * 4, 2), domain(-1,1) any coordinate value
        #form boudnary
        for i in range(dimension): #i=0-- index of x coordinate; i = 1, index of y coordinate
            xb[2*i*N_b: (2*i+1)*N_b, i] = -1 # x= -1 [0:N_b,0] and y = -1 [2N_b:3N_b,1]
            xb[(2*i+1)*N_b: (2*i+2)*N_b, i] = 1 #x=1 [N_b:2N_b,0] and y = -1 [3N_b:4N_b,1]
        return xb

    # def interface_points(self, N_f): #18%-20%
    #     #version1
    #     x = torch.rand(N_f, 3)*2 - 1
    #     y = torch.rand(N_f, 3)*2 - 1
    #     z = torch.rand(N_f, 3)*2 - 1
    #     r = torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z))
    #
    #     S = torch.zeros(N_f, 3)
    #     for k in range(12):
    #         xk, yk, zk = self.point_k(k)
    #         square = -(torch.square(x - xk) + torch.square(y - yk) + torch.square(z - zk))
    #         S += self.A * torch.exp(square / torch.square(self.sigma))
    #     values = r - self.r0 - S
    #
    #     mask = torch.abs(values) < 0.1
    #     xf = torch.stack([x[mask], y[mask], z[mask]], dim=1)
    #     # print('xf.shape', xf.shape)
    #     #v2
    #     # import matplotlib.pyplot as plt
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # ax.scatter(xf[:, 0].numpy(), xf[:, 1].numpy(), xf[:, 2].numpy(), c='b', s=1)
    #     # plt.show()
    #     return xf

    def interface_points(self, N_f):
        #v1
        p = torch.rand(N_f, 3, device=self.r0.device) * 2 - 1
        r = torch.norm(p, dim=1, keepdim=True)  # [N_f, 1]
        S = self.S(p)  # [N_f, 1]
        mask = torch.abs(r - self.r0 - S) < 0.1  # [N_f, 1]
        mask = mask.squeeze(1)
        xf =  p[mask]

        #v2
        # x = torch.rand(N_f, 3) * 2 - 1
        # y = torch.rand(N_f, 3) * 2 - 1
        # z = torch.rand(N_f, 3) * 2 - 1
        # r = torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z))
        # S = torch.zeros(N_f, 3)
        #
        # for k in range(12):
        #     xk, yk, zk = self.point_k(k)
        #     square = -(torch.square(x - xk) + torch.square(y - yk) + torch.square(z - zk))
        #     S += self.A * torch.exp(square / torch.square(self.sigma))
        # values = r - self.r0 - S
        # mask = torch.abs(values) < 0.1
        # xf = torch.stack([x[mask], y[mask], z[mask]], dim=1)
        # print('xf.shape', xf.shape)

        # import matplotlib.pyplot as plt
        # # 使用 matplotlib 进行 3D 可视化
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # 绘制满足条件的界面点
        # ax.scatter(xf[:, 0], xf[:, 1], xf[:, 2], s=1, c='b')
        # plt.show()
        return  xf# 返回满足条件的点




