import torch
class ellipsoidal_interface:
    def __init__(self, bp=60, bm=2, dim=3):
        self.dim = torch.tensor(dim)
        self.bp = torch.tensor(bp)
        self.bm = torch.tensor(bm)
        self.domain_area = torch.pow(2.0, self.dim)

    def dimension(self):
        d = 3
        return d

    def subdomain_areas(self):
        domain_area_minus = torch.tensor(1.53379534665262)
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        interface_length = torch.tensor(6.7083943643066, dtype=torch.float64)
        return interface_length

    def boundary_length(self):
        # boundary_length = 2*self.dim
        boundary_length = 24

        return boundary_length
    '''Differentiate value inside and outside the interface'''

    def phi(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        s = 2 * x**2 + 3 * y **2 + 6 * z ** 2 - 1.69
        s = s.reshape(-1, 1)
        # print('s.shape', s.shape)
        return s > 0 #outside interface

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

    def exact_judge(self, p):         #pass domain or boundary points
        judge_solution = torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))
        return judge_solution

    def rhs_plus(self, p): #subdomain 1 point
        m, n = p.size()
        rhs2 = torch.zeros(m, 1)
        return rhs2

    def rhs_minus(self, p):
        m, n = p.size()
        rhs2 = -12 * torch.ones(m, 1)
        return rhs2

    def rhs_judge(self, p): #pass domain points
        rhs_judge = torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))
        return rhs_judge

    #BC conditions u = g
    def BC_Dirichlet(self, p):
        BC_real = self.exact_judge(p) #Pass in BC_points--judge--return exact_minus2
        return BC_real

    '''Jump'''
    def interface_normal(self, p): #pass in: point on the interface
        x = p[..., 0:1]
        y = p[..., 1:2]
        z = p[..., 2:3]
        n_x = 4 * x
        n_y = 6 * y
        n_z = 12 * z
        len = torch.sqrt(torch.square(n_x) + torch.square(n_y) + torch.square(n_z))
        interface_normal = torch.cat((n_x/len, n_y/len, n_z/len), dim=1)
        # print('interface_normal,shape', interface_normal.shape)
        return interface_normal


    def function_jump_p(self, p): #pass interface points
        #version 1
        jump_p = self.exact_plus(p) - self.exact_minus(p)

        #version2
        # x = p[..., 0:1]
        # y = p[..., 1:2]
        # z = p[..., 2:3]
        # jump_p = (x + y + z) - (x**2 + y**2 + z**2)
        # jump_p = jump_p.reshape(-1, 1)
        return jump_p

    def flux_jump_q(self, p): #pass interface points
        # #version1
        # x = p[..., 0:1]
        # y = p[..., 1:2]
        # z = p[..., 2:3]
        # jump_q = (4 * x - 60) * 4 * x + (4 * y - 60) * 6 * y + (4 * z - 60) * 12 * z
        # jump_q = jump_q.reshape(-1, 1)

        #version2
        unv = self.interface_normal(p)
        m, n = p.size()
        grad_minus = 2 * p
        grad_plus = torch.ones(m, 3)

        # print('grad_minus.shape', grad_minus.shape)
        # print('grad_plus.shape', grad_plus.shape)
        jump_q = (torch.sum(grad_plus*unv, dim=1)*self.bp -
                  torch.sum(grad_minus*unv, dim=1)*self.bm)
        #version 3
        # x = p[..., 0]
        # y = p[..., 1]
        # z = p[..., 2]
        # len = torch.sqrt(torch.square(4 * x) + torch.square(6*y) + torch.square(12 * z))
        #
        # jump_q = (60-4 * x) * 4*x +(60-4 * y)*6*y + (60-4*z) * 12 * z
        # jump_q = jump_q/len
        jump_q = jump_q.reshape(-1, 1)
        return jump_q

    '''Data Generation'''
    def interior_points(self, N_omega=1024):
        dimension = 3
        xr = torch.rand(N_omega, dimension) * 2 - 1 #(N_omega, 2) [0,1]
        return xr

    def boundary_points(self, N_b=32):
        dimension = 3
        xb = torch.rand(2*dimension*N_b, dimension)*2 - 1 #size: (N_b * 4, 2), domain(-1,1) any coordinate value
        #form boudnary
        for i in range(dimension): #i=0-- index of x coordinate; i = 1, index of y coordinate
            xb[2*i*N_b: (2*i+1)*N_b, i] = -1 # x= -1 [0:N_b,0] and y = -1 [2N_b:3N_b,1]
            xb[(2*i+1)*N_b: (2*i+2)*N_b, i] = 1 #x=1 [N_b:2N_b,0] and y = -1 [3N_b:4N_b,1]
        return xb

    def interface_points(self, N_f=512):
        #version1
        a = torch.sqrt(torch.tensor(1.69/2))
        b = torch.sqrt(torch.tensor(1.69/3))
        c = torch.sqrt(torch.tensor(1.69/6))
        theta = 2 * torch.pi*torch.rand(N_f, 1)
        phi = torch.pi*torch.rand(N_f, 1)
        x = a * torch.cos(theta) * torch.sin(phi)
        y = b * torch.sin(theta) * torch.sin(phi)
        z = c * torch.cos(phi)
        xf = torch.cat((x, y, z), dim=1)
        return xf
        #version2
        # xf = []
        # for _ in range(N_f):
        #     x = np.random.uniform(-1, 1)
        #     y = np.random.uniform(-1, 1)
        #     z_squared = (1.69 - 2 * x ** 2 - 3 * y ** 2) / 6
        #     if z_squared >= 0:
        #         z = np.sqrt(z_squared)
        #         xf.append((x, y, z))
        #         xf.append((x, y, -z))  # 添加对称点
        # xf = torch.tensor(xf, dtype=torch.float32)  # 假设xf原来是一个列表
        # return xf