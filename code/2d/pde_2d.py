import torch
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
setup_seed(888888)


class flowerInterface():
    def __init__(self):
        self.beta_plus = torch.tensor(10)
        self.beta_minus = torch.tensor(1)
        self.domain_area = torch.tensor(4.0, dtype=torch.float64)

    def dimension(self):
        d = 2
        return d

    def subdomain_areas(self):
        r0 = torch.tensor(0.5, dtype=torch.float64)
        domain_area_minus = torch.pi*torch.square(r0) + torch.pi/98
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        interface_length = torch.tensor(4.40279704689901, dtype=torch.float64)
        return interface_length

    def boundary_length(self):
        boundary_length = torch.tensor(8.0, dtype=torch.float64)
        return boundary_length
    '''Differentiate value inside and outside the interface'''

    def phi(self, p):
        x = p[...,0]
        y = p[...,1]
        r = torch.sum(torch.square(p), dim=1)
        theta = torch.atan2(y, x)
        is_neg = theta < 0
        theta[is_neg] = theta[is_neg] + 2*torch.pi
        r0 = 0.5 + torch.sin(5*theta)/7
        s = r - torch.square(r0)
        s = s.reshape(-1,1)
        return s > 0 #outside interface


    #meet the condition (in interface), return beta_plus, otherwise return beta_minus
    def diffusion_coefficient_judge(self, p): # #pass domain points
        judge_coefficient = torch.where(self.phi(p), self.beta_plus, self.beta_minus)
        return judge_coefficient

    def diffusion_plus(self, p): #p.shape(m, 2)
        m, n = p.size()
        beta_plus = self.beta_plus * torch.ones(m, 1)
        return beta_plus

    def diffusion_minus(self, p):
        m, n = p.size()
        beta_minus = self.beta_minus * torch.ones(m, 1)
        return beta_minus


    '''exact and rhs'''
    def exact_minus(self, p):
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        solution_minus = torch.exp(torch.square(r))

        # m, n = p.size()
        # solution_minus = torch.ones(m, 1)
        solution_minus = solution_minus.reshape(-1, 1)

        # print('solution_minus', solution_minus)
        return solution_minus

    def exact_plus(self, p):
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        solution_plus = 0.1 * torch.pow(r, 4) - 0.01 * torch.log(2 * r)
        solution_plus = solution_plus.reshape(-1, 1)
        # print('solution_plus', solution_plus)
        return solution_plus

    def exact_judge(self, p):         #pass domain or boundary points
        judge_solution = torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))
        return judge_solution

    def rhs_minus(self, p): #subdomain 1 point
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        rhs1 = -4*(1 + torch.square(r))*torch.exp(torch.square(r))
        return rhs1
    def rhs_plus(self, p):

        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        rhs2 = -16*torch.square(r)
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
        x = p[..., 0]  # x coordinate
        y = p[..., 1]  # y coordinate

        theta = torch.atan2(y, x)
        is_neg = theta < 0
        theta[is_neg] = theta[is_neg] + 2 * torch.pi
        theta = theta.reshape(-1, 1)

        r = 1/2 + torch.sin(5*theta)/7
        dr_dtheta = 5/7*torch.cos(5*theta)
        n_x = dr_dtheta*torch.cos(theta) - r*torch.sin(theta)
        n_y = dr_dtheta*torch.sin(theta) + r*torch.cos(theta)
        len = torch.sqrt(torch.square(n_x) + torch.square(n_y))
        interface_normal = torch.cat((n_y/len, -n_x/len), 1)
        return interface_normal

    def function_jump_p(self, p): #pass interface points
        jump_p = self.exact_plus(p) - self.exact_minus(p)
        return jump_p

    def flux_jump_q(self, p): #pass interface points
        unv = self.interface_normal(p)
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)

        grad_minus = 2*p*torch.exp(torch.square(r))
        grad_plus = 0.4*p*torch.square(r) - 0.01*p*torch.pow(r, -2)
        jump_q = (torch.sum(grad_plus * unv, dim=1) * self.beta_plus -
                  torch.sum(grad_minus * unv, dim=1) * self.beta_minus)
        jump_q = jump_q.reshape(-1, 1)
        return jump_q


    '''Data Generation'''
    def interior_points(self, N_omega=1024):
        # h = 30
        # x_coords = torch.linspace(-1, 1, steps=h)
        # y_coords = torch.linspace(-1, 1, steps=h)
        # X, Y = torch.meshgrid(x_coords, y_coords)
        # grid_points = torch.stack([X, Y], dim=-1)
        # xr = grid_points.view(-1, 2)

        dimension = 2
        xr = torch.rand(N_omega, dimension) * 2 - 1 #(N_omega, 2)
        return xr

    def boundary_points(self, N_b=32):
        dimension = 2
        xb = torch.rand(2*dimension*N_b, dimension) * 2 - 1 #size: (N_b * 4, 2), domain(-1,1) any coordinate value
        #form boudnary
        for i in range(dimension): #i=0-- index of x coordinate; i = 1, index of y coordinate
            xb[2*i*N_b:(2*i+1)*N_b, i] = -1 # x= -1 [0:N_b,0] and y = -1 [2N_b:3N_b,1]
            xb[(2*i+1)*N_b:(2*i+2)*N_b, i] = 1 #x=1 [N_b:2N_b,0] and y = -1 [3N_b:4N_b,1]
        return xb

    def interface_points(self, N_f=256):
        theta = 2*torch.pi*torch.rand(N_f,1)
        r = 1/2 + torch.sin(5*theta)/7
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        xf = torch.cat((x, y), dim=1)
        return xf

class circle_interface():
    def __init__(self, bp, bm, r0=0.5):
        self.beta_plus = torch.tensor(bp, dtype=torch.float64)
        self.beta_minus = torch.tensor(bm, dtype=torch.float64)
        self.domain_area = torch.tensor(4.0, dtype=torch.float64)
        self.r0 = torch.tensor(r0, dtype=torch.float64)

    def dimension(self):
        d = 2
        return d

    def subdomain_areas(self):
        domain_area_minus = torch.pi*torch.square(self.r0)
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        interface_length = 2 * torch.pi * self.r0
        return interface_length

    def boundary_length(self):
        boundary_length = torch.tensor(8.0, dtype=torch.float64)
        return boundary_length
    '''Differentiate value inside and outside the interface'''

    def phi(self, p):
        r = torch.sum(torch.square(p), dim=1)
        s = r - torch.square(self.r0)
        s = s.reshape(-1, 1)
        return s > 0 #outside interface

    #meet the condition (in interface), return beta_plus, otherwise return beta_minus
    def diffusion_coefficient_judge(self, p): # #pass domain points
        judge_coefficient = torch.where(self.phi(p), self.beta_plus, self.beta_minus)
        return judge_coefficient

    def diffusion_plus(self, p): #p.shape(m, 2)
        m, n = p.size()
        beta_plus = self.beta_plus * torch.ones(m, 1)
        return beta_plus

    def diffusion_minus(self, p):
        m, n = p.size()
        beta_minus = self.beta_minus * torch.ones(m, 1)
        return beta_minus

    '''exact and rhs'''
    def exact_minus(self, p): #blue1
        #example 1
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        solution_minus = torch.pow(r, 3)/self.beta_minus
        solution_minus = solution_minus.reshape(-1, 1)
        #example 2
        # m, n = p.size()
        # solution_minus = torch.ones(m, 1)
        return solution_minus

    def exact_plus(self, p): #red2
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        solution_plus = (torch.pow(r, 3)/self.beta_plus +
                          (1/self.beta_minus - 1/self.beta_plus) *
                          torch.pow(self.r0, 3))

        # solution_plus = 1 + torch.log(2*r)
        solution_plus = solution_plus.reshape(-1, 1)

        # m, n = p.size()
        # solution_plus = torch.zeros(m, 1)
        return solution_plus

    def exact_judge(self, p):         #pass domain or boundary points
        judge_solution = torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))
        return judge_solution

    def rhs_plus(self, p): #subdomain 1 point

        #example 1
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        rhs_plus = -9 * r

        #example 2
        # m,n = p.size()
        # rhs_plus = torch.zeros(m, 1)
        return rhs_plus

    def rhs_minus(self, p):
        #example 1
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        rhs_minus = -9 * r
        #example 2
        # m, n = p.size()
        # rhs_minus = torch.zeros(m, 1)
        return rhs_minus

    def rhs_judge(self, p): #pass domain points
        rhs_judge = torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))
        return rhs_judge

    #BC conditions u = g
    def BC_Dirichlet(self, p):
        BC_real = self.exact_judge(p) #Pass in BC_points--judge--return exact_minus2
        # print(BC_real.shape)
        return BC_real

    '''Jump'''
    def interface_normal(self, p): #pass in: point on the interface
        len = torch.sqrt(torch.sum(torch.square(p), dim=1))
        len = len.reshape(-1, 1)
        # interface_normal = p/len
        #examle 2
        interface_normal = p/len
        return interface_normal

    def function_jump_p(self, p): #pass interface points
        m, n = p.size()
        jump_p = torch.zeros(m, 1)
        return jump_p

    def flux_jump_q(self, p): #pass interface points
        #example1
        m, n = p.size()
        jump_q = torch.zeros(m, 1)

        #example 2
        # unv = self.interface_normal(p)
        # r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        # r = r.reshape(-1, 1)
        # grad_plus = p * torch.pow(r, -2)
        # jump_q = torch.sum(grad_plus * unv, dim=1) * self.beta_plus
        # jump_q = jump_q.reshape(-1, 1)
        # print('jump_q.shape', jump_q.shape)
        # print(jump_q)
        # jump_q = 2 * self.diffusion_plus(p)
        # jump_q = jump_q.reshape(-1, 1)
        return jump_q

    '''Data Generation'''
    def interior_points(self, N_omega=1024):
        dimension = 2
        xr = torch.rand(N_omega, dimension) * 2 - 1 #(N_omega, 2)
        return xr

    def boundary_points(self, N_b=64):
        dimension = 2
        xb = torch.rand(2*dimension*N_b, dimension) * 2 - 1 #size: (N_b * 4, 2), domain(-1,1) any coordinate value
        #form boudnary
        for i in range(dimension): #i=0-- index of x coordinate; i = 1, index of y coordinate
            xb[2*i*N_b:(2*i+1)*N_b, i] = -1 # x= -1 [0:N_b,0] and y = -1 [2N_b:3N_b,1]
            xb[(2*i+1)*N_b:(2*i+2)*N_b, i] = 1 #x=1 [N_b:2N_b,0] and y = -1 [3N_b:4N_b,1]
        return xb

    def interface_points(self, N_f=128):
        theta = 2*torch.pi*torch.rand(N_f,1)
        x = self.r0 * torch.cos(theta)
        y = self.r0 * torch.sin(theta)
        xf = torch.cat((x, y), dim=1)
        return xf


class petal_shaped_interface():
    pass


class straight_interface:
    def __init__(self, bp, bm):
        self.beta_plus = torch.tensor(bp)
        self.beta_minus = torch.tensor(bm)
        self.domain_area = torch.tensor(1.0, dtype=torch.float64)

    def dimension(self):
        d = 2
        return d

    def subdomain_areas(self):
        domain_area_minus = torch.tensor(0.5)
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        interface_length = torch.tensor(1.414, dtype=torch.float64)
        return interface_length

    def boundary_length(self):
        boundary_length = torch.tensor(4.0, dtype=torch.float64)
        return boundary_length
    '''Differentiate value inside and outside the interface'''

    def phi(self, p):
        x = p[..., 0]
        y = p[..., 1]
        s = y-x
        s = s.reshape(-1, 1)
        return s > 0 #outside interface

    #meet the condition (in interface), return beta_plus, otherwise return beta_minus
    def diffusion_coefficient_judge(self, p): # #pass domain points
        judge_coefficient = torch.where(self.phi(p), self.beta_plus, self.beta_minus)
        return judge_coefficient

    def diffusion_plus(self, p): #p.shape(m, 2)
        m, n = p.size()
        beta_plus = self.beta_plus * torch.ones(m, 1)
        return beta_plus

    def diffusion_minus(self, p):
        m, n = p.size()
        beta_minus = self.beta_minus * torch.ones(m, 1)
        return beta_minus


    '''exact and rhs'''
    def exact_minus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        solution_minus = x ** 2 + x * y
        solution_minus = solution_minus.reshape(-1, 1)
        return solution_minus

    def exact_plus(self, p):
        x = p[..., 0]
        y = p[..., 1]
        solution_plus = x ** 2 + y ** 2
        solution_plus = solution_plus.reshape(-1, 1)
        return solution_plus

    def exact_judge(self, p):         #pass domain or boundary points
        judge_solution = torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))
        return judge_solution

    def rhs_minus(self, p): #subdomain 1 point
        m, n = p.size()
        rhs1 = -torch.ones(m, 1)
        return rhs1
    def rhs_plus(self, p):
        m, n = p.size()
        rhs2 = -torch.ones(m, 1)
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
        m, n = p.size()
        n_x = torch.ones(m, 1)
        n_y = torch.ones(m, 1)
        len = torch.sqrt(torch.square(n_x) + torch.square(n_y))
        # print(len)
        interface_normal = torch.cat((-n_y/len, n_x/len), 1)
        # print('interface_normal,shape', interface_normal.shape)
        return interface_normal


    def function_jump_p(self, p): #pass interface points
        m, n = p.size()
        jump_p = torch.zeros(m, 1)
        return jump_p

    def flux_jump_q(self, p): #pass interface points
        y = p[..., 1]
        jump_q = y/(2*torch.sqrt(torch.tensor(2)))
        jump_q = jump_q.reshape(-1, 1)
        return jump_q

    '''Data Generation'''
    def interior_points(self, N_omega=1024):
        dimension = 2
        xr = torch.rand(N_omega, dimension) #(N_omega, 2) [0,1]
        return xr

    def boundary_points(self, N_b=32):
        dimension = 2
        xb = torch.rand(2*dimension*N_b, dimension) #size: (N_b * 4, 2), domain(-1,1) any coordinate value
        #form boudnary
        for i in range(dimension): #i=0-- index of x coordinate; i = 1, index of y coordinate
            xb[2*i*N_b:(2*i+1)*N_b, i] = 0 # x= -1 [0:N_b,0] and y = -1 [2N_b:3N_b,1]
            xb[(2*i+1)*N_b:(2*i+2)*N_b, i] = 1 #x=1 [N_b:2N_b,0] and y = -1 [3N_b:4N_b,1]
        return xb

    def interface_points(self, N_f=256):
        x = torch.rand(N_f,1)
        y = x
        xf = torch.cat((x, y), dim=1)
        return xf


class sunflower_interface():

    def __init__(self, bp=10, bm=1, omega=15):
        self.bp = torch.tensor(bp, dtype=torch.float64)
        self.bm = torch.tensor(bm, dtype=torch.float64)
        self.omega = omega
        self.domain_area = torch.tensor(4.0, dtype=torch.float64)
        self.r0 = torch.tensor(0.4, dtype=torch.float64)
        self.r1 = torch.tensor(0.2, dtype=torch.float64)
        self.xc = 0.02 * torch.sqrt(torch.tensor(5))
        self.yc = 0.02 * torch.sqrt(torch.tensor(5))

    def dimension(self):
        d = 2
        return d

    def subdomain_areas(self):
        domain_area_minus = torch.pi*torch.square(self.r0) + torch.pi*torch.square(self.r1)/2
        domain_area_plus = self.domain_area - domain_area_minus
        return domain_area_plus, domain_area_minus

    def interface_length(self):
        len = torch.tensor(16.369554174673745, dtype=torch.float64)
        return len

    def boundary_length(self):
        return torch.tensor(8.0, dtype=torch.float64)

    def phi(self, p):
        x = p[..., 0]
        y = p[..., 1]

        theta = torch.atan2(y - self.yc, x - self.xc)
        is_neg = theta < 0
        theta[is_neg] = theta[is_neg] + 2 * torch.pi

        r_square = torch.square(x-self.xc) + torch.square(y-self.yc)
        r0 = self.r0 + self.r1 * torch.sin(self.omega * theta)
        s = r_square - torch.square(r0)
        s = s.reshape(-1, 1)
        return s > 0


    def interface_normal(self, p):
        x = p[..., 0:1]
        y = p[..., 1:2]
        omega = self.omega
        r0 = torch.tensor(0.4, dtype=torch.float64)
        r1 = torch.tensor(0.2, dtype=torch.float64)
        xc = 0.02 * np.sqrt(5.0)
        yc = 0.02 * np.sqrt(5.0)
        theta = torch.atan2(y - yc, x - xc)
        theta = theta.reshape(-1, 1)
        r = r0 + r1 * torch.sin(omega * theta)
        rp = omega * r1 * torch.cos(omega * theta)
        n1 = rp * torch.cos(theta) - r * torch.sin(theta)
        n2 = rp * torch.sin(theta) + r * torch.cos(theta)
        len = torch.sqrt(torch.square(n1) + torch.square(n2))
        return torch.cat((n2 / len, -n1 / len), dim=1)

    def diffusion_coefficient_judge(self,p):
        return torch.where(self.phi(p), self.bp, self.bm)

    def diffusion_plus(self,p):
        m, n = p.size()
        return self.bp*torch.ones(m, 1)

    def diffusion_minus(self,p):
        m, n = p.size()
        return self.bm*torch.ones(m, 1)

    def exact_plus(self, p):
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        r4 = torch.pow(r, 4)
        plus = (r4 - 0.1 * torch.log(2 * r)) / self.bp
        return plus

    def exact_minus(self, p):
        r2 = torch.sum(torch.square(p), dim=1)
        minus = r2/self.bm
        minus = minus.reshape(-1, 1)
        return minus

    def exact_judge(self, p):
        judge = torch.where(self.phi(p), self.exact_plus(p), self.exact_minus(p))
        return judge

    def rhs_plus(self, p):
        r2 = torch.sum(torch.square(p), dim=1)
        r2 = r2.reshape(-1, 1)
        plus = -16 * r2
        return plus

    def rhs_minus(self, p):
        m, n = p.size()
        minus = -4 * torch.ones(m, 1)
        return minus

    def rhs_judge(self, p):
        judge = torch.where(self.phi(p), self.rhs_plus(p), self.rhs_minus(p))
        return judge

    def BC_Dirichlet(self, p):
        BC_real = self.exact_judge(p)  # Pass in BC_points--judge--return exact_minus2
        return BC_real

    def function_jump_p(self, p):
        jump_p = self.exact_plus(p)-self.exact_minus(p)
        return jump_p

    def flux_jump_q(self, p):
        unv = self.interface_normal(p)
        r = torch.sqrt(torch.sum(torch.square(p), dim=1))
        r = r.reshape(-1, 1)
        grad_minus = (2 * p)/self.bm
        grad_plus = (4 * p * torch.square(r) - 0.1 * p * torch.pow(r, -2))/self.bp
        jump_q = (torch.sum(grad_plus * unv, dim=1) * self.bp -
                  torch.sum(grad_minus * unv, dim=1) * self.bm)
        jump_q = jump_q.reshape(-1, 1)
        return jump_q

    '''Data Generation'''
    def interior_points(self, N_omega=1024):
        dimension = 2
        xr = 2 * torch.rand(N_omega, dimension) - 1 #(N_omega, 2) [0,1]
        return xr

    def boundary_points(self, N_b=32):
        dimension = 2
        xb = 2 * torch.rand(2*dimension*N_b, dimension) - 1 #size: (N_b * 4, 2), domain(-1,1) any coordinate value
        #form boudnary
        for i in range(dimension): #i=0-- index of x coordinate; i = 1, index of y coordinate
            xb[2*i*N_b:(2*i+1)*N_b, i] = -1 # x= -1 [0:N_b,0] and y = -1 [2N_b:3N_b,1]
            xb[(2*i+1)*N_b:(2*i+2)*N_b, i] = 1 #x=1 [N_b:2N_b,0] and y = -1 [3N_b:4N_b,1]
        return xb

    def interface_points(self, N=128):
        theta = 2 * torch.pi * torch.rand(N, 1)
        r = self.r0 + self.r1 * torch.sin(self.omega * theta)
        x = self.xc + r * torch.cos(theta)
        y = self.yc + r * torch.sin(theta)
        p = torch.cat((x, y), dim=1)
        return p

