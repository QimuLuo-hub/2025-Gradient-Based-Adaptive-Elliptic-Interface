import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # 使用 Qt5Agg 后端
import matplotlib.pyplot as plt
from DUNM.Module.Res_Net import ResNet
from DUNM.Module.Res_Net import FCNet
'''#jump1-2'''
# from my_loss import *
# from Interface_2d import *

'''#jump2-1'''
from DUNM.Module.Loss_fun_work_v1 import *

#2d
from DUNM.Module.jump2minus1_2d import *

#3d
# from interface_3d import *
from DUNM.Module.interface_3d_v2 import *

#nd
# from interface_nd import *
# from interface_nd_v2 import *

from Adaptive_final_2025.visualize_all import abs_error_2d


def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
setup_seed(888888)

def weights_init_normal(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight, 0, 0.5)
        torch.nn.init.constant_(m.bias, 0.0)

def l2_error(u, uh):
    num = torch.sqrt(torch.mean(torch.pow(u-uh, 2)))
    den = torch.sqrt(torch.mean(torch.pow(u, 2)))
    error = num / den
    return error

batchstep = 10

def train(epochs, gamma_b, gamma_f, Nr, Nb, Nf, m, depth, d):
    loss_list = []
    l2_list = []
    # net_plus = ResNet(d, m, 1, depth=depth)
    # net_minus = ResNet(d, m, 1, depth=depth)
    net_plus = FCNet(d, m, 1, depth=depth)
    net_minus = FCNet(d, m, 1, depth=depth)

    # net_plus.apply(weights_init_normal)
    # net_minus.apply(weights_init_normal)

    opt1 = torch.optim.Adam(net_plus.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(net_minus.parameters(), lr=1e-3)

    # generate test points
    x_t = pde.interior_points(10000)
    u_exact = pde.exact_judge(x_t)

    #generate train points
    xr = pde.interior_points(Nr)
    xf = pde.interface_points(Nf)
    xb = pde.boundary_points(Nb)

    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()
    print('xf.shape', xf.shape)
    for i in range(epochs):
        # if epochs % batchstep == 0:
        #     xr = pde.interior_points(Nr)
        #     xr.requires_grad_()
        #     xf = pde.interface_points(Nf)
        #     xf.requires_grad_()
        #     xb = pde.boundary_points(Nb)
        #     xb.requires_grad_()

        loss_energy, loss_interior, loss_interface, loss_bc =\
        loss_nitsche(net_plus, net_minus, xr, xf, xb, pde, gamma_b, gamma_f)

        # loss_energy, loss_interior, loss_interface, loss_bc =\
        # loss_least_square(net_plus, net_minus, xr, xf, xb, pde, gamma_b, gamma_f)

        '''train pde network, record loss'''
        opt1.zero_grad()
        opt2.zero_grad()
        loss_energy.backward()
        opt1.step()
        opt2.step()

        if i % 100 == 0:
            loss_list.append(loss_energy.item())
            u_pred = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
            relative_l2 = l2_error(u_exact, u_pred)
            l2_list.append(relative_l2.item())
            print(f'At iteration = {i}, pde_loss = {loss_energy}, l2_error={relative_l2}, interior_loss = {loss_interior},'
                  f'bc loss = {loss_bc}, interface loss = {loss_interface}')
    min_l2error = min(l2_list)
    print('min_l2error', min_l2error)

    #PDE Loss
    plt.figure(figsize=(8, 6))
    plt.plot(loss_list, label='PDE Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (every 100 epochs)')
    # plt.title(f'{epochs} epochs, (bp, bm) = ({bp}, {bm}), {d}d: \n (gamma_b, f_parameter)=({gamma_b}, {gamma_f}), '
    #           f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}), (m, depth)=({m}, {depth})')
    plt.legend()

    # if epochs > 5:
        # pde_path = (f'{save_path_ellipsoidal}pdeloss{d}d{epochs}iterations, (bp, bm) = ({bp}, {bm}), (gamma_b, gamma_f)=({gamma_b}, {gamma_f}),'
        #           f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}), (m, depth)=({m}, {depth}).pdf')
        # plt.savefig(pde_path)

    # L2
    plt.figure(figsize=(8, 6))
    plt.plot(l2_list, label='relative l2 error')
    plt.xlabel('Iterations')
    plt.ylabel('Relative l2 error (every 100 epochs)')
    plt.yscale("log")
    plt.title(f'{epochs}epochs, (bp, bm) = ({bp}, {bm}), {d}d: \n (gamma_b, f_parameter)=({gamma_b}, {gamma_f}),'
              f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}), (m, depth)=({m}, {depth})')
    plt.legend()

    # if epochs > 5:
    #     l2_path = (f'{save_path_ellipsoidal}l2error{d}d{epochs}iterations, (bp, bm) = ({bp}, {bm}), (gamma_b, gamma_f)=({gamma_b}, {gamma_f}), '
    #               f'(Nr, Nb, Nf) = ({Nr}, {Nb}, {Nf}),(m, depth)=({m}, {depth}).pdf')
    #     plt.savefig(l2_path)
    # plt.show()

    #Final l2 error
    num_test = 1200
    with torch.no_grad():
       x_t = pde.interior_points(num_test)
       u = pde.exact_judge(x_t)
       uh = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
       uh = uh.reshape(-1, 1)
       err_l2 = l2_error(u, uh)
    print(f'Relative l2 error of u_pred (average on {num_test} points)', err_l2)
    return net_plus, net_minus, min_l2error, err_l2

def abs_error(h, pde, net_plus, net_minus):
    box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
    box = box.detach().numpy()
    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    X, Y = torch.meshgrid(x1, x2, indexing='ij')
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)

    real = pde.exact_judge(Z).reshape(h, h)
    pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(h, h)
    abs_error = torch.abs(real - pred)
    return abs_error

'''flower'''
# save_path_flower = 'Save_results/flower/'
# bp=10
# bm=1
# pde = flowerInterface()
#
# start_time = time.time()
# net_plus, net_minus, min_l2error, err_l2 = train(4500,
#                                          3000, 100,
#                                          1024, 64, 128,
#                                          64, 4, 2)
#
# abs_nitsche_flower = abs_error(300, pde, net_plus, net_minus)
#
# run_time = time.time()-start_time
# print(f'Run time fo flower is {run_time}')





'''sunflower'''
# bp = 10
# bm = 1
# omega = 15
# pde = sunflower_interface(bp, bm, omega)
# start_time = time.time()
# net_plus, net_minus, min_l2error, err_l2 = train(9000,
#                                          8000, 250,
#                                          1024, 64, 256,
#                                          64, 5, 2)

# box = torch.tensor([-1, 1, -1, 1], dtype=torch.float64)
# box = box.detach().numpy()
# x1 = torch.linspace(box[0], box[1], 300)
# x2 = torch.linspace(box[2], box[3], 300)
# X, Y = torch.meshgrid(x1, x2, indexing='ij')
# Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
#
# real = pde.exact_judge(Z).reshape(300, 300)
# pred = torch.where(pde.phi(Z), net_plus(Z), net_minus(Z)).reshape(300, 300)
# abs_nitsche_sun = torch.abs(real - pred)
#
# run_time = time.time()-start_time
# print(f'Run time of sunflower is {run_time}')



'''ellipsoidal'''
# bp = 60
# bm = 2
# pde = ellipsoidal_interface()

# start_time = time.time()
# net_plus, net_minus, min_l2error, err_l2 = train(6000, 9000, 300,2048,43,128,
#                                                  32, 3, 3)
#
# end_time = time.time()
# run_time = end_time - start_time
# print('run_time', run_time)
# save_path_ellipsoidal = 'Save_results/3d/ellipsoidal/'

'''popcorn'''
bp = 60
bm = 2
pde = popcorn_interface()
# save_path_popcorn = 'Save_results/3d/popcorn/'
start_time = time.time()
#original
# net_plus, net_minus, min_l2error, err_l2 = train(6000, 9000, 500,3035,65,79580,
#                                                  32, 3, 3)#total Nb = 6 * Nb, real Nf = 0.0674 * Nf
net_plus, net_minus, min_l2error, err_l2 = train(5400, 5000, 300,2048,43,1288,
                                                 32, 3, 3)#total Nb = 6 * Nb,

# net_plus, net_minus, min_l2error, err_l2 = train(5400, 5000, 300,2048,43,2450,
#                                                  32, 3, 3)#total Nb = 6 * Nb, 0.0509

end_time = time.time()
run_time = end_time - start_time
print('run_time', run_time)











'''nd'''
# 3d
# bp = 10 #beta2
# bm = 1  #beta1
# r0 = 0.4
# d = 3
# # #
# pde = sphere_nd(bp, bm, r0, d)


#nd
# save_path_ndsphere = 'Save_results/high_d/'
# # d_list = [3, 5, 10, 15, 20]
# #3d:
# #5d: (3000, 250), (2048, 256, 512), (10, 3)  work 2.03%
# #15d
# #20d
# d_list = [3, 5]
# df_nd = []
#
# epochs = 50000
#
# for i in range(len(d_list)):
#     pde = sphere_nd(bp, bm, r0, d_list[i])
#     start_time = time.time()
#     ResNet_omega1, ResNet_omega2, min_err_l2 = train(epochs,
#                                          3000, 250,
#                                          1024, 256, 256,
#                                          20, 3, d_list[i])
#     end_time = time.time()
#     train_time = end_time - start_time
#     results_nd = {'epochs': epochs, 'd': d_list[i], 'time': train_time, 'min l2_error': min_err_l2}
#     df_nd.append(results_nd)
#
# df_nd = pd.DataFrame(df_nd)
# df_nd.to_excel(f'nd_records_epochs{epochs}.xlsx', index=False)
# print(df_nd)


'''2d'''
'''straight change BOX!!'''
# bp = 0.25
# bm = 0.5
# pde = straight_interface(bp, bm)
# save_path_straight = 'Save_results/straight/'
#
# net_plus, net_minus, min_l2error, err_l2 = train(10000,
#                                          3000, 500,
#                                          1024, 64, 128,
#                                          10, 3, 2)
'''Circle'''
# beta_list = [1, 1000, 100000]
# bm = 1000 #beta1
# bp = 1 #beta2
#
# r0 = 0.5
# save_path_circle = f'Save_results/circle/bp={bp}, bm={bm}/'
# pde = circle_interface(bp, bm, r0)
# ResNet_omega1, ResNet_omega2, err_l2 = train(200000,
#                                          1000, 700,
#                                          1024, 64, 128,
#                                          10, 3, 2)





'''#3D --------------------------'''
'''heart'''
# pde = heart_3d()
# bp = 8
# bm = 1
# save_path_heart = 'Save_results/3d/heart/'
# ResNet_omega1, ResNet_omega2, err_l2 = train(1,
#                                          3000, 500,
#                                          5000, 64, 6000,
#                                          10, 3, 3)
















