import time

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端
from matplotlib import rcParams
import pandas as pd
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels

rcParams['text.usetex'] = True
# matplotlib.use('Agg')

from Adaptive_final_2025.Fc_Net import FCNet

from Adaptive_final_2025.indicator import *

'''#jump2-1'''
from Adaptive_final_2025.Loss_fun import *

import numpy as np
#3d
from pde_pop import *
import pyvista as pv

# from visual import visualize_3d_volume
# print('torch.__version__', torch.__version__)
from Adaptive_final_2025.visualize_all import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r0 = torch.tensor(0.6, device=device)


def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
setup_seed(888888)



def l2_error(u, uh):
    num = torch.sqrt(torch.mean(torch.pow(u-uh, 2)))
    den = torch.sqrt(torch.mean(torch.pow(u, 2)))
    error = num / den
    return error



def interface_points(N_f):
    xf = pde.interface_points(N_f)
    xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)
    # print('xf.shape', xf.shape)
    return xf

def interface_points_add(xf, num_new_points, perturb_scale):
    """Generate new points around the existing interface points using vectorized operations."""
    perturbations = torch.randn(len(xf) * num_new_points, 3, device=xf.device) * perturb_scale
    original_points = xf.repeat_interleave(num_new_points, dim=0)
    new_points = original_points[:, :3] + perturbations

    r_new = torch.norm(new_points, dim=1, keepdim=True)  # [N*num_new_points, 1]
    S_new = pde.S(new_points)

    # r0, A, sigma, factor, angles_first, angles_second = initialize_parameters()
    values_new = r_new - r0 - S_new
    mask = torch.abs(values_new).squeeze(1) < 0.1
    new_xf_candidates = new_points[mask]

    distances = torch.cdist(xf[:, :3], new_xf_candidates).float()
    overlap_mask = torch.any(distances < 1e-3, dim=0)
    final_new_xf = new_xf_candidates[~overlap_mask]
    final_new_xf = torch.cat((final_new_xf, torch.zeros(final_new_xf.size(0), 1, device=xf.device)), dim=-1)
    # print('final_new_xf.shape', final_new_xf.shape)
    return final_new_xf

def generate_grid_points(h):#default = 5
    x_coords = torch.linspace(-1, 1, steps=h)
    y_coords = torch.linspace(-1, 1, steps=h)
    z_coords = torch.linspace(-1, 1, steps=h)
    X, Y, Z = torch.meshgrid(x_coords, y_coords,z_coords,  indexing = 'xy')
    grid_points_3d = torch.stack([X, Y, Z], dim=-1)
    grid_points_3d = grid_points_3d.view(-1, 3)
    grid_points_4d = torch.cat((grid_points_3d, torch.zeros(grid_points_3d.size(0), 1)), dim=-1)
    return grid_points_4d


def add_grid(points, refinement_factor, h):
    count = points[:, -1]
    initial_spacing = (1 / (h - 1)) / (2 ** count)
    # Create offset grid
    offsets = torch.linspace(-1, 1, refinement_factor, device=points.device)
    dxs, dys, dzs = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    offset_grid = torch.stack([dxs, dys, dzs], dim=-1).reshape(-1, 3)

    # Expand initial_spacing to match each offset, then calculate offsets for each point
    expanded_offsets = (offset_grid[None, :, :] * initial_spacing[:, None, None]).reshape(-1, 3)

    # Expand points and add calculated offset
    expanded_points = points[:, :3].repeat_interleave(len(offset_grid), dim=0)
    refined_coords = expanded_points + expanded_offsets

    # Increment count for each new refined point
    refined_count = (count + 1).repeat_interleave(len(offset_grid))

    # Combine refined coordinates and counts
    refined_points = torch.cat((refined_coords, refined_count.unsqueeze(-1)), dim=-1)

    # Filter points within bounds
    refined_points = refined_points[(refined_points[:, :3].abs() <= 1).all(dim=1)]

    # Remove points overlapping with original points
    distances = torch.cdist(refined_points[:, :3], points[:, :3])
    mask = (distances.min(dim=1).values > 1e-6)  # Remove new points that are very close to existing points
    refined_points = refined_points[mask]
    refined_points = torch.unique(refined_points, dim=0).detach().requires_grad_(True)
    return refined_points


def initial_data(h, Nf):
    grid_points = generate_grid_points(h)
    boundary_mask = (torch.abs(grid_points[:,:3]).max(dim=1)[0] == 1.0).bool() #only max tensor not indices
    xb = grid_points[boundary_mask]
    xr = grid_points[~boundary_mask]
    xf = interface_points(Nf)

    xr = xr.clone().detach().requires_grad_(True)
    xf = xf.clone().detach().requires_grad_(True)
    xb = xb.clone().detach().requires_grad_(True)
    return xr, xf, xb



def classify_x_all(x_all):
    r = torch.norm(x_all[:,:3], dim=1, keepdim=True)
    S = pde.S(x_all[:,:3])
    interface_mask = torch.abs(r-r0-S) < 0.1
    interface_mask = interface_mask.squeeze(1)

    boundary_threshold = 1.0
    boundary_mask = (torch.abs(x_all[:,:3]).max(dim=1)[0] == boundary_threshold).bool() #only max tensor not indices
    combined_mask = interface_mask | boundary_mask
    interior_mask = ~combined_mask
    xb, xf, xr = x_all[boundary_mask], x_all[interface_mask], x_all[interior_mask]
    xr, xf, xb = map(lambda x: x.clone().detach().requires_grad_(True), (xr, xf, xb))
    return xr, xf, xb

def train_only1(epochs, gamma_b, gamma_f, xr, xf, xb, net_plus, net_minus, opt1, opt2):
    loss_list = []
    x_t = pde.interior_points(10000)
    # u_exact = pde.exact_judge(x_t)

    for i in range(epochs):
        # epoch_time = time.time()
        loss_energy, loss_interior, loss_interface, loss_bc =\
        loss_nitsche(net_plus, net_minus, xr, xf, xb, pde, gamma_b, gamma_f)
        # epoch_time = time.time() - epoch_time
        # print(f'epoch_time = {epoch_time}')
        '''train pde network, record loss'''
        opt1.zero_grad()
        opt2.zero_grad()
        loss_energy.backward()
        opt1.step()
        opt2.step()
        if i % 500 == 0:
            loss_list.append(loss_energy.item())
            # u_pred = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
            # relative_l2 = l2_error(u_exact, u_pred)
            # l2_list.append(relative_l2.item())
            # print(f'At iteration = {i}, pde_loss = {loss_energy}, l2_error={relative_l2}, interior_loss = {loss_interior},'
            #       f'bc loss = {loss_bc}, interface loss = {loss_interface}')

    num_test = 1200
    with torch.no_grad():
        x_t = pde.interior_points(num_test).to(device)
        u = pde.exact_judge(x_t).to(device)
        uh = torch.where(pde.phi(x_t), net_plus(x_t), net_minus(x_t))
        err_l2 = l2_error(u, uh)
    print(f"at g_b = {gamma_b}, g_f = {gamma_f}, Relative l2 error of u_pred (average over {num_test} points): {err_l2}")
    return net_plus, net_minus, err_l2


def train_adapt_grid(h, Nf, re_level, re_factor, num_insert, perturb_scale, epochs,
                      gamma_b, gamma_f, m, depth, d, lr):

    net_plus = FCNet(d, m, 1, depth=depth)
    net_minus = FCNet(d, m, 1, depth=depth)


    opt1 = torch.optim.Adam(net_plus.parameters(), lr=lr)
    opt2 = torch.optim.Adam(net_minus.parameters(), lr=lr)

    xr, xf, xb = initial_data(h, Nf)
    print('Initial Data (xr, xf, xb)', xr.shape, xf.shape, xb.shape)
    df, i, err_l2 = [], 0, None

    while i < re_level:
        i +=1
        old_point = torch.cat((xr, xf, xb), dim=0)
        start_time_train = time.time()
        net_plus, net_minus, err_l2 = train_only1(epochs, gamma_b, gamma_f, xr[:,:3], xf[:,:3], xb[:,:3], net_plus, net_minus, opt1, opt2)
        end_time_train = time.time()
        time_train = end_time_train - start_time_train
        # print(f'train time of i={i} is {time_train}')

        '''add and store more sampling points'''
        # addgrid_start = time.time()
        grid_add = add_grid(torch.cat((xr, xb), dim=0), re_factor, h)
        # add_grid_time = time.time() - addgrid_start
        # print(f'add grid time is {add_grid_time}')

        xr_add, _, xb_add = classify_x_all(grid_add)

        # time_addxf = time.time()
        xf_add = interface_points_add(xf[:, :3], num_insert, perturb_scale)
        # interface_points_add_time = time.time() - time_addxf
        # print('interface_points_add_time', interface_points_add_time)

        # v3-rank grad
        start_time_grad = time.time()
        res = indicator_grad(net_plus, net_minus, xr_add[:,:3], xf_add[:,:3], xb_add[:,:3], pde)#v1
        res_list = res.tolist()
        num_top10 = int(0.10 * len(res_list))
        sorted_indices = sorted(range(len(res_list)), key=lambda k: res_list[k], reverse=True)
        x_all_add = torch.cat((xr_add, xf_add, xb_add), dim=0)#N*3
        x_all_sorted_points = x_all_add[sorted_indices]
        x_top_ten = x_all_sorted_points[:num_top10]
        end_time_grad = time.time()
        time_grad = end_time_grad - start_time_grad
        refine_time = time_train + time_grad

        x_top_ten, _ = torch.unique(x_top_ten, dim=0, return_inverse=True)
        xr_top, xf_top, xb_top = classify_x_all(x_top_ten)#N*3 with the 3rd:+1
        xr = torch.cat((xr, xr_top), dim=0)
        xf = torch.cat((xf, xf_top), dim=0)
        xb = torch.cat((xb, xb_top), dim=0)
        add_point = x_top_ten


        # print(f'xr_top.shape={xr_top.shape}, xf_top.shape={xf_top.shape}, xb_top.shape={xb_top.shape}')
        print(f'(ADD already) xr.shape={xr.shape}, xf.shape={xf.shape}, xb.shape={xb.shape}')
        result = {'Refine Level': i, 'l2 error': err_l2.item(), 'Num of (xr,xf,xb)': [xr.size(0),xf.size(0), xb.size(0)],'Training time':refine_time}
        df.append(result)

    #     #save results
    # #     with open(f'samp_level_{i}.pkl', 'wb') as f:
    # #         import pickle
    # #         # print(f'old_point.type = {old_point.type}')
    # #         pickle.dump({'old': old_point.cpu().detach().numpy(), 'add': add_point.cpu().detach().numpy()}, f)
    # # df = pd.DataFrame(df)
    # # h_draw = 600
    # # print(f'h_draw = {h_draw}')
    # # box = torch.tensor([-1, 1, -1, 1, -1, 1], dtype=torch.float64, device=device)  # compute tensor on gpu
    # # x1 = torch.linspace(box[0], box[1], h_draw, device=device)
    # # x2 = torch.linspace(box[2], box[3], h_draw, device=device)
    # # x3 = torch.linspace(box[4], box[5], h_draw, device=device)
    # # X, Y, Z = torch.meshgrid(x1, x2, x3)
    # # points = torch.cat((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]), dim=1).to(device)
    # # net_plus = net_plus(points).to(device)
    # # mask = points[:, 0] < 0
    # # net_plus[mask] = 0
    # # net_clipped = torch.where(pde.phi(points).to(device), net_plus, net_minus(points).to(device))
    # # net_clipped = net_clipped.view(h_draw, h_draw, h_draw).cpu().detach().numpy()
    # # print('net_clipped.shape', net_clipped.shape)
    # # # real_clipped = real_clipped.cpu().detach().numpy()
    # # X_np = X.cpu().numpy()
    # # Y_np = Y.cpu().numpy()
    # # Z_np = Z.cpu().numpy()
    # # np.savez("net_clipped_data.npz", net_clipped=net_clipped, X=X_np, Y=Y_np, Z=Z_np)



        # figure_sampling
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(old_point[:, 0].detach().numpy(), old_point[:, 1].detach().numpy(), old_point[:, 2].detach().numpy(), c='r', label='Old', alpha=0.3)
        # ax.scatter(add_point[:, 0].detach().numpy(), add_point[:, 1].detach().numpy(), add_point[:, 2].detach().numpy(), c='b', label='Add')
        # # ax.scatter(xf_top[:, 0].detach().numpy(), xf_top[:, 1].detach().numpy(), xf_top[:, 2].detach().numpy(), c='g', label='add f')
        #
        # import matplotlib.ticker as ticker
        # ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        # ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
        #
        # ax.view_init(elev=23, azim=-59)  # elev控制z轴旋转，azim控制x-y平面旋转
        #
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title(f"Level {i}")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'pop_{i}.png', dpi=1200)
        # plt.show()
        # plt.close()

    df = pd.DataFrame(df)

    '''3d solution'''
    # solution_3d('real',350,300,'real_pop.png', pde,net_plus, net_minus)
    # solution_3d('pred',350,300,'pred_pop.png', pde,net_plus, net_minus)

    return net_plus, net_minus, df, err_l2


'''pop'''

pde = popcorn_interface()
# epochs, re_level = 1, 1
# epochs, re_level = 1000, 6


#original
# h, Nf = 9, 10000
#
# re_factor, num_insert, perturb_scale = 2, 10, 0.05
#
# gamma_b, gamma_f = 9000, 500
#
# m,depth = 32, 3
# d,lr = 3, 1e-3


#new
epochs, re_level = 600, 9
# epochs, re_level = 1, 1

h, Nf = 5, 1000

# re_factor, num_insert, perturb_scale = 2, 5, 0.09
re_factor, num_insert, perturb_scale = 2, 5, 0.09

gamma_b, gamma_f = 5000, 300

m,depth = 32, 3
d,lr = 3, 1e-3


net_plus, net_minus, df, err_l2 = train_adapt_grid(h, Nf, re_level, re_factor,num_insert,perturb_scale, epochs,
                      gamma_b, gamma_f, m, depth, d, lr)

print(df)
df.to_excel(f'pop_adapt_random.xlsx', index = False)


'''abs_error'''

def abs_error(h):
    box = torch.tensor([-1, 1, -1, 1, -1, 1], dtype=torch.float64)  # ellipsoidal

    x1 = torch.linspace(box[0], box[1], h)
    x2 = torch.linspace(box[2], box[3], h)
    x3 = torch.linspace(box[4], box[5], h)
    X, Y, Z = torch.meshgrid(x1, x2, x3, indexing='ij')

    points = torch.cat((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]), dim=1)

    real = pde.exact_judge(points)
    real = real.detach().numpy()
    real = real.reshape(h, h, h)

    pred = torch.where(pde.phi(points), net_plus(points), net_minus(points))
    pred = pred.detach().numpy()
    pred = pred.reshape(h, h, h)

    abs_error_adapt = np.abs(real - pred)
    abs_error_adapt = abs_error_adapt.reshape(h, h, h)
    from DUNM.Module.visualize_2d_surface import abs_error_nitsche_3d

    slice_idx = h // 3
    error_adapt_slice = abs_error_adapt[:, :, slice_idx]
    error_random_slice = abs_error_nitsche_3d[:, :, slice_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    heatmap1 = ax1.imshow(error_random_slice, cmap='jet',extent=[box[0], box[1], box[2], box[3]], origin='lower')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.set_title('Random Error')

    # 绘制第一个热力图
    heatmap2 = ax2.imshow(error_adapt_slice, cmap='jet',extent=[box[0], box[1], box[2], box[3]], origin='lower')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    # ax2.set_title('Adaptive Error')

    min_val = min(error_random_slice.min(), error_adapt_slice.min())
    max_val = max(error_random_slice.max(), error_adapt_slice.max())

    cbar = fig.colorbar(heatmap1, ax=[ax1, ax2])
    heatmap2.set_cmap(heatmap1.get_cmap())
    heatmap2.set_clim(heatmap1.get_clim())

    num_ticks = 10
    tick_values = np.linspace(min_val, max_val, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{tick:.3f}' for tick in tick_values * 100])
    cbar.ax.text(0.1, 1.06, r'$\times 10^{-2}$', transform=cbar.ax.transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='left')
    plt.show()
    # plt.savefig('pop_error2.png', dpi=600)

abs_error(300)







# df.to_excel(f'test_results/pop_(h={h},Nf={Nf}),re_factor=2, num_insert=10, '
#             f'perturb_scale=0.05, (b={gamma_b}, f={gamma_f}).xlsx')
# df.to_excel(f'11.24.xlsx')

# Define ranges for gamma_b and gamma_f
# gamma_b_values = [1000, 3000, 8000, 9000, 10000]  # Adjust the values as needed
# gamma_f_values = [100, 300, 500, 700]     # Adjust the values as needed
# gamma_b_values = [9000]  # Adjust the values as needed
# gamma_f_values = [500]     # Adjust the values as needed
# # Define a helper function for training
# def train_and_record(gamma_b, gamma_f):
#     # Call the train_adapt_grid function
#     net_plus, net_minus, df, err_l2 = train_adapt_grid(
#         h, Nf, re_level, re_factor, num_insert, perturb_scale, epochs,
#         gamma_b, gamma_f, m, depth, d, lr
#     )
#     # Return the results
#     return {"gamma_b": gamma_b, "gamma_f": gamma_f, "err_l2": err_l2}

# Create all combinations of gamma_b and gamma_f
# combinations = list(product(gamma_b_values, gamma_f_values))
#
# # Use ThreadPoolExecutor for parallel processing
# results = []
# with ThreadPoolExecutor() as executor:
#     futures = [executor.submit(train_and_record, gamma_b, gamma_f) for gamma_b, gamma_f in combinations]
#     for future in futures:
#         results.append(future.result())
#
# # Convert the results into a DataFrame for easy visualization and analysis
# results_df = pd.DataFrame(results)
#
# print('results_df', results_df)
#
# min_index = results_df.index(min(results_df['err_l2']))
# best_gamma_b = combinations[min_index][0]
# best_gamma_f = combinations[min_index][1]
# # Find the row with the minimum err_l2
# best_combination = results_df.loc[results_df['err_l2'].idxmin()]
# print(f'(gamma_b, gamma_f) = ({best_gamma_b},{best_gamma_f}) with min l2 error = {l2_penalty_list[min_index]} ')

# Display the best combination







