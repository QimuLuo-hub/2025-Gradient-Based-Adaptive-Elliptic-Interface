import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels
rcParams['text.usetex'] = True
import matplotlib
matplotlib.use('Qt5Agg')  # 使用R Qt5Agg 后端

def setup_seed(seed):
    torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
setup_seed(888888)

def generate_grid_points(h):#default = 5
    x_coords = torch.linspace(-1, 1, steps=h)
    y_coords = torch.linspace(-1, 1, steps=h)
    z_coords = torch.linspace(-1, 1, steps=h)
    X, Y, Z = torch.meshgrid(x_coords, y_coords,z_coords,  indexing = 'xy')
    grid_points_3d = torch.stack([X, Y, Z], dim=-1)
    grid_points_3d = grid_points_3d.view(-1, 3)

    counts = torch.randint(0, 4, (grid_points_3d.size(0), 1))#generate count value 0,1,2,3
    grid_points_4d = torch.cat((grid_points_3d, counts.float()), dim=-1)
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
    # refined_count = (count + 1).repeat_interleave(len(offset_grid))#original code
    refined_count = (count ).repeat_interleave(len(offset_grid))#for clear visualization

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

def add_grid_v1(points, refinement_factor, h):
    refined_points = []
    for point in points:
        count = point[-1].item()
        initial_spacing = (1 / (h - 1)) / (2 ** count)
        offsets = torch.linspace(-initial_spacing, initial_spacing, refinement_factor)
        offsets = offsets.clone().detach()
        dxs, dys, dzs = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
        offset_grid = torch.stack([dxs.ravel(), dys.ravel(), dzs.ravel()], dim=-1)
        for offset in offset_grid:
            new_point = point[:4] + torch.cat((offset, torch.tensor([0], dtype=points.dtype)))
            new_point[-1] = count + 1
            if not any(torch.allclose(new_point[:3], p[:3]) for p in points):
                refined_points.append(new_point)
    # refined_points = [p for p in refined_points if all(abs(x) <= 1 for x in p[:3])]
    refined_points = [p for p in refined_points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1 and -1 <= p[2] <= 1]
    refined_points = torch.stack(refined_points)
    refined_points = torch.unique(refined_points, dim=0)
    refined_points = refined_points.clone().detach().requires_grad_(True)
    return refined_points


# global（count=0）
def plot_global_points(old_point, add_point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(old_point[:, 0].detach().numpy(), old_point[:, 1].detach().numpy(), old_point[:, 2].detach().numpy(), c='r', label='Original', s=50)
    ax.scatter(add_point[:, 0].detach().numpy(), add_point[:, 1].detach().numpy(), add_point[:, 2].detach().numpy(), c='b', label='Refined', s=20, alpha=0.5)

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.view_init(elev=11, azim=-49)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sampling_compare/3d_global.png', dpi=300)
    plt.show()


# different counts
def plot_count_points(original_points, refined_points):
    original_x = original_points[:, 0].numpy()
    original_y = original_points[:, 1].numpy()
    original_z = original_points[:, 2].numpy()

    original_counts = original_points[:, 3].numpy()

    refined_x = refined_points.detach().numpy()[:, 0]
    refined_y = refined_points.detach().numpy()[:, 1]
    refined_z = refined_points.detach().numpy()[:, 2]

    refined_counts = refined_points.detach().numpy()[:, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # color map
    colors = ['blue', 'green', 'orange', 'red']
    labels = [r'$t$ = 0', r'$t$ = 1', r'$t$ = 2', r'$t$ = 3']


    # original
    for count in range(4):
        mask = original_counts == count
        ax.scatter(original_x[mask], original_y[mask],original_z[mask], color=colors[count], label=labels[count], s=50)

    # new points
    for count in range(4):
        mask = refined_counts == count
        ax.scatter(refined_x[mask], refined_y[mask], refined_z[mask], color=colors[count], s=20, alpha=0.5)

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.view_init(elev=11, azim=-49)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sampling_compare/3d_count.png', dpi=300)

    plt.show()

def generate_random_points(num_points):
    # 在 [-1, 1]^2 范围内随机生成点
    random_points = torch.rand(num_points, 3) * 2 - 1  # 范围 [-1, 1]

    counts = torch.randint(0, 4, (random_points.size(0), 1))#generate count value 0,1,2,3
    # random_points = torch.cat((random_points, torch.zeros(num_points, 1)), dim=-1)
    random_points = torch.cat((random_points,counts), dim=-1)

    return random_points

def plot_random_points(random_points):
    random_x = random_points[:, 0].numpy()
    random_y = random_points[:, 1].numpy()
    random_z = random_points[:, 2].numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(random_x, random_y, random_z, color='blue', label='Random', s=20)

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.view_init(elev=11, azim=-49)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig('sampling_compare/3d_random.png', dpi=300)
    plt.show()

# # main
# h = 3
# re_factor = 3
# grid_points = generate_grid_points(h)
# random_points = generate_random_points(100)
#
#
# refined_points_grid = add_grid(grid_points, re_factor, h)
# refined_points_random = add_grid(random_points, re_factor, h)
#
#
# plot_random_points(random_points)
#
#
# # global（count=0）
# global_grid_points = grid_points.clone()
# global_grid_points[:, 3] = 0
# global_refined_points = add_grid(global_grid_points, re_factor, h)
# plot_global_points(global_grid_points, global_refined_points)
#
# #count
# plot_count_points(grid_points, refined_points_grid)
# # plot_count_points(grid_points, refined_points_random)


'''ellip:interface'''

def interface_points_ellip(N_f=32):
    a = torch.sqrt(torch.tensor(1.69 / 2))
    b = torch.sqrt(torch.tensor(1.69 / 3))
    c = torch.sqrt(torch.tensor(1.69 / 6))

    # indices = torch.arange(0, N_f+1, dtype=torch.float32)#points from [0,N_f]
    indices = torch.arange(0, N_f, dtype=torch.float32) + 0.5 #do not include the extreme points
    phi = torch.acos(2 * indices / N_f-1) #phi-z axis，map the indicies into 1 - 2 * indices / N_f[-1,1]; use z=rcos(phi)，unit r=1
    theta = torch.pi * (1 + 5 ** 0.5) * indices #otherwise: equatorial dense, polar sparse

    x = a * torch.cos(theta) * torch.sin(phi)
    y = b * torch.sin(theta) * torch.sin(phi)
    z = c * torch.cos(phi)
    xf = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
    xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)
    return xf

def interface_points_add_ellip(xf, n_insert):
    N_f = xf.shape[0]
    new_xf = []

    a = torch.sqrt(torch.tensor(1.69 / 2))
    b = torch.sqrt(torch.tensor(1.69 / 3))
    c = torch.sqrt(torch.tensor(1.69 / 6))

    for i in range(N_f):
        # i_coords (\theta, \phi)
        theta_start = torch.atan2(xf[i, 1], xf[i, 0]).item()  # the same as 2d
        r_i = torch.sqrt((xf[i, 0] / a) ** 2 + (xf[i, 1] / b) ** 2 + (xf[i, 2] / c) ** 2)
        phi_start = torch.acos(xf[i, 2] / (r_i * c)).item()  #z =r*cos(\phi)

        # (i+1)_coords (\theta_{i+1}, \phi_{i+1})
        theta_end = torch.atan2(xf[(i + 1) % N_f, 1], xf[(i + 1) % N_f, 0]).item()
        r_next = torch.sqrt((xf[(i + 1) % N_f, 0] / a) ** 2 + (xf[(i + 1) % N_f, 1] / b) ** 2 + (xf[(i + 1) % N_f, 2] / c) ** 2)
        phi_end = torch.acos(xf[(i + 1) % N_f, 2] / (r_next * c)).item()

        if theta_end < theta_start:
            theta_end += 2 * torch.pi
        # insert points between (\theta, \phi)
        theta_insert = torch.linspace(theta_start, theta_end, n_insert + 2, dtype=torch.float32)[:-1]#2 for start and end; [:-1] for drop theta_end
        phi_insert = torch.linspace(phi_start, phi_end, n_insert + 2, dtype=torch.float32)[:-1]

        #make sure the inserted points lie on the interface
        x_insert = a * torch.cos(theta_insert) * torch.sin(phi_insert)
        y_insert = b * torch.sin(theta_insert) * torch.sin(phi_insert)
        z_insert = c * torch.cos(phi_insert)

        # form new interface points
        new_xf.append(torch.stack([x_insert, y_insert, z_insert], dim=1))

    new_xf = torch.cat(new_xf, dim=0)
    distances = torch.cdist(xf[:, :3], new_xf).float()
    overlap_indices = torch.any(distances < 1e-3, dim=0)
    new_xf = new_xf[~overlap_indices]
    new_xf = torch.cat((new_xf, torch.zeros(new_xf.size(0), 1)), dim=-1)
    # print('new_xf.shape', new_xf.shape)
    return new_xf


from pde_pop import  *
pde = popcorn_interface()
def interface_points(N_f):
    # xf = pde.interface_points(N_f)
    # xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xf[:, 0].detach().numpy(), xf[:, 1].detach().numpy(), xf[:, 2].detach().numpy(),
    #            c='b', label='Random', s=20)
    #
    # import matplotlib.ticker as ticker
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    # ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    # ax.view_init(elev=17, azim=-58)
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # plt.tight_layout()
    # plt.grid(True)
    # plt.legend()
    # plt.savefig('sampling_compare/3d_pop_random.png', dpi=300)
    # plt.show()

    #v2
    x = torch.rand(N_f, 3) * 2 - 1
    y = torch.rand(N_f, 3) * 2 - 1
    z = torch.rand(N_f, 3) * 2 - 1
    r = torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z))
    S = torch.zeros(N_f, 3)

    for k in range(12):
        xk, yk, zk = pde.point_k(k)
        square = -(torch.square(x - xk) + torch.square(y - yk) + torch.square(z - zk))
        S += pde.A * torch.exp(square / torch.square(pde.sigma))
    values = r - pde.r0 - S
    mask = torch.abs(values) < 0.1
    xf = torch.stack([x[mask], y[mask], z[mask]], dim=1)
    print('xf.shape', xf.shape)

    import matplotlib.pyplot as plt
    # 使用 matplotlib 进行 3D 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制满足条件的界面点
    ax.scatter(xf[:, 0], xf[:, 1], xf[:, 2], s=1, c='b')
    plt.show()

    return xf


def interface_points_add(xf, num_new_points, perturb_scale):
    """Generate new points around the existing interface points using vectorized operations."""
    perturbations = torch.randn(len(xf) * num_new_points, 3, device=xf.device) * perturb_scale
    original_points = xf.repeat_interleave(num_new_points, dim=0)
    new_points = original_points[:, :3] + perturbations

    r_new = torch.norm(new_points, dim=1, keepdim=True)  # [N*num_new_points, 1]
    S_new = pde.S(new_points)

    # r0, A, sigma, factor, angles_first, angles_second = initialize_parameters()
    r0 = torch.tensor(0.6)

    values_new = r_new - r0 - S_new
    mask = torch.abs(values_new).squeeze(1) < 0.1
    new_xf_candidates = new_points[mask]

    distances = torch.cdist(xf[:, :3], new_xf_candidates).float()
    overlap_mask = torch.any(distances < 1e-3, dim=0)
    final_new_xf = new_xf_candidates[~overlap_mask]
    final_new_xf = torch.cat((final_new_xf, torch.zeros(final_new_xf.size(0), 1, device=xf.device)), dim=-1)
    # print('final_new_xf.shape', final_new_xf.shape)
    return final_new_xf

def plot_interface_points(old_point, add_point):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(old_point[:, 0].detach().numpy(), old_point[:, 1].detach().numpy(), old_point[:, 2].detach().numpy(),
               c='r', label='Original', s=50)
    ax.scatter(add_point[:, 0].detach().numpy(), add_point[:, 1].detach().numpy(), add_point[:, 2].detach().numpy(),
               c='b', label='Refined', s=20, alpha=0.5)

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.view_init(elev=17, azim=-58)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig('sampling_compare/3d_pop_global.png', dpi=300)
    plt.show()

def random_interface_points_ellip(N_f=512):
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xf[:, 0].detach().numpy(), xf[:, 1].detach().numpy(), xf[:, 2].detach().numpy(),
               c='b', label='Random', s=20)

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.view_init(elev=17, azim=-58)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig('sampling_compare/3d_ellip_random.png', dpi=300)
    plt.show()
    return xf


#pop;
# xf = interface_points(N_f=10000)
# xf_random = random_interface_points_ellip(N_f=300)
# new_xf = interface_points_add(xf, 2, 0.09)
# plot_interface_points(xf, new_xf)

#ellip
xf = interface_points_ellip(N_f=100)
xf_random = random_interface_points_ellip(N_f=300)
new_xf = interface_points_add_ellip(xf, 2)
plot_interface_points(xf, new_xf)