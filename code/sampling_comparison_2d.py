import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels
rcParams['text.usetex'] = True



# def setup_seed(seed):
#     torch.manual_seed(seed)  # Results of random operations on the CPU will be deterministic and reproducible.
# setup_seed(888888)

def generate_grid_points(h):
    x_coords = torch.linspace(-1, 1, steps=h)
    y_coords = torch.linspace(-1, 1, steps=h)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    grid_points_2d = torch.stack([X, Y], dim=-1)
    grid_points_2d = grid_points_2d.view(-1, 2)

    counts = torch.randint(0, 4, (grid_points_2d.size(0), 1))#generate count value 0,1,2,3
    grid_points_3d = torch.cat((grid_points_2d, counts.float()), dim=-1)
    return grid_points_3d


def add_grid(points, refinement_factor, h):
    refined_points = []
    for i, point in enumerate(points):
        count = point[-1].item()  #current count
        initial_spacing = (1 / (h - 1)) / (2 ** count)  # compute \delta h
        for dx in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
            for dy in np.linspace(-initial_spacing, initial_spacing, refinement_factor):
                new_point = point[:3] + torch.tensor([dx, dy, 0], dtype=points.dtype)
                # new_point[-1] = count + 1#original
                new_point[-1] = count #mimic the adaptive sampling process

                if not any(torch.allclose(new_point[:2], p[:2]) for p in points):
                    refined_points.append(new_point)
    refined_points = [p for p in refined_points if -1 <= p[0] <= 1 and -1 <= p[1] <= 1]
    refined_points = torch.stack(refined_points)
    refined_points, _ = torch.unique(refined_points, dim=0, return_inverse=True)
    refined_points = refined_points.clone().detach().requires_grad_(True)
    return refined_points

# global（count=0）
def plot_global_points(original_points, refined_points):
    original_x = original_points[:, 0].numpy()
    original_y = original_points[:, 1].numpy()

    refined_x = refined_points.detach().numpy()[:, 0]
    refined_y = refined_points.detach().numpy()[:, 1]

    plt.figure()

    plt.scatter(original_x, original_y, color='blue', label='Original', s=50)

    plt.scatter(refined_x, refined_y, color='red', label='Refined', s=20)

    # plt.title('Global Refinement (t=0)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sampling_compare/2d_global.png', dpi=300)
    plt.show()


# different counts
def plot_count_points(original_points, refined_points):
    original_x = original_points[:, 0].numpy()
    original_y = original_points[:, 1].numpy()
    original_counts = original_points[:, 2].numpy()

    refined_x = refined_points.detach().numpy()[:, 0]
    refined_y = refined_points.detach().numpy()[:, 1]
    refined_counts = refined_points.detach().numpy()[:, 2]

    plt.figure()

    # color map
    colors = ['blue', 'green', 'orange', 'red']
    labels = [r'$t$ = 0', r'$t$ = 1', r'$t$ = 2', r'$t$ = 3']

    # original
    for count in range(4):
        mask = original_counts == count
        plt.scatter(original_x[mask], original_y[mask], color=colors[count], label=labels[count], s=50)

    # new points
    for count in range(4):
        mask = refined_counts == count
        plt.scatter(refined_x[mask], refined_y[mask], color=colors[count], s=20)

    # 设置图形标题和标签
    # plt.title('Refinement by Count Value')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sampling_compare/2d_count.png', dpi=300)

    plt.show()


def generate_random_points(num_points):
    # 在 [-1, 1]^2 范围内随机生成点
    random_points = torch.rand(num_points, 2) * 2 - 1  # 范围 [-1, 1]
    # 添加 count 值（这里设为 0，因为随机点没有 count 的概念）
    random_points = torch.cat((random_points, torch.zeros(num_points, 1)), dim=-1)
    return random_points


def plot_random_points(random_points):
    random_x = random_points[:, 0].numpy()
    random_y = random_points[:, 1].numpy()

    plt.figure()

    plt.scatter(random_x, random_y, color='blue', label='Random', s=20)

    # plt.title('Random Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.savefig('sampling_compare/2d_random.png', dpi=300)
    plt.show()



h = 5
re_factor = 3
grid_points = generate_grid_points(h)

# global（count=0）
global_grid_points = grid_points.clone()
global_grid_points[:, 2] = 0
global_refined_points = add_grid(global_grid_points, re_factor, h)
plot_global_points(global_grid_points, global_refined_points)
#
# #count
# refined_points = add_grid(grid_points, re_factor, h)
# plot_count_points(grid_points, refined_points)
#
# # random
# random_points = generate_random_points(100)
# plot_random_points(random_points)


'''interface'''
import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成初始的 interface 点
def interface_points(N_f=32):
    theta_initial = torch.linspace(0, 2 * torch.pi, N_f, dtype=torch.float32)
    theta_initial = theta_initial.reshape(-1, 1)
    r = 1/2 + torch.sin(5*theta_initial)/7  # 花朵形状的半径
    x_initial = r * torch.cos(theta_initial)
    y_initial = r * torch.sin(theta_initial)
    xf = torch.cat((x_initial, y_initial), dim=1)
    xf = torch.cat((xf, torch.zeros(xf.size(0), 1)), dim=-1)  # 添加 count 值（初始为 0）
    return xf

# 在 interface 点之间插入新点
def interface_points_add(xf, n_insert=1):
    N_f = xf.shape[0]
    new_xf = []
    for i in range(N_f):
        theta_start = torch.atan2(xf[i, 1], xf[i, 0]).item()
        theta_end = torch.atan2(xf[(i + 1) % N_f, 1], xf[(i + 1) % N_f, 0]).item()
        if theta_end < theta_start:
            theta_end += 2 * torch.pi
        theta_insert = torch.linspace(theta_start, theta_end, n_insert + 2, dtype=torch.float32)[:-1]
        # 花朵形状
        r_insert = 1 / 2 + torch.sin(5 * theta_insert) / 7
        x_insert = r_insert * torch.cos(theta_insert)
        y_insert = r_insert * torch.sin(theta_insert)
        new_xf.append(torch.stack([x_insert, y_insert], dim=1))
    new_xf = torch.cat(new_xf, dim=0)
    # 去除与原始点重叠的点
    distances = torch.cdist(xf[:, :2], new_xf).float()
    overlap_indices = torch.any(distances < 1e-2, dim=0)
    new_xf = new_xf[~overlap_indices]
    new_xf = torch.cat((new_xf, torch.zeros(new_xf.size(0), 1)), dim=-1)  # 添加 count 值（初始为 0）
    return new_xf

# 绘制 interface 点
def plot_interface_points(original_points, new_points):
    # 提取原始点的 x 和 y 坐标
    original_x = original_points[:, 0].numpy()
    original_y = original_points[:, 1].numpy()

    # 提取新增点的 x 和 y 坐标
    new_x = new_points[:, 0].numpy()
    new_y = new_points[:, 1].numpy()

    # 创建一个新的图形
    plt.figure()

    # 绘制原始点
    plt.scatter(original_x, original_y, color='blue', label='Original', s=50)

    # 绘制新增点
    plt.scatter(new_x, new_y, color='red', label='Refined', s=20)

    # 设置图形标题和标签
    # plt.title('Original vs New Interface Points')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.grid(True)
    # 调整布局，避免图例遮挡图片
    plt.tight_layout()
    plt.legend()
    plt.savefig('sampling_compare/2d_interface_global.png', dpi=300)
    # 显示图形
    plt.show()

def random_interface_points(N_f=256):
    theta = 2*torch.pi*torch.rand(N_f,1)
    r = 1/2 + torch.sin(5*theta)/7
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    xf = torch.cat((x, y), dim=1)

    plt.figure()
    plt.scatter(xf[:, 0], xf[:, 1], color='blue', label='Random', s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sampling_compare/2d_interface_random.png', dpi=300)
    plt.show()
    return xf


# 生成初始的 interface 点
xf = interface_points(N_f=32)
xf_random = random_interface_points(N_f=100)
# 在 interface 点之间插入新点
new_xf = interface_points_add(xf, n_insert=2)

# 绘制 interface 点
plot_interface_points(xf, new_xf)


