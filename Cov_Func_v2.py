import numpy as np

def Cov_Func_v2(pop, rs, Obstacle_Area, Covered_Area):
    # reset vùng đã phủ
    Covered_Area[Covered_Area != 0] = 0

    inside_sector = np.zeros_like(Covered_Area, dtype=bool)
    size_x, size_y = Covered_Area.shape
# %%
    for j in range(pop.shape[0]):
        x0, y0 = pop[j]
        rsJ = rs[j]

        # ràng buộc biên
        x_ub = min(int(np.ceil(x0 + rsJ)), size_x - 1)
        x_lb = max(int(np.floor(x0 - rsJ)), 0)
        y_ub = min(int(np.ceil(y0 + rsJ)), size_y - 1)
        y_lb = max(int(np.floor(y0 - rsJ)), 0)

        # tạo lưới con
        X, Y = np.meshgrid(np.arange(x_lb, x_ub + 1), np.arange(y_lb, y_ub + 1), indexing='ij')

        D = np.sqrt((X - x0)**2 + (Y - y0)**2)
        in_circle = D <= rsJ

        # cập nhật vùng trong sector
        inside_sector[x_lb:x_ub + 1, y_lb:y_ub + 1] |= (in_circle)

# %%
    # cập nhật vùng phủ
    Covered_Area = inside_sector * Obstacle_Area

    # xử lý vùng vật cản bị phủ
    mask_obstacle = (Obstacle_Area == 0) & (Covered_Area == 1)
    Covered_Area[mask_obstacle] = -2

    count1 = np.count_nonzero(Covered_Area == 1)
    count2 = np.count_nonzero(Covered_Area == -2)
    count3 = np.count_nonzero(Obstacle_Area == 1)
    coverage = 1 - (count1 - count2) / count3 if count3 > 0 else 0

    Covered_Area[Covered_Area == -2] = -1

    return coverage, Covered_Area
