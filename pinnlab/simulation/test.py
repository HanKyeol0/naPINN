import numpy as np

data = np.load("cylinder_random_data.npz")
for key in ["u_data", "v_data", "p_data", "u_grid", "v_grid", "p_grid"]:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, NaN_count={np.isnan(arr).sum()}")
    print(f"  nanmin={np.nanmin(arr)}, nanmax={np.nanmax(arr)}")
