import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from scipy.interpolate import griddata


def plot_rectangular_heatmaps_for_parameters(
    df,
    parameters=None,
    recommended_room=None,
    cmap='coolwarm',
    savefig_path=None
):
    """
    Displays a series of rectangular heatmaps (one per parameter),
    where each cell in the heatmap corresponds to one room,
    arranged row-by-row.

    :param df: DataFrame with at least these columns:
               ['Location'] + sensor columns
    :param parameters: list of column names to visualize, e.g. 
                       ['temperature_mean', 'humidity_mean', ...]
    :param recommended_room: (str) If provided, we highlight that room in each heatmap.
    :param cmap: Matplotlib/Seaborn color map to use.
    :param savefig_path: If provided, saves the figure to this path.
    """
    if parameters is None:
        parameters = ['temperature_mean', 'humidity_mean', 'co2_mean', 'light_mean', 'pir_mean']

    # 1) Sort the rooms in a fixed order (alphabetical or by the current order in df)
    #    Here we just use the DataFrame index order:
    df_ordered = df.reset_index(drop=True).copy()
    room_list = df_ordered['Location'].tolist()
    n_rooms = len(room_list)

    # 2) Determine grid size. We want something near sqrt(n_rooms).
    #    For 331 rooms, ~sqrt(331) ~ 18.2, so we choose 19 x 18 = 342 cells
    #    That's enough space for 331 rooms + some empty.
    rows = int(np.ceil(np.sqrt(n_rooms)))  # 19
    cols = int(np.ceil(n_rooms / rows))    # 18

    # 3) Prepare subplots: one column per parameter
    fig, axes = plt.subplots(nrows=1, ncols=len(parameters),
                             figsize=(5 * len(parameters), 6),
                             squeeze=False)
    axes = axes[0]  # because nrows=1

    # 4) Create a dictionary that maps room -> (row, col)
    #    e.g. room i goes to (i//cols, i%cols)
    room_positions = {}
    for i, room_name in enumerate(room_list):
        r = i // cols
        c = i % cols
        room_positions[room_name] = (r, c)

    # For each parameter, build a 2D matrix of shape (rows, cols)
    # fill with the parameter's values. If there's an unused cell,
    # fill with np.nan (to show blank or missing).
    for p_idx, param in enumerate(parameters):
        ax = axes[p_idx]

        # Create an empty matrix
        matrix = np.full((rows, cols), np.nan, dtype=float)

        # Fill it with the param value
        for i, row_name in enumerate(room_list):
            val = df_ordered.loc[i, param]
            r, c = room_positions[row_name]
            matrix[r, c] = val

        # 5) Plot the matrix as a heatmap
        sns.heatmap(
            matrix,
            cmap=cmap,
            ax=ax,
            annot=False,  # turn on if you want numeric values in each cell
            xticklabels=False,
            yticklabels=False
        )
        ax.set_title(param)

        # 6) Highlight the recommended room if given
        if recommended_room is not None and recommended_room in room_positions:
            (rr, rc) = room_positions[recommended_room]
            # Add a rectangle around that cell
            # Heatmap cells go from (top-left) y=0, x=0 
            # but note that the top row is index=0 visually,
            # so we have to invert or not based on how seaborn handles the matrix
            # By default, row=0 is at the top. So row rr is from the top.
            rect = Rectangle(
                xy=(rc, rr),  # x=col, y=row
                width=1,
                height=1,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            ax.add_patch(rect)

        # If you want to label each cell with the room name, that's possible,
        # but for 331 rooms it would be super crowded. 
        # You could do a tooltip approach or just do no labels for large data.

    plt.tight_layout()
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
    plt.show(block=False)


def plot_all_parameters_by_coordinates(
    sensor_df,
    coord_df,
    parameters=None,
    recommended_room=None,
    corners=None,
    cmap="coolwarm",
    savefig_path=None
):
    """
    Plots multiple subplots side by side, each showing a scatter of rooms by (x_coord, y_coord),
    color-coded by a different sensor parameter.

    :param sensor_df: DataFrame with columns like [Location, temperature_mean, humidity_mean, etc.]
    :param coord_df: DataFrame with columns [Location, x_coord, y_coord]
    :param parameters: List of parameters to plot, e.g. ["temperature_mean", "humidity_mean", ...].
                       Defaults to ["temperature_mean","humidity_mean","co2_mean","light_mean","pir_mean"].
    :param recommended_room: (str) If not None, highlight this room in each subplot.
    :param corners: A list of (x, y) tuples for corners of the floor plan (hard-coded).
    :param cmap: Color map for the scatter color scale.
    :param savefig_path: If provided, saves the figure.
    """
    if parameters is None:
        parameters = ["temperature_mean", "humidity_mean", "co2_mean", "light_mean", "pir_mean"]

    # Create a dict mapping from room -> { param: value, ... }
    # For example: room_sensors["TY-1110"]["temperature_mean"] = 22.3
    # So we can quickly look up each parameter for each room.
    room_sensors = {}
    for _, row in sensor_df.iterrows():
        loc = row["Location"]
        if loc not in room_sensors:
            room_sensors[loc] = {}
        for p in parameters:
            room_sensors[loc][p] = row.get(p, np.nan)

    # Prepare figure: one column per parameter
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(parameters),
        figsize=(5 * len(parameters), 6),
        squeeze=False
    )
    axes = axes[0]  # since nrows=1

    # We'll separate the corners from actual rooms if needed
    # but you said you'll just manually pass them in the `corners` list param.
    # So let's assume `coord_df` has only room coordinates.

    for i, param in enumerate(parameters):
        ax = axes[i]

        # 1) If corners are provided, draw them as a polygon or lines
        if corners and len(corners) > 1:
            # We'll connect them in order + repeat the first corner
            corner_points = np.array(corners + [corners[0]])
            ax.plot(corner_points[:, 0], corner_points[:, 1], color="black", linewidth=2)

        # 2) Build arrays for X, Y, color values
        xs = []
        ys = []
        colors = []
        room_locations = coord_df["Location"].values
        for idx, c_row in coord_df.iterrows():
            loc = c_row["Location"]
            x = c_row["x_coord"]
            y = c_row["y_coord"]

            param_val = np.nan
            if loc in room_sensors:
                param_val = room_sensors[loc].get(param, np.nan)

            xs.append(x)
            ys.append(y)
            colors.append(param_val)

        # 3) Scatter plot
        sc = ax.scatter(
            xs,
            ys,
            c=colors,
            cmap=cmap,
            s=100,
            alpha=0.8
        )
        ax.set_title(param, fontsize=10)

        # 4) Colorbar for each subplot
        # Usually you might want a single colorbar at the right,
        # but to keep it simple, we can do one per subplot using figure.colorbar
        # We pass the scatter object and specify its location
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)

        # 5) Highlight recommended room if provided
        if recommended_room is not None:
            recommended_row = coord_df[coord_df["Location"] == recommended_room]
            if not recommended_row.empty:
                rx = recommended_row["x_coord"].iloc[0]
                ry = recommended_row["y_coord"].iloc[0]
                # Draw a larger ring
                ax.scatter(rx, ry, s=300, facecolors="none", edgecolors="red", linewidths=2)

        # optional: labeling each point with text is possible but can get cluttered
        # for idx, c_row in coord_df.iterrows():
        #     ax.text(c_row["x_coord"], c_row["y_coord"], c_row["Location"], fontsize=6, ha="center")

        ax.set_xlabel("X Coord", fontsize=8)
        ax.set_ylabel("Y Coord", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches="tight")
    plt.show(block=False)


def plot_interpolated_heatmap(
    sensor_df,
    coord_df,
    parameter="temperature_mean",
    corners=None,
    resolution=100,
    recommended_room=None,
    method="nearest"
):
    """
    Creates a continuous heatmap across the rectangle defined by 'corners', 
    interpolating the scattered (x_coord,y_coord, parameter) from sensor_df/coord_df.

    :param sensor_df: DataFrame with columns [Location, temperature_mean, etc.]
    :param coord_df: DataFrame with columns [Location, x_coord, y_coord]
    :param parameter: The sensor column to interpolate, e.g. "temperature_mean"
    :param corners: A list of (x, y) tuples defining the boundary rectangle, e.g. [(0,0),(40,0),(40,20),(0,20)] 
    :param resolution: The number of grid points in each dimension for interpolation. 
                       e.g. 100 => a 100x100 grid
    :param recommended_room: If not None, highlight that room's location
    :param method: Interpolation method for griddata (e.g. "cubic", "linear", or "nearest")
    """
    # 1) Extract data points
    # We'll build arrays for the known (x, y) and the param value
    known_x = []
    known_y = []
    known_vals = []

    # dictionary: {Location -> param_value}
    param_dict = {}
    for _, row in sensor_df.iterrows():
        loc = row["Location"]
        param_val = row.get(parameter, np.nan)
        param_dict[loc] = param_val
    
    # Now fill known_x, known_y, known_vals by matching coord_df
    for _, c_row in coord_df.iterrows():
        loc = c_row["Location"]
        x = c_row["x_coord"]
        y = c_row["y_coord"]
        if loc in param_dict and not pd.isna(param_dict[loc]):
            known_x.append(x)
            known_y.append(y)
            known_vals.append(param_dict[loc])

    known_x = np.array(known_x)
    known_y = np.array(known_y)
    known_vals = np.array(known_vals)

    # 2) Determine bounding box (from corners or from data)
    if corners and len(corners) > 1:
        # We can compute min_x, max_x, min_y, max_y from corners
        x_coords = [pt[0] for pt in corners]
        y_coords = [pt[1] for pt in corners]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
    else:
        # fallback: use the min/max of known_x, known_y
        min_x, max_x = known_x.min(), known_x.max()
        min_y, max_y = known_y.min(), known_y.max()

    # 3) Create a grid for interpolation
    grid_x = np.linspace(min_x, max_x, resolution)
    grid_y = np.linspace(min_y, max_y, resolution)
    # make a meshgrid
    xx, yy = np.meshgrid(grid_x, grid_y)

    # 4) Interpolate using griddata
    # known_x, known_y => known_vals
    # we want to find Z for each (xx[i], yy[i])
    grid_z = griddata(
        points=(known_x, known_y),
        values=known_vals,
        xi=(xx, yy),
        method=method
    )

    # 5) Plot the interpolated heatmap
    plt.figure(figsize=(8,6))
    # We'll do a contourf or imshow approach
    # contourf can be nice for smooth color boundaries
    # If some grid cells are NaN (extrapolation out of range), they'll be blank
    cont = plt.contourf(
        xx, yy, grid_z, 
        levels=100,  # how many contour levels
        cmap="coolwarm",
        alpha=0.8
    )

    plt.colorbar(cont, label=parameter)

    # Optionally draw corners as polygon
    if corners and len(corners) > 1:
        corner_pts = np.array(corners + [corners[0]])
        plt.plot(corner_pts[:,0], corner_pts[:,1], color="black", linewidth=2)

    # 6) Overlay the actual sensor points
    plt.scatter(known_x, known_y, c="black", s=50, label="Sensor points")
    # label them if desired:
    # for (x, y, val) in zip(known_x, known_y, known_vals):
    #     plt.text(x, y, f"{val:.1f}", color="black", fontsize=8)

    # highlight recommended_room if present
    if recommended_room is not None:
        rec_row = coord_df[coord_df["Location"] == recommended_room]
        if not rec_row.empty:
            rx = rec_row["x_coord"].values[0]
            ry = rec_row["y_coord"].values[0]
            plt.scatter(rx, ry, s=200, edgecolors="red", facecolors="none", linewidths=2, label=f"Recommended {recommended_room}")

    plt.xlabel("X Coord")
    plt.ylabel("Y Coord")
    plt.title(f"Interpolated Heatmap of {parameter}")
    plt.legend()
    plt.axis("equal")  # keep aspect ratio
    plt.show(block=False)


def plot_multiple_interpolations_old(
    sensor_df,
    coord_df,
    parameters=["temperature_mean","humidity_mean","co2_mean","light_mean","pir_mean"],
    corners=None,
    recommended_room=None,
    padding_percent=0.1  # Add padding around the data points
):
    from scipy.interpolate import griddata
    from matplotlib.colors import LogNorm, SymLogNorm, Normalize
    
    fig, axes = plt.subplots(nrows=1, ncols=len(parameters), figsize=(5*len(parameters), 6), squeeze=False)
    axes = axes[0]

    # Calculate optimal boundaries if corners not provided
    if not corners:
        x_coords = coord_df['x_coord'].values
        y_coords = coord_df['y_coord'].values
        
        # Get data boundaries
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        # Calculate ranges and center
        x_range = max_x - min_x
        y_range = max_y - min_y
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        
        # Make it more square by taking the larger range
        max_range = max(x_range, y_range)
        
        # Add padding
        padding = max_range * padding_percent
        
        # Create square boundaries
        min_x = center_x - (max_range/2 + padding)
        max_x = center_x + (max_range/2 + padding)
        min_y = center_y - (max_range/2 + padding)
        max_y = center_y + (max_range/2 + padding)
        
        # Create corners in clockwise order
        corners = [
            (min_x, min_y),  # bottom-left
            (max_x, min_y),  # bottom-right
            (max_x, max_y),  # top-right
            (min_x, max_y)   # top-left
        ]

    for i, param in enumerate(parameters):
        ax = axes[i]
        
        # 1) build known_x, known_y, known_vals
        known_x, known_y, known_vals = [], [], []
        param_dict = {}
        for _, row in sensor_df.iterrows():
            loc = row["Location"]
            param_dict[loc] = row.get(param, np.nan)

        for _, c_row in coord_df.iterrows():
            loc = c_row["Location"]
            x = c_row["x_coord"]
            y = c_row["y_coord"]
            if loc in param_dict and not pd.isna(param_dict[loc]):
                known_x.append(x)
                known_y.append(y)
                known_vals.append(param_dict[loc])

        known_x = np.array(known_x)
        known_y = np.array(known_y)
        known_vals = np.array(known_vals)

        # 2) Use the calculated or provided corners for the bounding box
        xs_c = [pt[0] for pt in corners]
        ys_c = [pt[1] for pt in corners]
        min_x, max_x = min(xs_c), max(xs_c)
        min_y, max_y = min(ys_c), max(ys_c)

        # 3) Create a higher resolution grid for smoother interpolation
        resolution = 200  # Increased from 100 for smoother interpolation
        grid_x = np.linspace(min_x, max_x, resolution)
        grid_y = np.linspace(min_y, max_y, resolution)
        xx, yy = np.meshgrid(grid_x, grid_y)

        # 4) Two-step interpolation with transition zone
        # First, use cubic interpolation for the interior
        grid_z = griddata((known_x, known_y), known_vals, (xx, yy), method="cubic")
        
        # Create a mask for points far from any sensor
        from scipy.spatial.distance import cdist
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        sensor_points = np.column_stack((known_x, known_y))
        distances = cdist(grid_points, sensor_points).min(axis=1).reshape(xx.shape)
        
        # Calculate the transition distance (e.g., 20% of the diagonal)
        diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        transition_start = diagonal * 0.1  # Start transition at 10% of diagonal
        transition_end = diagonal * 0.2    # Complete transition at 20% of diagonal
        
        # Create a smooth transition mask
        transition_mask = np.clip((distances - transition_start) / (transition_end - transition_start), 0, 1)
        
        # Fill NaN values with nearest neighbor interpolation
        nearest_z = griddata((known_x, known_y), known_vals, (xx, yy), method="nearest")
        
        # Combine cubic and nearest based on the transition mask
        mask = np.isnan(grid_z)
        grid_z[mask] = nearest_z[mask]
        grid_z = (1 - transition_mask) * grid_z + transition_mask * nearest_z

        # 5) contourf with filled space and custom normalization
        if param == 'co2_mean':
            norm = SymLogNorm(linthresh=1000, linscale=1.0, vmin=min(known_vals), vmax=max(known_vals))
            cont = ax.contourf(xx, yy, grid_z, levels=100, cmap="coolwarm", alpha=0.8, norm=norm)
        else:
            cont = ax.contourf(xx, yy, grid_z, levels=100, cmap="coolwarm", alpha=0.8)

        cb = plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(param)

        # corners
        if corners and len(corners) > 1:
            cpts = np.array(corners + [corners[0]])
            ax.plot(cpts[:,0], cpts[:,1], color="black", linewidth=2)

        # highlight recommended
        if recommended_room:
            rec_row = coord_df[coord_df["Location"]==recommended_room]
            if not rec_row.empty:
                rx = rec_row["x_coord"].iloc[0]
                ry = rec_row["y_coord"].iloc[0]
                ax.scatter(rx, ry, s=100, edgecolors="red", facecolors="none", linewidths=2)

        ax.set_title(param)
        ax.set_aspect("equal", "box")

    plt.tight_layout()
    plt.show(block=False)

#-----------------------------------------------------------------------------------------------------------------------

from scipy.spatial.distance import cdist
from matplotlib.colors import LogNorm, SymLogNorm, Normalize


def plot_multiple_interpolations(
    sensor_df,
    coord_df,
    parameters=["temperature_mean","humidity_mean","co2_mean","light_mean","pir_mean"],
    corners=None,
    recommended_room=None,
    padding_percent=0.05
):
    from scipy.interpolate import Rbf
    from matplotlib.colors import SymLogNorm
    
    # Define manual ranges for temperature and humidity
    value_ranges = {}
    for param in ['temperature_mean', 'humidity_mean']:
        param_values = sensor_df[param].dropna()
        margin = (param_values.max() - param_values.min()) * 0.05  # 5% margin
        value_ranges[param] = (param_values.min() - margin, param_values.max() + margin)

    """    
    value_ranges = {
        'temperature_mean': (21, 22.5),    # adjust these ranges based on your data
        'humidity_mean': (28, 33)        # adjust these ranges based on your data
    }"""
    
    fig, axes = plt.subplots(nrows=1, ncols=len(parameters), figsize=(5*len(parameters), 6), squeeze=False)
    axes = axes[0]

    # Calculate natural rectangular boundaries if not provided
    if not corners:
        x_coords = coord_df['x_coord'].values
        y_coords = coord_df['y_coord'].values
        
        padding_x = (x_coords.max() - x_coords.min()) * padding_percent
        padding_y = (y_coords.max() - y_coords.min()) * padding_percent
        
        min_x = x_coords.min() - padding_x
        max_x = x_coords.max() + padding_x
        min_y = y_coords.min() - padding_y
        max_y = y_coords.max() + padding_y
        
        corners = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ]

    for i, param in enumerate(parameters):
        ax = axes[i]
        
        # Prepare data points
        known_x, known_y, known_vals = [], [], []
        param_dict = {}
        for _, row in sensor_df.iterrows():
            loc = row["Location"]
            param_dict[loc] = row.get(param, np.nan)

        for _, c_row in coord_df.iterrows():
            loc = c_row["Location"]
            x = c_row["x_coord"]
            y = c_row["y_coord"]
            if loc in param_dict and not pd.isna(param_dict[loc]):
                known_x.append(x)
                known_y.append(y)
                known_vals.append(param_dict[loc])

        known_x = np.array(known_x)
        known_y = np.array(known_y)
        known_vals = np.array(known_vals)

        # Create grid
        resolution = 100
        grid_x = np.linspace(min([c[0] for c in corners]), max([c[0] for c in corners]), resolution)
        grid_y = np.linspace(min([c[1] for c in corners]), max([c[1] for c in corners]), resolution)
        xx, yy = np.meshgrid(grid_x, grid_y)

        # Add virtual boundary points to control edge behavior
        if param in value_ranges:
            # Create a wider boundary for placing virtual points
            boundary_padding = 0.2  # 20% extra padding for virtual points
            x_range = max(known_x) - min(known_x)
            y_range = max(known_y) - min(known_y)
            
            virtual_x = []
            virtual_y = []
            virtual_vals = []
            
            # Create virtual points around the boundary
            n_points = 20  # number of virtual points per side
            for edge_x in [min(known_x) - x_range*boundary_padding, max(known_x) + x_range*boundary_padding]:
                ys = np.linspace(min(known_y) - y_range*boundary_padding, max(known_y) + y_range*boundary_padding, n_points)
                virtual_x.extend([edge_x] * n_points)
                virtual_y.extend(ys)
                virtual_vals.extend([min(known_vals)] * n_points)  # Use minimum value for edges
                
            for edge_y in [min(known_y) - y_range*boundary_padding, max(known_y) + y_range*boundary_padding]:
                xs = np.linspace(min(known_x) - x_range*boundary_padding, max(known_x) + x_range*boundary_padding, n_points)
                virtual_x.extend(xs)
                virtual_y.extend([edge_y] * n_points)
                virtual_vals.extend([min(known_vals)] * n_points)  # Use minimum value for edges
            
            # Combine real and virtual points
            combined_x = np.concatenate([known_x, virtual_x])
            combined_y = np.concatenate([known_y, virtual_y])
            combined_vals = np.concatenate([known_vals, virtual_vals])
            
            # RBF interpolation with combined points
            rbf = Rbf(combined_x, combined_y, combined_vals, function='multiquadric', smooth=0.1)
            grid_z = rbf(xx, yy)
        else:
            # For other parameters, use regular RBF
            rbf = Rbf(known_x, known_y, known_vals, function='multiquadric', smooth=0.1)
            grid_z = rbf(xx, yy)

        # Apply value constraints based on parameter
        if param in value_ranges:
            vmin, vmax = value_ranges[param]
            grid_z = np.clip(grid_z, vmin, vmax)
            cont = ax.contourf(xx, yy, grid_z, levels=100, cmap="coolwarm", alpha=0.8,
                             vmin=vmin, vmax=vmax)
        elif param == 'co2_mean':
            norm = SymLogNorm(linthresh=1000, linscale=1.0, vmin=min(known_vals), vmax=max(known_vals))
            cont = ax.contourf(xx, yy, grid_z, levels=100, cmap="coolwarm", alpha=0.8, norm=norm)
        else:
            cont = ax.contourf(xx, yy, grid_z, levels=100, cmap="coolwarm", alpha=0.8)

        plt.colorbar(cont, ax=ax, fraction=0.046, pad=0.04, label=param)

        # Draw boundary
        if corners and len(corners) > 1:
            corner_points = np.array(corners + [corners[0]])
            ax.plot(corner_points[:,0], corner_points[:,1], color="black", linewidth=2)

        # Highlight recommended room
        if recommended_room:
            rec_row = coord_df[coord_df["Location"]==recommended_room]
            if not rec_row.empty:
                rx = rec_row["x_coord"].iloc[0]
                ry = rec_row["y_coord"].iloc[0]
                ax.scatter(rx, ry, s=100, edgecolors="red", facecolors="none", linewidths=2)

        ax.set_title(param)
        ax.set_aspect("equal", "box")

    plt.tight_layout()
    plt.show(block=False)