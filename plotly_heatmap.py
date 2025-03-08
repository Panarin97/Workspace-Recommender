# interactive_heatmap.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf
import plotly.io as pio

# Set default template for Plotly
pio.templates.default = "plotly_white"

def create_plotly_interpolated_maps(
    sensor_df,
    coord_df,
    parameters=["temperature_mean", "humidity_mean", "co2_mean", "light_mean", "pir_mean"],
    recommended_room=None,
    padding_percent=0.05
):
    """
    Creates interactive interpolated heatmaps for each parameter using Plotly.
    
    Args:
        sensor_df: DataFrame with sensor data (includes Location and parameter columns)
        coord_df: DataFrame with coordinates (includes Location, x_coord, y_coord)
        parameters: List of parameters to visualize
        recommended_room: ID of room to highlight (optional)
        padding_percent: Padding to add around data boundaries (default: 0.05)
        
    Returns:
        Dictionary of Plotly figures for each parameter
    """
    # Create value ranges for parameters that need custom scaling
    value_ranges = {}
    for param in ['temperature_mean', 'humidity_mean']:
        param_values = sensor_df[param].dropna()
        margin = (param_values.max() - param_values.min()) * 0.05  # 5% margin
        value_ranges[param] = (param_values.min() - margin, param_values.max() + margin)
    
    # Store figures in a dictionary
    figures = {}
    
    for param in parameters:
        # Extract coordinates and values
        data_points = []
        for _, sensor_row in sensor_df.iterrows():
            loc = sensor_row['Location']
            coord_row = coord_df[coord_df['Location'] == loc]
            
            if not coord_row.empty and not pd.isna(sensor_row[param]):
                x = coord_row['x_coord'].values[0]
                y = coord_row['y_coord'].values[0]
                val = sensor_row[param]
                data_points.append((x, y, val))
        
        if not data_points:
            continue
            
        x, y, z = zip(*data_points)
        x, y, z = np.array(x), np.array(y), np.array(z)
        
        # Calculate natural boundaries with padding
        padding_x = (max(x) - min(x)) * padding_percent
        padding_y = (max(y) - min(y)) * padding_percent
        
        x_min, x_max = min(x) - padding_x, max(x) + padding_x
        y_min, y_max = min(y) - padding_y, max(y) + padding_y
        
        # Create grid for interpolation
        resolution = 100
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_y = np.linspace(y_min, y_max, resolution)
        
        # Add virtual boundary points for better interpolation
        boundary_padding = 0.2  # 20% extra padding for virtual points
        x_range = max(x) - min(x)
        y_range = max(y) - min(y)
        
        virtual_x = []
        virtual_y = []
        virtual_vals = []
        
        # Create virtual points around the boundary
        n_points = 20  # number of virtual points per side
        for edge_x in [min(x) - x_range*boundary_padding, max(x) + x_range*boundary_padding]:
            ys = np.linspace(min(y) - y_range*boundary_padding, max(y) + y_range*boundary_padding, n_points)
            virtual_x.extend([edge_x] * n_points)
            virtual_y.extend(ys)
            virtual_vals.extend([min(z)] * n_points)  # Use minimum value for edges
            
        for edge_y in [min(y) - y_range*boundary_padding, max(y) + y_range*boundary_padding]:
            xs = np.linspace(min(x) - x_range*boundary_padding, max(x) + x_range*boundary_padding, n_points)
            virtual_x.extend(xs)
            virtual_y.extend([edge_y] * n_points)
            virtual_vals.extend([min(z)] * n_points)  # Use minimum value for edges
        
        # Combine real and virtual points
        combined_x = np.concatenate([x, virtual_x])
        combined_y = np.concatenate([y, virtual_y])
        combined_vals = np.concatenate([z, virtual_vals])
        
        # Perform RBF interpolation
        if param in value_ranges:
            rbf = Rbf(combined_x, combined_y, combined_vals, function='multiquadric', smooth=0.1)
        else:
            rbf = Rbf(x, y, z, function='multiquadric', smooth=0.1)
        
        # Create meshgrid and interpolate values
        XX, YY = np.meshgrid(grid_x, grid_y)
        if param in value_ranges:
            grid_z = rbf(XX, YY)
            vmin, vmax = value_ranges[param]
            grid_z = np.clip(grid_z, vmin, vmax)
        else:
            grid_z = rbf(XX, YY)
        
        # Create figure
        fig = go.Figure()
        
        # Add contour plot
        colorscale = 'RdBu_r'  # Default colorscale
        if param == 'co2_mean':
            colorscale = 'Plasma'
        
        contour = go.Contour(
            z=grid_z,
            x=grid_x,
            y=grid_y,
            colorscale=colorscale,
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(
                title=dict(
                    text=get_parameter_display_name(param),
                    side='right',  # Move 'right' to be inside the title dict
                    font=dict(size=14)  # Move font settings inside title dict
                )
            ),
            hoverinfo='none'
        )
        
        fig.add_trace(contour)
        
        # Add sensor points - now much smaller and less visible
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=3,  # Much smaller dots
                color='gray',  # Light gray color for less visual dominance
                opacity=0.5,   # Semi-transparent
                symbol='circle'
            ),
            name='Sensor Locations',
            hovertemplate='<b>%{text}</b><br>Value: %{customdata:.2f}<extra></extra>',
            text=[f"Room: {sensor_df.loc[sensor_df['Location'] == loc, 'Location'].values[0]}" 
                  for loc in [loc for i, loc in enumerate(sensor_df['Location']) if loc in coord_df['Location'].values]],
            customdata=z
        ))
        
        # Add recommended room highlight if specified
        if recommended_room:

            recommended_rooms = [recommended_room] if isinstance(recommended_room, str) else recommended_room

            for room_id in recommended_rooms:
                rec_row = coord_df[coord_df['Location'] == room_id]
                if not rec_row.empty:
                    rx = rec_row['x_coord'].values[0]
                    ry = rec_row['y_coord'].values[0]
                    
                    # Find the value for this room
                    room_value = None
                    room_row = sensor_df[sensor_df['Location'] == room_id]
                    if not room_row.empty and param in room_row.columns:
                        room_value = room_row[param].values[0]
                    
                    tooltip_text = f"Selected Room(s): {room_id}"
                    if room_value is not None:
                        tooltip_text += f"<br>Value: {room_value:.2f}"
                    
                    fig.add_trace(go.Scatter(
                        x=[rx],
                        y=[ry],
                        mode='markers',
                        marker=dict(
                            size=12,  # Larger, but not too large
                            color='red',
                            symbol='circle',
                            line=dict(width=2, color='red')
                        ),
                        name=f'Recommended: {room_id}',
                        hovertemplate=tooltip_text + '<extra></extra>'
                    ))
            
        # Improve layout
        fig.update_layout(
            title=dict(
                text=f"{get_parameter_display_name(param)} Distribution",
                font=dict(size=16)
            ),
            xaxis=dict(
                title="X Coordinate",
                showgrid=False
            ),
            yaxis=dict(
                title="Y Coordinate",
                showgrid=False,
                scaleanchor="x",  # Make aspect ratio 1:1
                scaleratio=1
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=60, b=40),
            height=650,
            width=500
        )
        
        figures[param] = fig
    
    return figures

def get_parameter_display_name(param):
    """Convert parameter code names to display names"""
    display_names = {
        'temperature_mean': 'Temperature (°C)',
        'humidity_mean': 'Humidity (%)',
        'co2_mean': 'CO₂ (ppm)',
        'light_mean': 'Light (lux)',
        'pir_mean': 'Occupancy'
    }
    return display_names.get(param, param)