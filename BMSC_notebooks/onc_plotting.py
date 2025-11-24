"""
Plotting functions for ONC sensor data visualization.
Functions take DataFrames and configuration from onc_config module.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import plotly.express as px
import matplotlib.colors as mcolors
import cmocean


from onc_config import SENSOR_INFO, SENSOR_COLORS, PLACE, HYPOXIA_THRESHOLD

# Hypoxia threshold (ml/l)
HYPOXIA_THRESHOLD: float = 1.4

# Color mapping for sensors
SENSOR_COLORS: dict[str:str] = {
    "Oxygen (ml/l)": "royalblue",
    "PAR (µmol/m²/s)": "goldenrod",
    "Chlorophyll (µg/l)": "darkgreen",
    "Temperature (°C)": "crimson",
    "Salinity (psu)": "#b1d12e",
    "Turbidity (NTU)": "saddlebrown",
    "Conductivity (S/m)": "mediumorchid",
    "Density (kg/m3)": "darkcyan",
    "Sigma-t (kg/m3)": mcolors.to_hex(cmocean.cm.thermal(0.75))
}

# Location information
PLACE: dict[str:str] = {
    "FGPPN": {
        "name": "Folger Pinnacle",
        "mountCode": "FGPPN",
        "castCode": "CF341",
        "mountDepth": 25,
    }, "FGPD": {
        "name": "Folger Deep",
        "mountCode": "FGPD",
        "castCode": "CF340",
        "mountDepth": 100,
    }
}

def plot_dataframe(df: pd.DataFrame, location_code: str, start: pd.Timestamp = None, end: pd.Timestamp = None) -> None:
    """
    Plots each numeric sensor column in the DataFrame against time, 
    with line priority and unit-labeled legend entries.

    Args:
        df (pd.DataFrame): Time-indexed DataFrame with sensor columns
        location_code (str): ONC location code (e.g., "FGPPN")
        start (pd.Timestamp, optional): Start time for x-axis. Defaults to df.index[0].
        end (pd.Timestamp, optional): End time for x-axis. Defaults to df.index[-1].
    """
    # 1. Data Preparation
    plot_df = df.copy()
    start_time = start or plot_df.index[0]
    end_time = end or plot_df.index[-1]

    # 2. Plot Setup
    fig, ax = plt.subplots(figsize=(16, 9))

    # 3. Plot Sensor Lines
    for col in plot_df.columns:
        color = SENSOR_COLORS.get(col, "black")
        z_order = 10 if col == "Oxygen (ml/l)" else 1
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, color=color, zorder=z_order)

    # 4. Axis Formatting

    ## X-Axis (Date)
    # Calculate number of full months in range
    num_months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
    num_days = (end_time - start_time).days

    # Choose appropriate locator based on range
    if num_days <= 31:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)  # weekly
    elif num_months <= 3:
        locator = mdates.DayLocator(bymonthday=[1, 15])        # twice a month
    elif num_months % 2 != 0 and num_months <= 10:
        locator = mdates.MonthLocator(interval=1, bymonthday=1)
    else:
        locator = mdates.MonthLocator(interval=2, bymonthday=1)

    formatter = mdates.DateFormatter('%b %d, %Y')
    locator.prune = None  # don't cut off start or end ticks

    # Apply x-axis formatting
    time_range = end_time - start_time
    x_padding = time_range * 0.03 
    ax.set_xlim(start_time - x_padding, end_time + x_padding)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ## Y-Axis (Sensor Values)
    y_min = 0
    y_max_data = plot_df.max().max()
    y_padding = (y_max_data - y_min) * 0.01  # Add 1% headroom
    ax.set_ylim(bottom=y_min - y_padding, top=y_max_data + y_padding)

    # 5. Labels and Title
    ax.set_xlabel("Date", labelpad=12)
    ax.set_ylabel("Sensor Value", labelpad=12)
    ax.set_title(
        f"{PLACE[location_code]['name']}\n"
        f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        pad=15
    )

    # 6. Annotations and Legend
    ax.plot([start_time, end_time], [HYPOXIA_THRESHOLD, HYPOXIA_THRESHOLD], color='red', linestyle='--', linewidth=1, label=f"Hypoxia Threshold ({HYPOXIA_THRESHOLD} ml/l)")
    ax.legend(loc="upper right")

    # 7. Grid and Show
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def subplot_all_with_oxygen(df: pd.DataFrame, 
                            location_code: str, 
                            start: pd.Timestamp = None, 
                            end: pd.Timestamp = None) -> None:
    """
    Creates subplots showing oxygen vs each other sensor over time.
    Applies consistent styling, color, and a hypoxia threshold line.

    Args:
        df (pd.DataFrame): Time-indexed DataFrame containing oxygen column
        location_code (str): ONC location code (e.g., "FGPPN")
        start (pd.Timestamp, optional): Start time for x-axis
        end (pd.Timestamp, optional): End time for x-axis
    """
    # 1. Data Preparation
    plot_df = df.copy()
    oxygen_col = "Oxygen (ml/l)"

    if oxygen_col not in plot_df.columns:
        raise ValueError(f"'{oxygen_col}' not found in DataFrame columns.")

    start_time = start or plot_df.index[0]
    end_time = end or plot_df.index[-1]
    time_range = end_time - start_time
    x_padding = time_range * 0.03

    sensor_cols = [col for col in plot_df.columns if col != oxygen_col]
    num_sensors = len(sensor_cols)

    # 2. Plot Setup
    fig, axs = plt.subplots(num_sensors, 1, figsize=(16, 3.2 * num_sensors))
    if num_sensors == 1:
        axs = [axs]

    # 3. Plot Each Subplot
    for i, sensor_col in enumerate(sensor_cols):
        ax = axs[i]
        ax_right = ax.twinx()

        # 3.1 Plot Oxygen
        oxy_color = SENSOR_COLORS.get(oxygen_col, "royalblue")
        sensor_color = SENSOR_COLORS.get(sensor_col, "black")

        ax.plot(plot_df.index, plot_df[oxygen_col], color=oxy_color, linewidth=0.8, label=oxygen_col)
        ax.set_ylabel(oxygen_col, color=oxy_color, labelpad=12)
        ax.tick_params(axis='y', labelcolor=oxy_color)
        oxy_max = plot_df[oxygen_col].max()
        ax.set_ylim(0, max(oxy_max * 1.03, HYPOXIA_THRESHOLD + 0.5))

        ax.plot([start_time, end_time], [HYPOXIA_THRESHOLD, HYPOXIA_THRESHOLD], color='red', linestyle='--', linewidth=1, label=f"Hypoxia Threshold ({HYPOXIA_THRESHOLD} ml/l)")

        # 3.2 Plot Other Sensor
        ax_right.plot(plot_df.index, plot_df[sensor_col], color=sensor_color, linewidth=0.8, label=sensor_col)
        ax_right.set_ylabel(sensor_col, color=sensor_color, labelpad=12)
        ax_right.tick_params(axis='y', labelcolor=sensor_color)
        if plot_df[sensor_col].min() <= 1:
            ax_right.set_ylim(bottom=0)

        # 3.3 Match Tick Counts
        ax_right.yaxis.set_major_locator(MaxNLocator(nbins=len(ax.get_yticks()), prune=None))

        # 3.4 Subplot Title
        ax.set_title(f"{oxygen_col} vs {sensor_col}", y=1.0, pad=10, fontsize=12)

        # 3.5 Combined Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        l = ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        l.set_zorder(100)

        # 3.6 Grid
        ax.grid(True, linestyle='--', linewidth=0.5)

    # 4. X-Axis Formatting
    num_months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
    num_days = (end_time - start_time).days

    if num_days <= 31:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
    elif num_months <= 3:
        locator = mdates.DayLocator(bymonthday=[1, 15])
    elif num_months % 2 != 0 and num_months <= 10:
        locator = mdates.MonthLocator(interval=1, bymonthday=1)
    else:
        locator = mdates.MonthLocator(interval=2, bymonthday=1)

    formatter = mdates.DateFormatter('%b %d, %Y')
    locator.prune = None

    for ax in axs:
        ax.set_xlim(start_time - x_padding, end_time + x_padding)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    axs[-1].set_xlabel("Date", labelpad=15)

    # 5. Overall Title and Layout
    fig.suptitle(
        f"{PLACE[location_code]['name']}\n"
        f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        y=0.98, x=0.51
    )
    plt.subplots_adjust(top=0.89, hspace=0.4)
    plt.show()

def subplot_all_with_time(df: pd.DataFrame, location_code: str, start: pd.Timestamp = None, end: pd.Timestamp = None) -> None:
    """
    Subplots all properties in a data frame against time.

    Args:
        df (pd.DataFrame): Time-indexed DataFrame with sensor columns
        location_code (str): ONC location code (e.g., "FGPPN")
        start (pd.Timestamp, optional): Start time for x-axis
        end (pd.Timestamp, optional): End time for x-axis
    """
    # 1. Data Preparation
    start_time = start or df.index[0]
    end_time = end or df.index[-1]
    time_range = end_time - start_time
    padding = time_range * 0.03  # 3% padding on each side
    sensor_cols = df.columns.to_list()

    # 2. Plot Setup
    fig, axes = plt.subplots(figsize=(16, len(sensor_cols) * 4), nrows=len(sensor_cols), ncols=1)
    if len(sensor_cols) == 1:
        axes = [axes]  # ensure iterable

    # 3. Plot Each Sensor Column
    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        color = SENSOR_COLORS[col] or "black"
        z_order = 0.8
        label = col

        # Plot the sensor line
        ax.plot(df.index, df[col], color=color, linewidth=0.7, zorder=z_order, label=label)

        # Labels and title
        ax.set_title(f"{PLACE[location_code]['name']} - {label}")
        ax.set_ylabel(label, labelpad=13)
        ax.set_xlabel("Date", labelpad=13)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

        # 4. X-Axis Formatting
        # Calculate how many full months are in the range
        num_months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
        num_days = (end_time - start_time).days

        # Choose appropriate locator based on range
        if num_days <= 31:
            locator = mdates.WeekdayLocator(byweekday=mdates.MO)  # weekly
        elif num_months <= 3:
            locator = mdates.DayLocator(bymonthday=[1, 15])        # twice a month
        elif num_months % 2 != 0 and num_months <= 10:
            locator = mdates.MonthLocator(interval=1, bymonthday=1)
        else:
            locator = mdates.MonthLocator(interval=2, bymonthday=1)

        formatter = mdates.DateFormatter('%b %d, %Y')
        locator.prune = None  # don't cut off start or end ticks
        
        ax.xaxis.set_major_formatter(formatter)

        # 5. Add Hypoxia Threshold (if oxygen)
        if "oxygen" in col.lower():
            ax.plot([start_time, end_time], [HYPOXIA_THRESHOLD, HYPOXIA_THRESHOLD], color='red', linestyle='--', linewidth=1, label=f"Hypoxia Threshold ({HYPOXIA_THRESHOLD} ml/l)")
            legend = ax.legend()
            legend.set_zorder(50)

    # 6. Shared X-Ticks Across Subplots
    shared_xticks = axes[0].get_xticks()
    for ax in axes[1:]:
        ax.set_xticks(shared_xticks)

    # 7. Title and Layout
    fig.suptitle(
        f"{PLACE[location_code]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        y=0.97,
        x=0.51
    )

    plt.subplots_adjust(top=0.93, hspace=0.4)
    plt.show()

def compare_sensor_subplots(df1: pd.DataFrame, df2: pd.DataFrame, sensor_cols: list[str], location_code1: str, location_code2: str) -> None:
    """
    Creates subplots comparing the same sensor parameters from two locations.

    Args:
        df1 (pd.DataFrame): First location's data (timestamp as index)
        df2 (pd.DataFrame): Second location's data (timestamp as index)
        sensor_cols (list[str]): List of sensor columns to plot
        location_code1 (str): Location code for first dataset (e.g., 'FGPPN')
        location_code2 (str): Location code for second dataset (e.g., 'FGPD')
    """
    # Check time period in each dataframe (i.e. for missing data)
    min_time = min(df1.index[0], df2.index[0])
    max_time = max(df1.index[-1], df2.index[-1])

    # Define figure and axes for subplots
    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(16, len(sensor_cols)*4))

    if len(sensor_cols) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, col in zip(axes, sensor_cols):

        label = col  # subplot title

        # color options: seagreen & tomato, royalblue & crimson, teal & darkorange, Slateblue & firebrick. steelblue & indianred
        ax.plot(df2.index, df2[col], label=f"{PLACE[location_code2]['name']} - {PLACE[location_code2]['mountDepth']}m", linewidth=0.8, color="firebrick")
        ax.plot(df1.index, df1[col], label=f"{PLACE[location_code1]['name']} - {PLACE[location_code1]['mountDepth']}m", linewidth=0.8, color="slateblue")

        ax.set_title(f"{label}", fontsize=12)
        ax.set_ylabel(col, labelpad=12)
        ax.set_xlabel("Time", labelpad=12)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()

        # Compute dynamic padding (e.g. 3% of full time range)
        time_range = max_time - min_time
        padding = time_range * 0.03  # 3% padding on each side
        ax.set_xlim(min_time - padding, max_time + padding)  # Make sure all axes span same range

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    start_time = df1.index[0]
    end_time = df1.index[-1]

    fig.suptitle(
        f"{PLACE[location_code2]['name']} vs {PLACE[location_code1]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        y=0.97,
        x=0.51
    )

    plt.subplots_adjust(top=0.89, hspace=0.45)
    plt.show()

def plot_with_twin_y_axis_for_outlier(df: pd.DataFrame, location_code: str) -> None:
    """
    Plots time series data using a twin y-axis. The highest-magnitude parameter is plotted on the
    right y-axis using its defined color and label; all others are on the left.

    Args:
        df (pd.DataFrame): Time-indexed DataFrame of numeric sensor columns
        location_code (str): ONC location code (e.g., "FGPPN")
    """
    # For title
    start_time = df.index[0]
    end_time = df.index[-1]

    # Identify the highest-magnitude column
    mean_mags = df.abs().mean()
    high_param = mean_mags.idxmax()
    low_params = [col for col in df.columns if col != high_param]

    # Set up plot
    fig, ax_left = plt.subplots(figsize=(16, 9))
    ax_right = ax_left.twinx()

    # Plot left y-axis (all but the high param)
    for col in low_params:
        color = SENSOR_COLORS[col] or "black"
        label = col
        z_order = 10 if col == "Oxygen (ml/l)" else 1  # default base layer

        ax_left.plot(df.index, df[col], label=label, color=color, linewidth=0.8, zorder=z_order)

    # Plot right y-axis (high-magnitude param)
    color = SENSOR_COLORS[high_param] or "black"
    label = high_param
    
    ax_right.plot(df.index, df[high_param], label=label, color=color, linewidth=0.8)
    ax_right.tick_params(axis='y', colors=color)
    ax_right.yaxis.label.set_color(color)

    formatter = mdates.DateFormatter('%b %d, %Y')
    ax_left.xaxis.set_major_formatter(formatter)

    # Axis labels
    ax_left.set_ylabel("Standard Scale Parameters")
    ax_right.set_ylabel(label, labelpad=13)
    ax_left.set_xlabel("Date", labelpad=13)

    # Combine legends
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.grid(True, linestyle="--", alpha=0.6)
    ax_left.legend(lines_left + lines_right, labels_left + labels_right)

    # Title
    ax_left.set_title(f"{PLACE[location_code]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}", fontweight='bold')

    fig.tight_layout()

    # Match the y-axis heights proportionally
    left_min, left_max = ax_left.get_ylim()
    right_min, right_max = ax_right.get_ylim()

    # Compute data ranges
    left_range = left_max - left_min
    right_range = right_max - right_min

    if right_range < left_range:
        diff = left_range - right_range
        new_right_max = right_max + (diff/2)
        new_right_min = right_min - (diff/2)
        ax_right.set_ylim(new_right_min, new_right_max)

    plt.show()

def plot_dataframe_plotly(df: pd.DataFrame, location_code: str) -> None:
    """
    Creates an interactive Plotly line chart for all numeric sensor columns in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with datetime index and sensor columns
        location_code (str): ONC location code (e.g., "FGPPN")
    """
    start_time = df.index[0]
    end_time = df.index[-1]

    # Reset index so timestamp becomes a column
    df = df.copy().reset_index()

    # Ensure the timestamp column is explicitly named
    if "timestamp" not in df.columns:
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

    # Filter numeric columns
    value_vars = [col for col in df.columns if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col])]
    if not value_vars:
        raise ValueError("No numeric sensor columns found to plot.")

    # Reshape to long format
    melted = df.melt(id_vars="timestamp", value_vars=value_vars, var_name="Sensor", value_name="Value")

    # Assign label = column name and color from SENSOR_COLORS (default to black)
    melted["Label"] = melted["Sensor"]
    melted["Color"] = melted["Sensor"].apply(lambda col: SENSOR_COLORS.get(col, "black"))

    # Create color mapping from Label to Color
    color_discrete_map = dict(zip(melted["Label"], melted["Color"]))

    # Plot
    fig = px.line(
        melted,
        x="timestamp",
        y="Value",
        color="Label",
        color_discrete_map=color_discrete_map,
    )

    # Set line width 
    fig.update_traces(line=dict(width=1))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Sensor",
        hovermode="x unified",
        title={
            'text': f"{PLACE[location_code]['name']} {start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        }
    )

    fig.show()