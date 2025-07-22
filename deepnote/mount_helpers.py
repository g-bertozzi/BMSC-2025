import onc
import pandas as pd
import plotly 
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import plotly.express as px
import cmocean
import matplotlib.colors as mcolors
from typing import List, Tuple
from functools import reduce # used for dataframes
import os

# token = os.environ["GRACE_TOKEN"]
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("ONC_TOKEN")

# Create ONC client
my_onc = onc.ONC(token)


# schema: propertyCode: {label, deviceCategoryCode, color}
sensor_info = {
    "oxygen": {
        "label": "Oxygen (ml/l)",
        "deviceCategoryCode": "OXYSENSOR",
        "color": "royalblue",
    },
    "parphotonbased": {
        "label": "PAR (µmol/m²/s)",
        "deviceCategoryCode": "radiometer",
        "color": "goldenrod",
    },
    "chlorophyll": {
        "label": "Chlorophyll (µg/l)",
        "deviceCategoryCode": "FLNTU",
        "color": "darkgreen",
    },
    "seawatertemperature": {
        "label": "Temperature (°C)",
        "deviceCategoryCode": "CTD",
        "color": "crimson",
    },
    "salinity": {
        "label": "Salinity (psu)",
        "deviceCategoryCode": "CTD",
        "color": "#b1d12e",
    },
    "turbidityntu": {
        "label": "Turbidity (NTU)",
        "deviceCategoryCode": "FLNTU",
        "color": "saddlebrown",
    },
    "density": {
        "label": "Density (kg/m3)",
        "deviceCategoryCode": "CTD",
        "color": "darkcyan",
    },
    "sigmat": {
        "label": "Sigma-t",
        "deviceCategoryCode": "CTD",
        "color": mcolors.to_hex(cmocean.cm.thermal(0.75)),
    },
}

sensor_colors = {
    "Oxygen (ml/l)": "royalblue",
    "PAR (µmol/m²/s)": "goldenrod",
    "Chlorophyll (µg/l)": "darkgreen",
    "Temperature (°C)": "crimson",
    "Salinity (psu)": "#b1d12e",
    "Turbidity (NTU)": "saddlebrown",
    "Conductivity (S/m)": "mediumorchid",
    "Density (kg/m3)": "darkcyan",
    "Sigma-t": mcolors.to_hex(cmocean.cm.thermal(0.75))
}

# schema: locationCode: {name, mountCode, castCode, mountDepth, depthThreshold}
place = {
    "FGPPN": {
        "name": "Folger Pinnacle",
        "mountCode": "FGPPN",
        "castCode":"CF341",
        "mountDepth": 23,
        # "depthThreshold": 20 # depth to be considered for deep section
    },
    "FGPD": {
        "name": "Folger Deep",
        "mountCode": "FGPD",
        "castCode": "CF340",
        "mountDepth": 90,
        # "depthThreshold": 85
    }
}


# API CALL FUNCTIONS

# NOTE: consider changing column naming wiht units and all the functions reliant on column names with units
def fetch_property_result(start: str, end: str, locationCode: str, propertyCode: str, updates: bool, resample: int = None) -> dict:
    """
    Makes ONC API call to get scalar data for a single propertyCode.

    Inputs:
        start (str): ISO 8601 UTC start timestamp (e.g., "2021-07-01T00:00:00Z")
        end (str): ISO 8601 UTC end timestamp (e.g., "2021-07-02T00:00:00Z")
        locationCode (str): ONC location code (e.g., "FGPPN")
        propertyCode (str): ONC property code (e.g., "oxygen", "seawatertemperature")
        updates (bool): If True, prints API call details for debugging
        resample (int, optional): Resample period in seconds (e.g., 60 for 1-minute averages)

    Output:
        result (dict): JSON-like dictionary from ONC API containing the requested scalar data

    """
    device_cat = sensor_info[propertyCode]["deviceCategoryCode"]

    # Default parameters
    params = {
        "locationCode": locationCode,
        "deviceCategoryCode": device_cat,
        "propertyCode": propertyCode,
        "dateFrom": start,
        "dateTo" : end,
    }

    # If resampling
    if resample: 
        params["metadata"] = "minimum"
        params["qualityControl"] = "clean"
        params["resamplePeriod"] = resample
        params["resampleType"] = "avg"

    # For oxygen: multiple types of oxygen avaliable
    if device_cat == "OXYSENSOR":
        params["sensorCategoryCodes"] = "oxygen_corrected"

        # For oxygen at FGPD: multiple oxygen sensors avaliable
        if locationCode == "FGPD":
            params["locationCode"] = "FGPD.O2"

    if updates: print(f"API Request: getScalarData{params}") # NOTE: for clarity 

    result = my_onc.getScalardata(params)
    return result

def result_to_dataframe(result: dict, propertyCode: str) -> pd.DataFrame:
    """
    Converts API result for a single propertyCode to a labeled, time-indexed DataFrame.

    Inputs:
        result (dict): JSON-like response from ONC API for a single propertyCode.
                       Expected to contain 'sensorData' with 'sampleTimes' and 'values'.
        propertyCode (str): The ONC propertyCode used to fetch the data (e.g., "oxygen").

    Output:
        pd.DataFrame: A DataFrame indexed by timestamp with a single column for the specified property.
                      Returns None if no data is found.

    """

    if not result or "sensorData" not in result or not result["sensorData"]:
        print(f"No data for {propertyCode}")
        return None

    sensor = result["sensorData"][0]
    times = sensor["data"]["sampleTimes"]
    values = sensor["data"]["values"]
    column_title = sensor_info[propertyCode]["label"] if propertyCode in sensor_info else propertyCode

    df = pd.DataFrame({
        "Time": pd.to_datetime(times),
        column_title: values
    })
    df.set_index("Time", inplace=True)
    df.sort_index(inplace=True)
    return df

def get_multi_property_dataframe(start: str, end: str, locationCode: str, propertyCodes: list[str], resample: int = None, updates: bool = False) -> pd.DataFrame:
    """
    Fetches, formats, and merges multiple properties into one time-indexed DataFrame.

    Inputs:
        start (str): Start datetime in ISO 8601 format (e.g., "2023-07-11T00:00:00.000Z").
        end (str): End datetime in ISO 8601 format.
        locationCode (str): ONC location code (e.g., "FGPPN").
        propertyCodes (list[str]): List of ONC propertyCodes to fetch (e.g., ["oxygen", "temperature"]).
        resample (int, optional): Resample period in seconds. If set, API will average data over this period.
        updates (bool, optional): If True, prints status updates during execution.

    Output:
        pd.DataFrame: A time-indexed DataFrame with one column per property.
                      Columns are labeled according to the `sensor_info` dictionary.
                      Returns None if no data is retrieved.
    """
    dfs = []

    for prop in propertyCodes:
        try:
            result = fetch_property_result(start, end, locationCode, prop, updates, resample)

            df = result_to_dataframe(result, prop)
            #if updates: print(f"Creating data frame for {prop}.\nPreview: {df.columns.tolist()}") # NOTE: for clarity
            if df is not None:
                dfs.append(df)
        except Exception as e:
            print(f"Error retrieving {prop}: {e}")

    if not dfs:
        return None

    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), dfs)
    merged_df.sort_index(inplace=True)

    if updates: print(f"Generated combined data frame for {propertyCodes} at {locationCode}.") # NOTE: for clarity TODO: make actually df preview

    return merged_df


# PROCESSING FUNCTIONS

def smooth_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rolling mean smoothing and rolling z-score outlier filtering 
    to all data (i.e. numeric) columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'timestamp' and sensor data.

    Returns:
        pd.DataFrame: Smoothed and filtered DataFrame (same shape).
    """
    import numpy as np

    window = 12  # Size of the rolling window for smoothing
    z_thresh = 3.0  # Z-score threshold for outlier detection

    smoothed_df = df.copy()  # Work on a copy to preserve the original
    numeric_cols = df.columns

    # Apply rolling smoothing and z-score filtering to each numeric column
    for col in numeric_cols:
        # Compute rolling mean and std deviation using centered window
        roll_mean = smoothed_df[col].rolling(window=window, center=True).mean()
        roll_std = smoothed_df[col].rolling(window=window, center=True).std()

        # Calculate z-scores for detecting outliers
        z_scores = (smoothed_df[col] - roll_mean) / roll_std

        # Replace values with rolling mean where the z-score is within the threshold; otherwise set to NaN
        smoothed_df[col] = roll_mean.where(z_scores.abs() < z_thresh, np.nan)

    return smoothed_df

# PLOTTING FUNCTIONS

# TODO: make consistent x axis labels for limits
def plot_dataframe(df: pd.DataFrame, locationCode: str, start: pd.Timestamp = None, end: pd.Timestamp = None, ymax: float = None, normalized: bool = False) -> None:
    """
    Plots each numeric sensor column in the DataFrame against time, 
    with line priority and unit-labeled legend entries.
    Option to normalize.

    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and sensor columns.
        title (str): Title of the plot.
        ymax (float): Optional maximum y-axis value. Default shows all values.

    Returns:
        None
    """

    copy_df = df.copy()
    plot_df = smooth_df(copy_df) if normalized else copy_df # if selected normalized then do so

    # Define figure and axes for subplots
    fig, ax = plt.subplots(figsize=(16, 9))

    # Get sensor columns
    for col in plot_df.columns:
        color = sensor_colors[col] or "black"  # default color
        z_order = 10 if col == "Oxygen (ml/l)" else 1  # default base layer
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, color=color, zorder=z_order)

    # Isolate times for title
    start_time = start or plot_df.index[0]
    end_time = end or plot_df.index[-1]

    # ax.plot([start_time, start_time], [0, 100], color='purple', linestyle='--', linewidth=1) # NOTE: debug
    # print(f"start df: {start_time}, end: {end_time}") # NOTE: debug

    # Labels and title
    ax.set_xlabel("Date", labelpad=12)
    ax.set_ylabel("Sensor Value", labelpad=12)
    ax.set_title(f"{place[locationCode]['name']}{' (Denoised)' if normalized else ''}\n"
             f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
            fontweight='bold',
            pad=15)

    # X-axis formatting
    time_range = end_time - start_time # Compute dynamic padding
    x_padding = time_range * 0.03  # 3% padding on each side
    ax.set_xlim(start_time - x_padding, end_time + x_padding)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    locator.intervald[mdates.MONTHLY] = [2]  # set for every 2 months # TODO: make this dynamic depending on time range
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    xticks = ax.get_xticks()
    xtick_dates = [mdates.num2date(tick) for tick in xticks]
    # print(xtick_dates) # NOTE: debugging
    ax.set_xticks(xticks)
    ax.set_xticklabels([dt.strftime("%b %d, %Y") for dt in xtick_dates])

    # Y-axis formatting
    y_min = 0 #plot_df.min().min()
    y_max = plot_df.max().max()
    y_padding = (y_max - y_min) * 0.01  # 1% vertical padding
    ax.set_ylim(bottom=y_min - y_padding, top=y_max + y_padding if ymax else None)

    # Grid and legend
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="upper right")

    # Add line to show low ox level
    ax.plot([start_time, end_time], [3, 3], color='red', linestyle='--', linewidth=1, label='Low Oxygen Threshold (3 ml/l)')
    ax.legend(loc="upper right")
        
    plt.tight_layout()
    plt.show()

# TODO: make y axis ticks consistent in terms of min and max labels, i.e. num ticks
# TODO: make y axis bottom padding
def subplot_all_with_oxygen(df: pd.DataFrame, locationCode: str, normalized: bool = False) -> None:
    """
    Creates a series of subplots where each subplot shows oxygen vs another sensor over time.
    Applies consistent colors and a hypoxia threshold line.
    """
    copy_df = df.copy()
    plot_df = smooth_df(copy_df) if normalized else copy_df

    # Identify oxygen column
    oxygen_col = [col for col in plot_df.columns if "oxygen" in col.lower()]
    if not oxygen_col:
        raise ValueError("No column containing 'oxygen' found.")
    oxygen_col = oxygen_col[0]

    # All other columns
    sensor_cols = [col for col in plot_df.columns if col != oxygen_col]
    num_sensors = len(sensor_cols)

    fig, axs = plt.subplots(num_sensors, 1, figsize=(16, 3.2 * num_sensors))

    if num_sensors == 1:
        axs = [axs]  # ensure it's always iterable

    for i, sensor_col in enumerate(sensor_cols):
        ax = axs[i]
        ax2 = ax.twinx()

        # Oxygen color and label
        oxygen_meta = next((meta for meta in sensor_info.values() if meta["label"] in oxygen_col), {})
        oxygen_color = "royalblue"
        oxygen_label = oxygen_meta.get("label", "Oxygen")

        # Sensor color and label
        sensor_meta = next((meta for meta in sensor_info.values() if meta["label"] in sensor_col), {})
        sensor_color = sensor_meta.get("color", "red")
        sensor_label = sensor_meta.get("label", sensor_col)

        # Plot oxygen
        ax.plot(plot_df.index, plot_df[oxygen_col], color=oxygen_color, label=oxygen_label, linewidth=0.8, zorder=10)
        ax.set_ylabel(oxygen_label, color=oxygen_color, labelpad=12)
        ax.tick_params(axis='y', labelcolor=oxygen_color)
        ax.set_ylim(bottom=0) # Ensure oxygen starts at 0

        # Plot sensor
        ax2.plot(plot_df.index, plot_df[sensor_col], color=sensor_color, label=sensor_label, linewidth=0.8, zorder=1)
        ax2.set_ylabel(sensor_label, color=sensor_color, labelpad=12)
        ax2.tick_params(axis='y', labelcolor=sensor_color)

       # Align y=0 if other sensor values are low
        if plot_df[sensor_col].min() <= 1:
            ax2.set_ylim(bottom=0)
        

        # TODO: make y axis ticks consistent in terms of min and max labels, i.e. num ticks
        from matplotlib.ticker import MaxNLocator

        # Get number of ticks on the left (oxygen) y-axis
        num_oxy_ticks = len(ax.get_yticks())

        # Force right y-axis (sensor) to use same number of ticks
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=num_oxy_ticks, prune=None))

        # TODO: if you want line then add legend
        # ax.axhline(y=3, color='red', linestyle='--', linewidth=1, label="Low Oxygen Threshold (3 ml/l)")

        # Title per subplot
        ax.set_title(f"{oxygen_label} vs {sensor_label}", y=1.0, pad=10, fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5)

    # Time range
    start_time = df.index[0]
    end_time = df.index[-1]

    # Shared x-label
    axs[-1].set_xlabel("Date", labelpad=15)

    # X-axis formatting
    # Compute dynamic padding (e.g. 3% of full time range)
    time_range = end_time - start_time
    padding = time_range * 0.03  # 3% padding on each side

    locator = mdates.AutoDateLocator(minticks=6, maxticks=6)
    locator.intervald[mdates.MONTHLY] = [2]  # set for every 2 months

    for ax in axs:
        ax.set_xlim(start_time - padding, end_time + padding) # Make sure all axes span same range
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    # Layout and title
    fig.suptitle(f"{place[locationCode]['name']}{' (Smoothed)' if normalized else ''}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold', 
                 y=0.98, 
                 x=0.51)

    plt.subplots_adjust(top=0.89, hspace=0.4)
    plt.show()

# NOTE: this is set for x axis ticks every two months
def subplot_all_with_time(df: pd.DataFrame, locationCode: str, start: pd.Timestamp = None, end: pd.Timestamp = None) -> None:
    """
    Subplots all properties in a data frame against time.
    """
    start_time = start or df.index[0]
    end_time = end or df.index[-1]
    sensor_cols = df.columns.to_list()

    fig, axes = plt.subplots(figsize=(16, len(sensor_cols)*4), nrows=len(sensor_cols), ncols=1)

    if len(sensor_cols) == 1:
        axes = [axes]  # ensure iterable

    for i, col in enumerate(sensor_cols):
        color = sensor_colors[col] or "black"
        z_order = 0.8
        label = col

        ax = axes[i]
        ax.plot(df.index, df[col], color=color, linewidth=0.8, zorder=z_order, label=label)

        ax.set_title(f"{place[locationCode]['name']} - {label}")
        ax.set_ylabel(label, labelpad=13)
        ax.set_xlabel("Date", labelpad=13)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

        # Set consistent x-ticks across all subplots

        # Compute dynamic padding (e.g. 3% of full time range)
        time_range = end_time - start_time
        padding = time_range * 0.03  # 3% padding on each side
        ax.set_xlim(start_time - padding, end_time + padding) # Make sure all axes span same range

        locator = mdates.AutoDateLocator(minticks=6, maxticks=6)
        locator.intervald[mdates.MONTHLY] = [2]  # set for every 2 months
        locator.prune = None  
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'),)

        if "oxygen" in col.lower():
            #ax.axhline(y=3, color='red', linestyle='--', label="Hypoxia Threshold")
            ax.plot([start_time, end_time], [3, 3], color='red', linestyle='--', linewidth=1, label='Low Oxygen Threshold (3 ml/l)')
            ax.legend(loc="upper right")
        

    # match x-ticks across all subplots
    shared_xticks = axes[0].get_xticks()
    for ax in axes[1:]:
        ax.set_xticks(shared_xticks)

    fig.suptitle(
        f"{place[locationCode]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        y=0.97,
        x=0.51
    )

    plt.subplots_adjust(top=0.93, hspace=0.4)
    plt.show()

def compare_sensor_subplots(df1: pd.DataFrame, df2: pd.DataFrame, sensor_cols: list[str], locationCode1: str, locationCode2: str) -> None:
    """
    Creates subplots comparing the same sensor parameters from two locations using shared styling
    from the global sensor_info dictionary.

    Dataframes must be from the same time period and have the same index (timestamps). TODO: add check for this

    Parameters:
        df1 (pd.DataFrame): First location's data (timestamp as index).
        df2 (pd.DataFrame): Second location's data (timestamp as index).
        sensor_cols (list[str]): List of shared propertyCodes to plot.
        locationCode1 (str): Location code for first dataset (e.g., 'FGPPN').
        locationCode2 (str): Location code for second dataset (e.g., 'FDPP').

    Returns:
        None
    """
    # Check time period in each dataframe (i.e. for missing data)
    min_time = min(df1.index[0], df2.index[0])
    max_time = max(df1.index[-1], df2.index[-1])

    # Define figure and axes for subplots
    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(14, 3.5 * len(sensor_cols)))

    if len(sensor_cols) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, col in zip(axes, sensor_cols):

        label = col #subplot title

        # color options: seagreen & tomato, royalblue & crimson, teal & darkorange, Slateblue & firebrick. steelblue & indianred
        ax.plot(df1.index, df1[col], label=f"{locationCode1}", linewidth=0.8, color="slateblue")
        ax.plot(df2.index, df2[col], label=f"{locationCode2}", linewidth=0.8, color="firebrick")

        ax.set_title(f"{label}", fontsize=12)
        ax.set_ylabel(col, labelpad=12)
        ax.set_xlabel("Time", labelpad=12)
        ax.grid(True)
        ax.legend()

        # Compute dynamic padding (e.g. 3% of full time range)
        time_range = max_time - min_time
        padding = time_range * 0.03  # 3% padding on each side
        ax.set_xlim(min_time - padding, max_time + padding) # Make sure all axes span same range

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'),)

    start_time = df1.index[0]
    end_time = df1.index[-1]

    fig.suptitle(
        f"{place[locationCode1]['name']} vs {place[locationCode2]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        y=0.97,
        x=0.51
    )

    plt.subplots_adjust(top=0.89, hspace=0.45)
    plt.show()

def plot_dataframe_plotly(df: pd.DataFrame, locationCode: str) -> None:
    """
    Creates an interactive Plotly line chart for all numeric sensor columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index and sensor columns.
        locationCode (str): Optional identifier used in the title.

    Returns:
        None (displays interactive plot inline)
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

    # Assign label = column name and color from global sensor_colors (default to black)
    melted["Label"] = melted["Sensor"]
    melted["Color"] = melted["Sensor"].apply(lambda col: sensor_colors.get(col, "black"))

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
        title ={
            'text': f"{place[locationCode]['name']} {start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
            # 'x': 0.5,  # Center the title
            'xanchor': 'center',
            'yanchor': 'top'
        }  
    )

    fig.show()


def plot_min_max_normalized(df: pd.DataFrame, locationCode: str) -> None:
    """
    Plots all sensor parameters normalized to 0–1 on a single time-series plot.
    Each line is colored and labeled using sensor_info.
    """
    start_time = df.index[0]
    end_time = df.index[-1]
    sensor_cols = df.columns.to_list()

    # Normalize each column (0–1)
    norm_df = df.copy()
    for col in sensor_cols:
        col_min = norm_df[col].min()
        col_max = norm_df[col].max()
        if col_max != col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.5  # fallback for constant column

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot each normalized series
    for col in sensor_cols:
        color = sensor_colors[col] or "black"
        label = col
        # for prop, meta in sensor_info.items():
        #     if meta["label"] in col:
        #         color = meta.get("color", "black")
        #         label = meta["label"]
        #         break

        ax.plot(norm_df.index, norm_df[col], label=label, color=color, linewidth=1)

    # Formatting
    ax.set_title(
         f"{place[locationCode]['name']} —  Min Max Normalized\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontsize=14,
        fontweight='bold'
    )
    ax.set_ylabel("Normalized Value (0–1)")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Date formatting
    locator = mdates.MonthLocator(interval=2)
    formatter = mdates.DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend()
    plt.tight_layout()
    plt.show()

# TODO: set y padding
def plot_with_twin_y_axis_for_outlier(df: pd.DataFrame, locationCode: str) -> None:
    """
    Plots time series data using a twin y-axis. The highest-magnitude parameter is plotted on the
    right y-axis using its defined color and label; all others are on the left.

    Parameters:
        df (pd.DataFrame): Time-indexed DataFrame of numeric sensor columns.
        locationCode (str): Optional label to include in title.
        title (str): Optional custom title.
        sensor_info (dict): Maps propertyCode -> {label, color}.

    Returns:
        None
    """
    # For title
    start_time = df.index[0]
    end_time = df.index[-1]

    # Identify the highest-magnitude column
    mean_mags = df.abs().mean()
    high_param = mean_mags.idxmax()
    low_params = [col for col in df.columns if col != high_param]

    # Set up plot
    fig, ax_left = plt.subplots(figsize=(16, 6))
    ax_right = ax_left.twinx()

    # Plot left y-axis (all but the high param)
    for col in low_params:
        color = sensor_colors[col] or "black"
        label = col
        z_order = 10 if col == "Oxygen (ml/l)" else 1  # default base layer

        ax_left.plot(df.index, df[col], label=label, color=color, linewidth=0.8, zorder=z_order)

    # Plot right y-axis (high-magnitude param)
    color = sensor_colors[high_param] or "black"
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
    ax_left.set_title(f"{place[locationCode]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}", fontweight='bold')

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
