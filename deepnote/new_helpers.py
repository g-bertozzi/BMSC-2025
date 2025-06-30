import onc
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

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
        "color": "orange",
    },
    "turbidityntu": {
        "label": "Turbidity (NTU)",
        "deviceCategoryCode": "FLNTU",
        "color": "saddlebrown",
    },
    "conductivity": {
        "label": "Conductivity (S/m)",
        "deviceCategoryCode": "CTD",
        "color": "mediumorchid",
    },
    "density": {
        "label": "Density (kg/m3)",
        "deviceCategoryCode": "CTD",
        "color": "darkcyan",
    },
}


# TODO: make cast plot dunctions consider within 10m of mount depth to be "deep"
"""
Global metadata dictionary for each location - e.g. FGPD, FGPPN

Useful for accessing name for titles, mount vs cast codes, mount depths, and range considered deep for casts.

SCHEMA: locationCode: {name, mountCode, castCode, mountDepth, depthThreshold}
"""
place = {
    "FGPPN": {
        "name": "Folger Pinnacle",
        "mountCode": "FGPPN",
        "castCode":"CF341",
        "mountDepth": 23,
        "depthThreshold": 20 # depth to be considered for deep section
    },
    "FGPD": {
        "name": "Folger Deep",
        "mountCode": "FGPD",
        "castCode": "CF340",
        "mountDepth": 90,
        "depthThreshold": 85
    }
}

def fetch_property_result(start: str, end: str, locationCode: str, propertyCode: str, updates: bool, resample: int = None) -> dict:
    """
    Makes ONC API call to get scalar data for a single propertyCode.
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
    Converts ONC result for a single propertyCode to a labeled, time-indexed DataFrame.
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
    numeric_cols = df.columns #[col for col in df.columns if col != "timestamp"] # Select numeric columns only TODO: confirm

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

def round_data_tick_size(value):
    """
    Safely round a step size to a clean value: 1, 2, 5, or 10 × 10^n
    """
    import math
    if value <= 0:
        return 1  # fallback

    for base in [1, 2, 5, 10]:
        if value <= base:
            return base

    magnitude = 10 ** math.floor(math.log10(value))
    residual = value / magnitude

    if residual <= 1.5:
        return 1 * magnitude
    elif residual <= 3:
        return 2 * magnitude
    elif residual <= 7:
        return 5 * magnitude
    else:
        return 10 * magnitude

# TODO: get cast and mount functions

# TODO: make consistent x axis labels for limits
def plot_dataframe(df: pd.DataFrame, locationCode: str, ymax: float = None, normalized: bool = False) -> None:
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
    fig, ax = plt.subplots(figsize=(16, 10))

    # Get sensor columns
    for col in plot_df.columns:

        #ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8) # NOTE: plot without priority

        # Dynamically match the label to a propertyCode
        color = "black"  # default color
        z_order = 1  # default base layer

        for prop, meta in sensor_info.items():
            if meta["label"] in col:
                color = meta.get("color", "black")
                if prop == "oxygen":
                    z_order = 10  # ensure oxygen is on top
                break
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, color=color, zorder=z_order)


    # Isolate times for title
    start_time = plot_df.index[0]
    end_time = plot_df.index[-1]
    # title = f"{locationCode}\n{start_time} to {end_time}"

    # NOTE: debug
    print(f"start df: {start_time}, end: {end_time}")

    # Labels and title
    ax.set_xlabel("Time", labelpad= 12)
    ax.set_ylabel("Sensor Value", labelpad=12)
    ax.set_title(f"{place[locationCode]['name']}{' (Denoised)' if normalized else ''}\n"
             f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
            fontweight='bold',
            pad=15
            )

    # Set axis limits and margin
    ax.margins(x=0.01,y=0.01)
    ax.set_ylim(top=ymax if ymax else None)
    #ax.set_xlim(left=start_time, right=end_time)

    # --- Y-axis ticks ---
    # y_min, y_max = ax.get_ylim()
    # y_step = round_data_tick_size((y_max - y_min) / 15)
    # ax.yaxis.set_major_locator(MultipleLocator(y_step))

    # --- X-axis ticks ---
    total_days = (end_time - start_time).days
    x_step = round_data_tick_size(total_days / 10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    # Grid and legend
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="upper right")

    # Add line to show low ox level
    ax.axhline(y=3, color='red', linestyle='--', linewidth=1, label='Low Oxygen Threshold (3 ml/l)')
    ax.legend(loc="upper right")
        
    plt.tight_layout()
    plt.show()

def plot_dataframe_norm(df: pd.DataFrame, locationCode: str) -> None:
    """
    Plots each sensor column in the DataFrame twice: raw and smoothed (normalized),
    in vertically stacked subplots.

    Parameters:
        df (pd.DataFrame): DataFrame indexed by time, with labeled sensor columns.
        locationCode (str): ONC location code, used for plot titles.

    Returns:
        None
    """
    copy_df = df.copy()
    smoothed_df = smooth_df(copy_df)

    dfs = [copy_df, smoothed_df]
    titles = ["", " - Smoothed"]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12))

    for i, ax in enumerate(axes):
        added_hypoxia_label = False
        plot_df = dfs[i]

        for col in plot_df.columns:

            # Dynamically match the label to a propertyCode
            color = "black"  # default color
            z_order = 1  # default base layer

            for prop, meta in sensor_info.items():
                if meta["label"] in col:
                    color = meta.get("color", "black")
                    if prop == "oxygen":
                        z_order = 10  # ensure oxygen is on top
                    break
            ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, color=color, zorder=z_order)

        # Plot hypoxia line on both, label only once
        ax.axhline(
            y=3,
            color='red',
            linestyle='--',
            linewidth=1,
            label='Low Oxygen Threshold (3 ml/l)' if not added_hypoxia_label else None
        )
        added_hypoxia_label = True  # Prevent further labels
                    
        # Time range for title and axis
        start_time = plot_df.index[0]
        end_time = plot_df.index[-1]

        # Title and labels
        ax.set_title(f"{place[locationCode]['name']}{titles[i]}")
        ax.set_ylabel("Sensor Value", labelpad= 12)
        ax.set_xlabel("Time", labelpad= 12)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right")

        # Y-axis tick spacing
        # ymin, ymax = ax.get_ylim()
        # y_range = ymax - ymin
        # ytick_step = round_data_tick_size(y_range / 10)
        # ax.yaxis.set_major_locator(MultipleLocator(ytick_step))

        # X-axis tick spacing (days)
        x_range_days = (end_time - start_time).days
        xtick_step = max(1, round_data_tick_size(x_range_days / 6))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=xtick_step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    # X-axis label on bottom plot only
    axes[-1].set_xlabel("Date")

    # Figure title
    fig.suptitle(f"{place[locationCode]['name']}\n {start_time.strftime('%b %d, %Y')} to {end_time.strftime('%b %d, %Y')}",
                 fontweight='bold', 
                 x=0.51,
                 y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.2)
    plt.show()

def plot_dataframe_norm_and_scale(df: pd.DataFrame, locationCode: str, ymax: float = None) -> None:
    """
    Plots raw, smoothed, and normalized+smoothed sensor values over time in 3 vertically stacked subplots.

    Parameters:
        df (pd.DataFrame): DataFrame indexed by datetime with sensor value columns.
        ymax (float): Optional upper Y-axis limit.
        title (str): Plot title.
    """
    copy_df = df.copy()
    smoothed_df = smooth_df(copy_df)

    dfs = [copy_df, smoothed_df, smoothed_df]
    subtitles = ["", " - Smoothed", " - Smoothed (Zoomed In? )"]


    fig, ax = plt.subplots(figsize=(16, 22), nrows=3, ncols=1)

    sensor_cols = df.columns.tolist()

    for i, df_i in enumerate(dfs):
        added_hypoxia_label = False
        for col in sensor_cols:

        # Dynamically match the label to a propertyCode
            color = "black"  # default color
            z_order = 1  # default base layer

            for prop, meta in sensor_info.items():
                if meta["label"] in col:
                    color = meta.get("color", "black")
                    if prop == "oxygen":
                        z_order = 10  # ensure oxygen is on top
                    break
            ax[i].plot(df_i.index, df_i[col], label=col, linewidth=1, color=color, zorder=z_order)

        # Plot hypoxia line on both, label only once
        ax[i].axhline(
            y=3,
            color='red',
            linestyle='--',
            linewidth=1,
            label='Low Oxygen Threshold (3 ml/l)' if not added_hypoxia_label else None
        )
        added_hypoxia_label = True  # Prevent further labels
            
        # Time range
        start_time = df_i.index[0]
        end_time = df_i.index[-1]

        # Labels and title
        ax[i].set_title(f"{place[locationCode]['name']} {subtitles[i]}", y=1.0, pad=8)
        ax[i].set_xlabel("Time", labelpad=13)
        ax[i].set_ylabel("Sensor Value", labelpad=13)

        # Axis limits
        ax[i].margins(x=0.01, y=0.01)
        if ymax and i == 2:  # Only apply ymax to the third (clipped) plot
            ax[i].set_ylim(top=ymax)


        # Y-axis ticks
        # ymin, ymax_actual = ax[i].get_ylim()
        # y_range = (ymax or ymax_actual) - ymin
        # ytick_step = round_data_tick_size(y_range / 6)
        # ax[i].yaxis.set_major_locator(MultipleLocator(ytick_step))

        # X-axis ticks
        x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)
        xtick_step = max(1, round_data_tick_size(x_range / 6))
        ax[i].xaxis.set_major_locator(mdates.DayLocator(interval=xtick_step))
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

        # Grid and legend
        ax[i].grid(True, linestyle="--", linewidth=0.5)
        ax[i].legend(loc="upper right", title="Sensors")

    # Add overall title
    fig.suptitle(f"{place[locationCode]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold', 
                 x=0.51,
                 y=0.98)

    # Adjust layout
    plt.subplots_adjust(top=0.94, hspace=0.2)
    plt.show()

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

        # Plot both
        ax.plot(plot_df.index, plot_df[oxygen_col], color=oxygen_color, label=oxygen_label, linewidth=0.8, zorder=10)
        ax.set_ylabel(oxygen_label, color=oxygen_color, labelpad=12)
        ax.tick_params(axis='y', labelcolor=oxygen_color)

        ax2.plot(plot_df.index, plot_df[sensor_col], color=sensor_color, label=sensor_label, linewidth=0.8, zorder=1)
        ax2.set_ylabel(sensor_label, color=sensor_color, labelpad=12)
        ax2.tick_params(axis='y', labelcolor=sensor_color)

        # Hypoxia threshold line
        ax.axhline(y=3, color='red', linestyle='--', linewidth=1, label="Low Oxygen Threshold (3 ml/l)")

        # Title per subplot
        ax.set_title(f"{oxygen_label} vs {sensor_label}", y=1.0, pad=10)
        ax.grid(True, linestyle='--', linewidth=0.5)

    # Time range
    start_time = df.index[0]
    end_time = df.index[-1]

    # Shared x-label
    axs[-1].set_xlabel("Timestamp", labelpad=15)

    # X-axis formatting
    x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)
    xtick_step = round_data_tick_size(x_range / 5)
    locator = mdates.DayLocator(interval=xtick_step)
    formatter = mdates.DateFormatter('%b %d, %Y')

    for ax in axs:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # Layout and title
    fig.suptitle(f"{place[locationCode]['name']}{' (Smoothed)' if normalized else ''}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold', 
                 y=0.98, 
                 x=0.51)

    plt.subplots_adjust(top=0.89, hspace=0.4)
    plt.show()
