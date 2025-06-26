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

# schema: propertyCode: {label, deviceCategoryCode}
sensor_info = {
    "oxygen": {
        "label": "Oxygen (ml/l)",
        "deviceCategoryCode": "OXYSENSOR",
    },
    "parphotonbased": {
        "label": "PAR (µmol/m²/s)",
        "deviceCategoryCode": "radiometer",
    },
    "chlorophyll": {
        "label": "Chlorophyll (µg/l)",
        "deviceCategoryCode": "FLNTU",
    },
    "seawatertemperature": {
        "label": "Temperature (°C)",
        "deviceCategoryCode": "CTD",
    },
    "salinity": {
        "label": "Salinity (psu)",
        "deviceCategoryCode": "CTD",
    },
    "turbidityntu": {
        "label": "Turbidity (NTU)",
        "deviceCategoryCode": "FLNTU",
    },
    "conductivity": {
        "label": "Conductivity (S/m)",
        "deviceCategoryCode": "CTD"
    },
    "density": {
        "label": "Density (kg/m3)",
        "deviceCategoryCode": "CTD"
    }
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

def fetch_property_result(start: str, end: str, locationCode: str, propertyCode: str, resample: int = None) -> dict:
    """
    Makes ONC API call to get scalar data for a single propertyCode.
    """
    device_cat = sensor_info[propertyCode]["deviceCategoryCode"]

    if resample: 
        # # If OXYSENSOR, must use sensorCategoryCodes instead of propertyCode
        # if device_cat == "OXYSENSOR":
        #         params = {
        #         "locationCode": locationCode,
        #         "deviceCategoryCode": device_cat,
        #         "sensorCategoryCodes": "oxygen_corrected",
        #         "dateFrom": start,
        #         "dateTo" : end,
        #         "metadata": "minimum",
        #         "qualityControl": "clean",
        #         "resamplePeriod": resample,
        #         "resampleType": "avg"
        #         }
        # else:
            # For other devices, use propertyCode list for resampled query
            params = {
                "locationCode": locationCode,
                "deviceCategoryCode": device_cat,
                "propertyCode": propertyCode,
                "dateFrom": start,
                "dateTo" : end,
                "metadata": "minimum",
                "qualityControl": "clean",
                "resamplePeriod": resample,
                "resampleType": "avg"
                }
    else:
        # # No resampling: same distinction between OXYSENSOR and other devices
        # if device_cat == "OXYSENSOR":
        #         params = {
        #         "locationCode": locationCode,
        #         "deviceCategoryCode": device_cat,
        #         "sensorCategoryCodes": "oxygen_corrected",
        #         "dateFrom": start,
        #         "dateTo" : end,
        #         }
        # else:   
            params = {
                "locationCode": locationCode,
                "deviceCategoryCode": device_cat,
                "propertyCode": propertyCode,
                "dateFrom": start,
                "dateTo" : end,
            }
    
    # multiple types of oxygen avaliable
    if device_cat == "OXYSENSOR":
        params["sensorCategoryCodes"] = "oxygen_corrected"

        # multiple oxygen sensors avaliable at FGPD
        if locationCode == "FGPD":
            params["locationCode"] = "FGPD.O2"

    print(f"API Request: getScalarData{params}") # NOTE: for clarity

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

def get_multi_property_dataframe(start: str, end: str, locationCode: str, propertyCodes: list[str], resample: int = None) -> pd.DataFrame:
    """
    Fetches, formats, and merges multiple properties into one time-indexed DataFrame.
    """
    dfs = []

    for prop in propertyCodes:
        try:
            result = fetch_property_result(start, end, locationCode, prop, resample)

            df = result_to_dataframe(result, prop)
            print(f"Creating data frame for {prop}.\nPreview: {df.columns.tolist()}") # NOTE: for clarity
            if df is not None:
                dfs.append(df)
        except Exception as e:
            print(f"Error retrieving {prop}: {e}")

    if not dfs:
        return None

    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), dfs)
    merged_df.sort_index(inplace=True)

    print(f"Combining data frames for {propertyCodes}") # NOTE: for clarity

    return merged_df

# TODO: integrate with time as index
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
# TODO: use?
def get_priority_zorder(sensor_name: str) -> int:
    """
    Assigns a plot z-order priority to specific sensor types.
    Higher z-order means plot on top.

    Parameters:
        sensor_name (str): Name of the sensor base type (e.g., "oxygen").

    Returns:
        int: z-order value for plotting.
    """
    if "oxygen" in sensor_name:
        return 5
    elif "par" in sensor_name:
        return 4
    elif "turbidity" in sensor_name:
        return 3
    else:
        return 1

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
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8) # NOTE: plot without priority

    # Isolate times for title
    start_time = plot_df.index[0]
    end_time = plot_df.index[-1]
    # title = f"{locationCode}\n{start_time} to {end_time}"

    # NOTE: debug
    print(f"start df: {start_time}, end: {end_time}")

    # Labels and title
    ax.set_xlabel("Time", labelpad= 12)
    ax.set_ylabel("Sensor Value", labelpad=12)
    ax.set_title(f"{place[locationCode]["name"]}{' (Denoised)' if normalized else''}\n"
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
    x_step = round_data_tick_size(total_days / 5)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=x_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    # Grid and legend
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="upper right")

    plt.tight_layout()
    plt.show()


    # Grid and legend
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="upper right")

    plt.tight_layout()
    plt.show()

def plot_all_norm(df: pd.DataFrame, title: str = "Sensor Readings Over Time") -> None:
    """
    Plots each numeric sensor column in the DataFrame against time, with line priority and unit-labeled legend entries, then subplots the same data but normalized.

    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and sensor columns.
        title (str): Title of the plot.

    Returns:
        None
    """

    copy_df = df.copy()
    plot_df = copy_df # if selected normalized then do so
    normalized_df = smooth_df(plot_df) # rollowing window mean filter for outliers

    dfs = [plot_df, normalized_df]

    # TODO: confrim
    plot_df = plot_df.set_index("Time")
    normalized_df = normalized_df.set_index("Time")

    # Define figure and axes for subplots
    fig, ax = plt.subplots(figsize=(16, 20), nrows=2, ncols=1)

    # Get sensor columns
    sensor_cols = plot_df.columns
    
    for i, df in enumerate(dfs):

        for col in sensor_cols:
            # NOTE: plot with priority 
            base = col.lower().split(" (")[0] # Get base property name (remove units in parentheses)
            z_order = get_priority_zorder(base)
            ax[i].plot(plot_df.index, df[col], label=col, linewidth=1, zorder=z_order)

        # Isolate times for title
        start_time = plot_df.index[0]
        end_time = plot_df.index[-1]

        # Labels and title
        ax[i].set_xlabel("Time", labelpad=12)
        ax[i].set_ylabel("Sensor Value", labelpad=12)
        ax[i].set_title(f"{title}{': Denoised' if i % 2 !=0 else''}\n", y=1.0, pad=8)

        # Set axis limits
        ax[i].margins(x=0.01, y=0.01)
        #ax[i].set_ylim(top=ymax if ymax else None)
        #ax.set_xlim(left=start_time, right=end_time)

        # Set tick frequencies
        ymin, ymax = ax[i].get_ylim()
        y_range = ymax - ymin
        raw_ytick_step = y_range / 5  # target ~5 major ticks
        ytick_step = round_data_tick_size(raw_ytick_step)
        ax[i].yaxis.set_major_locator(MultipleLocator(ytick_step))

        x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)    
        raw_xtick_step = x_range / 5 # Target: ~5 x-axis ticks
        xtick_step = round_data_tick_size(raw_xtick_step)
        ax[i].xaxis.set_major_locator(mdates.DayLocator(interval=xtick_step))
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

        # Grid and legend
        ax[i].grid(True, which="major", linestyle="--", linewidth=0.5)
        ax[i].legend(title="Sensors", loc="upper right")

    # Add overall title
    fig.suptitle(f"{title}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold',
                 y=0.98,
                 x=0.51
                 )

    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.92, hspace=0.2)
    plt.show()

def plot_all_norm_and_scale(df: pd.DataFrame, ymax: float, title: str = "Sensor Readings Over Time") -> None:
    """
    Plots each numeric sensor column in the DataFrame against time, with line priority and unit-labeled legend entries, then subplots the same data but normalized.

    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and sensor columns.
        title (str): Title of the plot.
        ymax (float): Optional maximum y-axis value. Default shows all values.

    Returns:
        None
    """
    # Set style for plots
    # sns.set_style("darkgrid")

    renamed_df = rename_columns_with_units(df) # rename sensor columns to include units
    normalized_df = smooth_df(renamed_df) # rollowing window mean filter for outliers

    dfs = [renamed_df, normalized_df]

    # Set 'timestamp' as index (no longer a column)
    renamed_df = renamed_df.set_index("timestamp")
    normalized_df = normalized_df.set_index("timestamp")

    # Define figure and axes for subplots
    fig, ax = plt.subplots(figsize=(16, 22), nrows=3, ncols=1)

    # Get sensor columns
    sensor_cols = renamed_df.columns.tolist()
    
    # TODO: unhard code this
    for i, df in enumerate(dfs):

        for col in sensor_cols:
            # NOTE: plot with priority 
            base = col.lower().split(" (")[0] # Get base property name (remove units in parentheses)
            z_order = get_priority_zorder(base)
            ax[i].plot(renamed_df.index, df[col], label=col, linewidth=1, zorder=z_order)

        # Isolate times for title
        start_time = df["timestamp"].iloc[0]
        end_time = df["timestamp"].iloc[-1]

        # Labels and title
        ax[i].set_xlabel("Time", labelpad=12)
        ax[i].set_ylabel("Sensor Value", labelpad=12)
        ax[i].set_title(f"{title}{': Denoised' if i == 1 else''}\n", y=1.0, pad=8)

        # Set axis limits
        ax[i].margins(x=0.01, y=0.01)
        #ax[i].set_ylim(top=ymax if ymax else None)
        #ax.set_xlim(left=start_time, right=end_time)

        # Set tick frequencies
        ymin, ymax_actual = ax[i].get_ylim()
        y_range = ymax - ymin if ymax else ymax_actual - ymin
        raw_ytick_step = y_range / 5  # target ~5 major ticks
        ytick_step = round_data_tick_size(raw_ytick_step)
        ax[i].yaxis.set_major_locator(MultipleLocator(ytick_step))

        x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)    
        raw_xtick_step = x_range / 5 # Target: ~5 x-axis ticks
        xtick_step = round_data_tick_size(raw_xtick_step)
        ax[i].xaxis.set_major_locator(mdates.DayLocator(interval=xtick_step))
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

        # Grid and legend
        ax[i].grid(True, which="major", linestyle="--", linewidth=0.5)
        ax[i].legend(title="Sensors", loc="upper right")


    # Plot scaled and denoised
    for col in sensor_cols:
        # NOTE: plot with priority 
        base = col.lower().split(" (")[0] # Get base property name (remove units in parentheses)
        z_order = get_priority_zorder(base)
        ax[2].plot(renamed_df.index, normalized_df[col], label=col, linewidth=1, zorder=z_order)

    # Isolate times for title
    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]

    # Labels and title
    ax[2].set_xlabel("Time", labelpad=12)
    ax[2].set_ylabel("Sensor Value", labelpad=12)
    ax[2].set_title(f"{title}: Denoised and Scaled\n", y=1.0, pad=8)

    # Set axis limits
    ax[2].margins(x=0.01, y=0.01)
    ax[2].set_ylim(top=ymax)
    #ax.set_xlim(left=start_time, right=end_time)

    # Set tick frequencies
    ymin, ymax_actual = ax[i].get_ylim()
    y_range = ymax - ymin if ymax else ymax_actual - ymin
    raw_ytick_step = y_range / 5  # target ~5 major ticks
    ytick_step = round_data_tick_size(raw_ytick_step)
    ax[2].yaxis.set_major_locator(MultipleLocator(ytick_step))

    x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)    
    raw_xtick_step = x_range / 5 # Target: ~5 x-axis ticks
    xtick_step = round_data_tick_size(raw_xtick_step)
    ax[2].xaxis.set_major_locator(mdates.DayLocator(interval=xtick_step))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    # Grid and legend
    ax[2].grid(True, which="major", linestyle="--", linewidth=0.5)
    ax[2].legend(title="Sensors", loc="upper right")


    # Add overall title
    fig.suptitle(f"{title}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold',
                 y=0.98,
                 x=0.51
                 )

    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.93, hspace=0.3)
    plt.show()


def subplot_all_with_oxygen(df: pd.DataFrame, title: str = "Oxygen vs", normalized: bool = False) -> None:
    """
    Creates a series of subplots where each subplot shows oxygen vs another sensor over time.

    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and multiple sensor columns.
        title_prefix (str): Prefix for each subplot's title.
        ytick_freq (float): Frequency of y-ticks on the right axis (sensor).
        oxygen_ytick_freq (float): Frequency of y-ticks on the left axis (oxygen).

    Returns:
        None
    """
    renamed_df = rename_columns_with_units(df) # rename columns with units
    plot_df = smooth_df(renamed_df) if normalized else renamed_df

    # Set 'timestamp' as index
    plot_df = plot_df.set_index("timestamp")

    # Identify the oxygen column
    oxygen_col = [col for col in plot_df.columns if "oxygen" in col.lower()]
    if not oxygen_col:
        raise ValueError("No column containing 'oxygen' found.")
    oxygen_col = oxygen_col[0]

    # Get list of all other sensor columns
    sensor_cols = [col for col in plot_df.columns if col != oxygen_col]

    # Create subplots
    num_sensors = len(sensor_cols)
    fig, axs = plt.subplots(num_sensors, 1, figsize=(16, 3 * num_sensors))

    # Plot each sensor vs oxygen
    for i, sensor_col in enumerate(sensor_cols):
        ax = axs[i]
        ax2 = ax.twinx()

        ax.plot(plot_df.index, plot_df[oxygen_col], color='blue', label='Oxygen', linewidth=1)
        ax.set_ylabel('Oxygen ()', color='blue', labelpad=12)
        ax.tick_params(axis='y', labelcolor='blue')

        ax2.plot(plot_df.index, plot_df[sensor_col], color='red', label=sensor_col, linewidth=1)
        ax2.set_ylabel(sensor_col, color='red', labelpad=12)
        ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title(f"Oxygen (ml/l) vs {sensor_col}", y=1.0, pad=10)
        ax.grid(True, linestyle='--', linewidth=0.5)

    # Isolate for title etc.
    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]

    # Shared x-axis label
    axs[-1].set_xlabel("Timestamp", labelpad=15)

    # Set tick formatting to all subplots
    x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)
    raw_xtick_step = x_range / 5
    xtick_step = round_data_tick_size(raw_xtick_step)
    date_format = mdates.DateFormatter('%b %d, %Y')
    date_locator = mdates.DayLocator(interval=xtick_step)

    for ax in axs:
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_format)

    # Set axis limits and margin
    ax.margins(x=0.05,y=0.01)

    # Add overall title
    fig.suptitle(f"{title}{' (Denoised)' if normalized else''}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold',
                 y=0.98,
                 x=0.51
                 )
                 
    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.92, hspace=0.4)
    plt.show()