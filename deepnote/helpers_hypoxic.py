import pandas as pd
import onc
import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns

from functools import reduce

# token = os.environ["TRICY_TOKEN"]
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("ONC_TOKEN")

# Create ONC client
my_onc = onc.ONC(token)

# NOTE: HARDCODED - global dataframe of sensor info
# schema: propertyCode, name, unit, deviceCategoryCode
sensor_info = pd.DataFrame([
    {"propertyCode": "oxygen", "name": "Dissolved Oxygen", "unit": "ml/l", "deviceCategoryCode": "OXYSENSOR"},
    {"propertyCode": "parphotonbased", "name": "PAR", "unit": "µmol/m²/s", "deviceCategoryCode": "radiometer"},
    {"propertyCode": "chlorophyll", "name": "Chlorophyll", "unit": "µg/l", "deviceCategoryCode": "FLNTU"},
    {"propertyCode": "seawatertemperature", "name": "Temperature", "unit": "°C", "deviceCategoryCode": "CTD"},
    {"propertyCode": "salinity", "name": "Salinity", "unit": "psu", "deviceCategoryCode": "CTD"},
    {"propertyCode": "turbidityntu", "name": "Turbidity", "unit": "NTU", "deviceCategoryCode": "FLNTU"},
])


def find_properties_by_location(locationCode: str):
    """
    Retrieves and prints a list of all sensor properties available at a given ONC location.
    For each property, includes its display name, property code, and whether it has associated device data.

    Parameters:
        locationCode (str): The ONC location code to query (e.g., "FGPPN").

    Returns:
        None: Prints a DataFrame of available properties to stdout.
    """
    params = {
        "locationCode": locationCode,
        #"deviceCategoryCode" : "CTD" 
    }

    result = my_onc.getProperties(params)
    extracted = []

    for entry in result:
        # Defensive check: make sure these keys exist
        name = entry.get("propertyName", "")
        code = entry.get("propertyCode", "")
        has_data = entry.get("hasDeviceData", False)

        # Optionally: filter out properties that aren't actually measured
        if name and code:
            extracted.append({
                "propertyName": name,
                "propertyCode": code,
                "hasDeviceData": has_data
            })
    
    df = pd.DataFrame(extracted)
    print(df)

def get_dataframe(start: str, end: str, props: list[str]) -> list[pd.DataFrame]:
    """
    Retrieves ONC scalar sensor data for a list of propertyCodes (e.g., 'oxygen', 'chlorophyll', etc.)
    over a specified time range at a fixed location (FGPPN). Each property is fetched from the appropriate
    deviceCategoryCode as defined in the global `sensor_info` table.

    Parameters:
        start (str): ISO 8601 formatted start time (e.g., "2021-07-15T12:00:00.000Z").
        end (str): ISO 8601 formatted end time (e.g., "2021-07-15T12:30:00.000Z").
        props (list[str]): A list of sensor property codes to fetch (e.g., ["oxygen", "chlorophyll"]).

    Returns:
        list[pd.DataFrame]: A list of individual DataFrames, each containing timestamped data
                            for one sensor property. Schema of each DataFrame: [timestamp, {property}]
    """
    dfs = []

    # iterate through each property wanted and access it's deviceCategory via global dataframe
    for prop in props: 
        device_cat = sensor_info.set_index("propertyCode").at[prop, "deviceCategoryCode"]
        
        # call helper to fetch data for each property
        df = get_property(start=start, end=end, locationCode="FGPPN", deviceCategoryCode=device_cat, propertyCode=prop)

        if df is not None:
            dfs.append(df)

    # NOTE: ERROR HANDLE if no data fetched
    if not dfs:
        print(f"No data returned for {props} from {start} to {end}")
        return
    
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs) # merge all the dataframes by joining on time
    merged_df.sort_values("timestamp", inplace=True) 

    # print(f"{props} returned {len(merged_df)} rows") # NOTE: debugging
    return merged_df

def get_property(start: str, end: str, locationCode: str, deviceCategoryCode: str, propertyCode: str = None) -> pd.DataFrame:
    """
    Fetches scalar data for specified single property from specified location and device category and time window. 
    Returns a  DataFrame with timestamps and sensor values.

    Parameters:
        start (str): Start date in ISO 8601 format (e.g., "2023-07-11T17:00:00.000Z").
        end (str): End date in ISO 8601 format (e.g., "2023-07-11T22:30:00.000Z").
        locationCode (str): ONC location code (e.g., "CF341").
        deviceCategoryCode (str): ONC device category (e.g., "CTD").
        propertyCode(str): Comma-separated sensor types to fetch (e.g., "depth,temperature").

    Returns:
        pd.DataFrame: DataFrame containing sensor values with a timestamp index.
                    schema: timestamp: datetime obj, {prop}: int
    """

    # TODO: fix oxygen to filter for sensorCategoryCode = "oxygen_corrected"
    #if propertyCode == "oxygen":

    # TODO: optimize - presently doing ruqest fro every porpertyCode, but I think I could limit this ß

    params = {
        "locationCode": locationCode,
        "deviceCategoryCode": deviceCategoryCode,
        "propertyCode": propertyCode,
        "dateFrom": start,
        "dateTo" : end,
        "metadata": "minimum",
        "qualityControl": "clean",
        "resamplePeriod": 900,
        "resampleType": "avg"
        }
    
    # print(f"Requesting {propertyCode} from {start} to {end}") # NOTE: debugging
     
    # JSON response from ONC
    result = my_onc.getScalardata(params)

    # NOTE: ERROR HANDLE if there is no data returned
    if not result or "sensorData" not in result or result["sensorData"] is None or len(result["sensorData"]) == 0:
        print(f"No data returned for devices in {deviceCategoryCode} at {locationCode} between {start} and {end}.")
        return
        
    else:
        sensor = result["sensorData"][0] # isolate sensor
        
        # extract sensors data fields
        prop = sensor["propertyCode"]
        times = sensor["data"]["sampleTimes"]
        values = sensor["data"]["values"]

        # populate dataframe
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times), # convert strings to datetime objects
            prop: values
        })

        # print(f"{prop} min time: {df["timestamp"].min()}") # NOTE: debugging
        # print(f"{prop} max time: {df["timestamp"].max()}") # NOTE: debugging

    df.sort_values("timestamp", inplace=True)
    
    #print(df.head()) # NOTE: debugging
    return df

def smooth_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rolling mean smoothing and rolling z-score outlier filtering 
    to all data (i.e. numeric) columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'timestamp' and sensor data.

    Returns:
        pd.DataFrame: Smoothed and filtered DataFrame (same shape).
    """
    window = 12  # Size of the rolling window for smoothing
    z_thresh = 3.0  # Z-score threshold for outlier detection

    smoothed_df = df.copy()  # Work on a copy to preserve the original
    numeric_cols = [col for col in df.columns if col != "timestamp"] # Select numeric columns only

    # Apply rolling smoothing and z-score filtering to each numeric column
    for col in numeric_cols:
        # Compute rolling mean and std deviation using centered window
        roll_mean = smoothed_df[col].rolling(window=window, center=True).mean()
        roll_std = smoothed_df[col].rolling(window=window, center=True).std()

        # Calculate z-scores for detecting outliers
        z_scores = (smoothed_df[col] - roll_mean) / roll_std

        # Replace values with rolling mean where the z-score is within the threshold; otherwise set to NaN
        smoothed_df[col] = roll_mean.where(z_scores.abs() < z_thresh, np.nan)

    # Fill in missing values (NaNs) by interpolating between valid data points
    #smoothed_df[numeric_cols] = smoothed_df[numeric_cols].interpolate()

    return smoothed_df

def rename_columns_with_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames sensor data columns in a DataFrame by appending their display name and unit of measure,
    using the global `sensor_info` table for lookup. The 'timestamp' column is left unchanged.

    Parameters:
        df (pd.DataFrame): A DataFrame containing time-series sensor data. Assumes columns include
                           propertyCodes such as 'oxygen', 'salinity', etc.

    Returns:
        pd.DataFrame: A new DataFrame with sensor columns renamed to include human-readable names and units.
                      Example: 'oxygen' -> 'Dissolved Oxygen (ml/l)'
    """
    renamed = {}

    # iterate through each column
    for col in df.columns:
        if col == "timestamp":
            continue
        
        # search for current col
        match = sensor_info[sensor_info["propertyCode"] == col]
        if not match.empty:
            name = match.iloc[0]["name"]
            unit = match.iloc[0]["unit"]
            renamed[col] = f"{name} ({unit})"

    return df.rename(columns=renamed)

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
    Round a numeric step size to a 'clean' number: 1, 2, 5, or 10 × 10^n
    """
    import math
    magnitude = 10 ** math.floor(math.log10(value))
    residual = value / magnitude

    if residual < 1.5:
        nice = 1
    elif residual < 3:
        nice = 2
    elif residual < 7:
        nice = 5
    else:
        nice = 10

    return nice * magnitude

def plot_all_normalization(df: pd.DataFrame, title: str = "Sensor Readings Over Time", ymax: float = None) -> None:
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
    scaled_dfs = [renamed_df, normalized_df]

    # Set 'timestamp' as index (no longer a column)
    renamed_df = renamed_df.set_index("timestamp")
    normalized_df = normalized_df.set_index("timestamp")

    # Define figure and axes for subplots
    fig, ax = plt.subplots(figsize=(16, 26), nrows=4, ncols=1)

    # Get sensor columns
    sensor_cols = renamed_df.columns.tolist()
    
    # TODO: unhard code this

    ####
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
        ax[i].set_title(f"{'Data'}{': Denoised' if i % 2 !=0 else': ReSampling Period = 900s'}\n", y=1.0, pad=8)

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


    # scaled
    for i, df in enumerate(scaled_dfs):

        for col in sensor_cols:
            # NOTE: plot with priority 
            base = col.lower().split(" (")[0] # Get base property name (remove units in parentheses)
            z_order = get_priority_zorder(base)
            ax[i+2].plot(renamed_df.index, df[col], label=col, linewidth=1, zorder=z_order)

        # Isolate times for title
        start_time = df["timestamp"].iloc[0]
        end_time = df["timestamp"].iloc[-1]

        # Labels and title
        ax[i+2].set_xlabel("Time", labelpad=12)
        ax[i+2].set_ylabel("Sensor Value", labelpad=12)
        ax[i+2].set_title(f"{'Scaled Data'}{': Denoised' if i % 2 !=0 else': ReSampling Period = 900s'}\n", y=1.0, pad=8)

        # Set axis limits
        ax[i+2].margins(x=0.01, y=0.01)
        ax[i+2].set_ylim(top=ymax)
        #ax.set_xlim(left=start_time, right=end_time)

        # Set tick frequencies
        ymin, ymax_actual = ax[i].get_ylim()
        y_range = ymax - ymin if ymax else ymax_actual - ymin
        raw_ytick_step = y_range / 5  # target ~5 major ticks
        ytick_step = round_data_tick_size(raw_ytick_step)
        ax[i+2].yaxis.set_major_locator(MultipleLocator(ytick_step))

        x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)    
        raw_xtick_step = x_range / 5 # Target: ~5 x-axis ticks
        xtick_step = round_data_tick_size(raw_xtick_step)
        ax[i+2].xaxis.set_major_locator(mdates.DayLocator(interval=xtick_step))
        ax[i+2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

        # Grid and legend
        ax[i+2].grid(True, which="major", linestyle="--", linewidth=0.5)
        ax[i+2].legend(title="Sensors", loc="upper right")


    # Add overall title
    fig.suptitle(f"{title}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold',
                 y=0.98,
                 x=0.51
                 )

    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.94, hspace=0.3)
    plt.show()

def plot_all(df: pd.DataFrame, title: str = "Sensor Readings Over Time", ymax: float = None, normalized: bool = False) -> None:
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
    # Set style for plots
    # sns.set_style("darkgrid")

    renamed_df = rename_columns_with_units(df) # rename columns with units
    plot_df = smooth_df(renamed_df) if normalized else renamed_df # if selected normalized then do so

    # Set 'timestamp' as index (no longer a column)
    plot_df = plot_df.set_index("timestamp")

    # Define figure and axes for subplots
    fig, ax = plt.subplots(figsize=(16, 10))

    # Get sensor columns
    sensor_cols = plot_df.columns.tolist()

    for col in sensor_cols:
        # ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8) # NOTE: plot without priority

        # NOTE: plot with priority 
        base = col.lower().split(" (")[0] # Get base property name (remove units in parentheses)
        z_order = get_priority_zorder(base)
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1, zorder=z_order)

    # Isolate times for title
    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]

    # Labels and title
    ax.set_xlabel("Time", labelpad= 12)
    ax.set_ylabel("Sensor Value", labelpad=12)
    ax.set_title(f"{title}{' (Normalized)' if normalized else''}\n"
            f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
            fontweight='bold',
            pad=15
            )
    
    # Set axis limits and margin
    ax.margins(x=0.01,y=0.01)
    ax.set_ylim(top=ymax if ymax else None)
    #ax.set_xlim(left=start_time, right=end_time)

    # Set tick frequency
    ymin, ymax_actual = ax.get_ylim()
    y_range = ymax - ymin if ymax else ymax_actual - ymin
    raw_ytick_step = y_range / 5  # target ~5 major ticks
    ytick_step = round_data_tick_size(raw_ytick_step)
    ax.yaxis.set_major_locator(MultipleLocator(ytick_step))

    x_range = (end_time - start_time).total_seconds() / (60 * 60 * 24)    
    raw_xtick_step = x_range / 5 # Target: ~5 x-axis ticks
    xtick_step = round_data_tick_size(raw_xtick_step)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=xtick_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    #fig.autofmt_xdate()

    # Grid and legend
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="upper right")

    plt.tight_layout()
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

        ax.set_title(f"Oxygen vs {sensor_col}", y=1.0, pad=10)
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
    fig.suptitle(f"{title}{' (Normalized)' if normalized else''}\n"
                 f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                 fontweight='bold',
                 y=0.98,
                 x=0.51
                 )
                 
    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.92, hspace=0.4)
    plt.show()