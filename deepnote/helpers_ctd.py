import onc
import pandas as pd

from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from functools import reduce # used for dataframes

import os

# token = os.environ["GRACE_TOKEN"]
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("ONC_TOKEN")

# Create ONC client
my_onc = onc.ONC(token)

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

# Schema: sensorCategoryCodes : {propertyCode, name, unit}
sensor_info = {
    "conductivity": {"propertyCode": "conductivity", "name": "Conductivity", "unit": "S/m"},
    "temperature": {"propertyCode": "seawatertemperature", "name": "Temperature", "unit": "°C"},
    "density": {"propertyCode": "density", "name": "Density", "unit": "kg/m3"},
}

""" 
NOTE goals:

- plot data from CTD casts and mounts at pinnacle and or deep

- choose to plot salinity or temparture:

- overlay cast and mount property data on a subplot (done for temp)
- longer term plot of mount data, with marks for where casts are

"""
def get_property(start: str, end: str, locationCode: str, sensorCategoryCodes: str, resample: int = None) -> pd.DataFrame:
    """
    Fetches scalar data for CTDs given a given location, sensor properties and time window. 
    Returns a merged DataFrame with timestamps and sensor values.

    Parameters:
        start (str): Start date in ISO 8601 format (e.g., "2023-07-11T17:00:00.000Z").
        end (str): End date in ISO 8601 format (e.g., "2023-07-11T22:30:00.000Z").
        locationCode (str): ONC location code (e.g., "CF341").
        sensorCategoryCode (str): Comma-separated sensor types to fetch for casts and mount respectively
            options: "depth,conductivity", "conductivity", "depth,temperature", "temperature", "depth,density", "density"

    Returns:
        pd.DataFrame: DataFrame containing merged sensor values with a timestamp index.
                    schema: timestamp: datetime obj, {prop}: int or float
    """
    # print(f"Requesting CTD data at {locationCode} from {start} to {end}") # NOTE: debugging

    if resample:
        params = {
        "locationCode": locationCode,
        "deviceCategoryCode": "CTD",
        "sensorCategoryCodes": sensorCategoryCodes,
        "dateFrom": start,
        "dateTo" : end,
        "metadata": "minimum",
        "qualityControl": "clean",
        "resamplePeriod": resample,
        "resampleType": "avg"
        }
    else:
        params = {
        "locationCode": locationCode,
        "deviceCategoryCode": "CTD",
        "sensorCategoryCodes": sensorCategoryCodes,
        "dateFrom": start,
        "dateTo" : end
        }

    # JSON response from ONC
    result = my_onc.getScalardata(params)

    # error handle if there is no data returned
    if not result or "sensorData" not in result or result["sensorData"] is None or len(result["sensorData"]) == 0:
        print(f"No data returned for CTD devices at {locationCode} between {start} and {end}.")
        return
        
    else:
        dfs = []

        # Extract the sensors from the JSON response
        for sensor in result["sensorData"]:
            # Extract each sensors data fields
            prop = sensor["sensorCategoryCode"]
            times = sensor["data"]["sampleTimes"]
            values = sensor["data"]["values"]

            # Populate dataframe with induvidual sensor property
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(times), # convert strings to datetime objects
                prop: values,
            })
            dfs.append(df)

    # Merge dataframes by joining on timestamp    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs)
    df_merged.sort_values("timestamp", inplace=True)

    start_time = df_merged["timestamp"].iloc[0]
    end_time = df_merged["timestamp"].iloc[-1]

    # print(f"Dataframe start: {start_time} Dataframe end: {end_time}") # NOTE: debugging

    return df_merged

def detect_cast_intervals(df: pd.DataFrame, gap_threshold_minutes: int = 10) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Detects individual CTD cast intervals based on time gaps in the data.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'timestamp' column (datetime format).
        gap_threshold_minutes (int): Time gap threshold to detect breaks between casts.

    Returns:
        List[Tuple[pd.Timestamp, pd.Timestamp]]: List of (dateFrom, dateTo) pairs in ISO 8601 UTC format.
    """
    if df.empty or "timestamp" not in df.columns:
            return []

    # sort df by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    gaps = df["timestamp"].diff().fillna(pd.Timedelta(seconds=0)) # calculate difference between each time stamp and store in gaps: pandas.Series
    new_cast_starts = df.index[gaps > pd.Timedelta(minutes=gap_threshold_minutes)].tolist() # if gap > 10 mins then it's index is added to new_cast_starts list

    cast_starts = [0] + new_cast_starts
    cast_ends = new_cast_starts + [len(df)] # list of end timestamps

    # Format as ISO 8601 string with milliseconds and 'Z' for UTC
    intervals = [
        (
            df["timestamp"].iloc[start_idx],
            df["timestamp"].iloc[end_idx - 1]
        )
        for start_idx, end_idx in zip(cast_starts, cast_ends)
    ]

    return intervals
# TODO: make deep just be within say 10m of max depth?
def detect_deep_intervals(df: pd.DataFrame, depth_threshold: int, gap_threshold_seconds: int = 60) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Detects continuous time intervals where depth exceeds a threshold.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp' and 'depth' columns.
        depth_threshold (float): Minimum depth to include.
        gap_threshold_seconds (int): Max allowed time gap between points within an interval.
    
    Returns:
        List[Tuple[str, str]]: List of (start, end) timestamp strings in ISO 8601 UTC format.
    """

    # Filter for rows deeper than the threshold
    df_deep = df[df["depth"] > depth_threshold].copy()

    if df_deep.empty:
        return []

    # Sort by time and calculate time gaps
    df_deep = df_deep.sort_values("timestamp").reset_index(drop=True)
    df_deep["delta"] = df_deep["timestamp"].diff().dt.total_seconds().fillna(0)

    intervals = []
    start_time = df_deep.loc[0, "timestamp"]

    for i in range(1, len(df_deep)):
        if df_deep.loc[i, "delta"] > gap_threshold_seconds:
            end_time = df_deep.loc[i - 1, "timestamp"]
            intervals.append((start_time, end_time))
            start_time = df_deep.loc[i, "timestamp"]

    # Add the final interval
    intervals.append((start_time, df_deep.iloc[-1]["timestamp"]))

    return intervals


def get_cast_info(cast_df: pd.DataFrame, depth_threshold: int) -> list[dict]:
    """
    Extracts information about each cast, including start/end and deep section timing.

    Parameters:
        cast_df (pd.DataFrame): CTD data containing 'timestamp' and 'depth'.
        mount_df (pd.DataFrame): Mount data (currently unused, but included for future expansion).
        depth_threshold (int): Minimum depth to define the 'deep' section of a cast.

    Returns:
        list[dict]: A list where each entry represents a cast and its deep interval:
            {
                'cast_id': str,
                'start': pd.Timestamp,
                'end': pd.Timestamp,
                'deep_start': pd.Timestamp,
                'deep_end': pd.Timestamp
            }
    """

    casts = []

    # isolate cast time intervals, and deep section of each cast  -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    cast_ints = detect_cast_intervals(df=cast_df, gap_threshold_minutes=10) 
    deep_ints = detect_deep_intervals(df=cast_df, depth_threshold=depth_threshold, gap_threshold_seconds=60)

    for i, cast in enumerate(cast_ints):
        deep = deep_ints[i]
        cast_info = {
            "cast_id": f"cast_{i}",
            "start": cast[0],
            "end": cast[1],
            "deep_start": deep[0],
            "deep_end": deep[1]
        }
        casts.append(cast_info)

    return casts


def plot_longterm_mount(df: pd.DataFrame, locationCode: str, sensor_cols: list[str], title: str = None) -> None:
    """
    Plots long-term mount data for multiple sensor parameters (e.g., temperature, density, conductivity).

    Parameters:
        df (pd.DataFrame): Must contain 'timestamp' and one or more sensor parameter columns.
        locationCode (str): ONC locationCode.
        title (str): Optional custom title.

    Returns:
        None
    """
    df = df.copy().sort_values("timestamp").set_index("timestamp")


    fig, ax = plt.subplots(figsize=(12, 6))

    for col in sensor_cols:
        meta = sensor_info[col]
        label = f"{meta['name']} ({meta['unit']})"
        ax.plot(df.index, df[col], label=label)

    ax.set_title(f"CTD Mount\n{place[locationCode]['name']}", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sensor Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def subplot_longterm_mount(df: pd.DataFrame, locationCode: str, title: str = None) -> None:
    """
    Subplots long-term mount data for all 3 sensor parameters (i.e. temperature, density, conductivity).

    Parameters:
        df (pd.DataFrame): Must contain 'timestamp' and one or more sensor parameter columns.
        locationCode (str): ONC locationCode.
        title (str): Optional custom title.

    Returns:
        None

    """
        # Isolate times for title
    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]

    # print(f"Subplotting CTD data at {place[locationCode]["name"]} from {start_time} to {end_time}")


    df = df.copy().sort_values("timestamp").set_index("timestamp")
    sensor_cols = df.columns.to_list()

    fig, axes = plt.subplots(figsize=(14, 15), nrows=3, ncols=1)

    for i, col in enumerate(sensor_cols):
        meta = sensor_info[col]
        label = f"{meta['name']} ({meta['unit']})"

        ax = axes[i]

        ax.plot(df.index, df[col], color="tab:blue", label=label)

        # TODO: fix x axis lims
        ax.set_title(f'{place[locationCode]["name"]} - {sensor_info[col]["name"]} ({sensor_info[col]["unit"]})')
        ax.set_ylabel(label, labelpad=15)
        ax.set_xlabel("Date", labelpad=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    fig.suptitle(f"{place[locationCode]['name']} CTD Mount\n"
                f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
                fontweight='bold',
                y=0.97,
                x=0.515
                )

    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.92, hspace=0.23)

    plt.show()

# TODO: make a version of this function that will subplot casts and mounts with either: conductivity, temp, or
# TODO: make input df able to have all CTD props for the time frame and select temp/ one?
def subplot_cast_and_mount_temp_by_place(dataframes: list[pd.DataFrame], locationCode: str) -> None:
    """
    Plots cast and mount temperature as color gradients vs. time and depth. Works for inputs of 1 or 2 casts.

    Cast data is plotted at measured depths.
    Mount data is plotted at fixed depth (mount_depth_m), colored by temperature.

    Parameters:
        cast_df (pd.DataFrame): DataFrame with 'timestamp', 'temperature', 'depth'
        mount_df (pd.DataFrame): DataFrame with 'timestamp', 'temperature'
        mount_depth_m (int): Depth of fixed mount sensor
        locationCode (str): ONC location code
        title (str): Plot title
    """
    # Isolate each dataframe
    mount_df, cast_df = dataframes[0], dataframes[1]

    # Get cast interval details- returns a list of dicts containg info on each cast
    mount_depth_m = place[locationCode]["mountDepth"]
    depth_threshold = place[locationCode]["depthThreshold"]
    # print(f"{locationCode} {mount_depth_m}")
    cast_info = get_cast_info(cast_df=cast_df, depth_threshold=depth_threshold)
    num_casts = len(cast_info)

    # Define figure and axes 
    fig, axes = plt.subplots(nrows=num_casts, ncols=2, figsize=(14, 5*num_casts), sharex=False)
    axes = axes.flatten() # make 1D

    for i, cur in enumerate(cast_info):

        # isolate df for each cast interval
        cast = cast_df[(cast_df["timestamp"] >= cur["start"]) & (cast_df["timestamp"] <= cur["end"])]
        mount = mount_df[(mount_df["timestamp"] >= cur["start"]) & (mount_df["timestamp"] <= cur["end"])]

        cast_deep = cast_df[(cast_df["timestamp"] >= cur["deep_start"]) & (cast_df["timestamp"] <= cur["deep_end"])]
        mount_deep = mount_df[(mount_df["timestamp"] >= cur["deep_start"]) & (mount_df["timestamp"] <= cur["deep_end"])]

        # GLOBAL color limits (defined outside the loop) NOTE: hardcoded for specific casts
        vmin = 8
        vmax = 16

        # Iterate through entire and deep section for each cast
        for j, (c, m, label) in enumerate([(cast, mount, "Entire Cast"), (cast_deep, mount_deep, "Deep Section")]):
            ax = axes[i * 2 + j] # to iterate through 2D grid

            # Error handle: no data returned
            if c.empty or m.empty:
                ax.set_title(f"No data | {label} Cast {i}")
                continue

            sc_cast = ax.scatter(
                c["timestamp"], c["depth"],
                c=c["temperature"],
                cmap="turbo",
                s=20,
                edgecolor="none",
                label="Cast",
                vmin=vmin, vmax=vmax
            )

            sc_mount = ax.scatter(
                m["timestamp"], [mount_depth_m] * len(m),
                c=m["temperature"],
                cmap="turbo",
                s=20,
                edgecolor="none",
                marker="s",
                label="Mount",
                vmin=vmin, vmax=vmax
            )

            # Compute min and max temperature for this subplot
            local_min = min(c["temperature"].min(), m["temperature"].min())
            local_max = max(c["temperature"].max(), m["temperature"].max())

            # Label local temp mins and maxes
            ax.text(
                0.04, 0.11 if num_casts == 2 else 0.15, f"Min: {local_min:.3f}°C\nMax: {local_max:.3f}°C", # NOTE: hardcoded positions for specific casts
                             transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    facecolor='lightgrey',
                    edgecolor='grey',
                    boxstyle='round',
                    linewidth=0.8
                )
            )

            # Label axes and titles
            ax.set_xlabel("Time (UTC)", labelpad=10)
            ax.set_ylabel("Depth (m)", labelpad=10)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.set_title(f"{label} | Cast {i+1}", pad=10)
            ax.legend()
            ax.grid(True)
            
            # Isolate for title and x axis
            start_time = c["timestamp"].min()
            end_time = c["timestamp"].max()

            # Format x axis for 5 ticks
            ticks = pd.date_range(start=start_time, end=end_time, periods=5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([t.strftime('%H:%M:%S') for t in ticks])

            cbar = plt.colorbar(sc_cast, ax=ax)
            cbar.set_label("Temperature (°C)", labelpad=13)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}")) # set sig fig default to 3

    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.89 if locationCode == "FGPPN" else 0.79, hspace=0.3, wspace=0.3)  # NOTE: hardcoded positions for specific casts
    fig.suptitle(f'{place[locationCode]["name"]}\n{start_time.strftime("%B %d, %Y")}', y=0.97, fontsize=14, fontweight="bold")

    plt.show()