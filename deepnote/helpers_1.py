import pandas as pd
import onc
import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from functools import reduce

# token = os.environ["GRACE_TOKEN"]
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("ONC_TOKEN")

# Create ONC client
my_onc = onc.ONC(token)

places = {"FGPPN": "Folger Pinnacle", "FGPD": "Folger Deep", "CF341": "Cast at Folger Pinnacle", "CF340": "Cast at Folger Deep"}


def get_property(start: str, end: str, locationCode: str, deviceCategoryCode: str, sensorCategoryCode: str) -> pd.DataFrame:
    """
    Fetches scalar data using the ONC Python SDK for a given location, device category, sensor property,
    and time window. Returns a merged DataFrame with timestamps and sensor values.

    Parameters:
        start (str): Start date in ISO 8601 format (e.g., "2023-07-11T17:00:00.000Z").
        end (str): End date in ISO 8601 format (e.g., "2023-07-11T22:30:00.000Z").
        locationCode (str): ONC location code (e.g., "CF341").
        deviceCategoryCode (str): ONC device category (e.g., "CTD").
        sensorCategoryCode (str): Comma-separated sensor types to fetch (e.g., "depth,temperature").

    Returns:
        pd.DataFrame: DataFrame containing merged sensor values with a timestamp index.
                    schema: timestamp: datetime obj, {prop}: int, uom: str
    """
    params = {
        "locationCode": locationCode,
        "deviceCategoryCode": deviceCategoryCode,
        "sensorCategoryCodes": sensorCategoryCode,
        "dateFrom": start,
        "dateTo" : end
    }

    # JSON response from ONC
    result = my_onc.getScalardata(params)

    # error handle if there is no data returned
    if not result or "sensorData" not in result or result["sensorData"] is None or len(result["sensorData"]) == 0:
        print(f"No data returned for devices in {deviceCategoryCode} at {locationCode} between {start} and {end}.")
        return
        
    else:
        # extract the relevant sensors from the JSON response

        dfs = []

        for sensor in result["sensorData"]:
            # extract each sensors data fields
            prop = sensor["sensorCategoryCode"]
            times = sensor["data"]["sampleTimes"]
            values = sensor["data"]["values"]
            unit = sensor["unitOfMeasure"]

            # populate dataframe (Pandas)
            # schema: timestamp: datetime obj, {prop}: int, uom: str
            df = pd.DataFrame({
                # syntax: "label": variable
                "timestamp": pd.to_datetime(times), # convert strings to datetime objects
                prop: values,
                "uom": unit
            })
            dfs.append(df)

    # merge dataframes by joining on timestamp    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs)
    df_merged.sort_values("timestamp", inplace=True)
    
    df_merged.head()
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
    deep_ints = detect_deep_intervals(df=cast_df, depth_threshold=20, gap_threshold_seconds=60)

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

def plot_cast_depth_vs_temp(start: pd.Timestamp, end: pd.Timestamp, locationCode: str, df: pd.DataFrame, depth_threshold: int) -> None:
    """
    Plots CTD cast: Temperature vs Depth, ignoring time axis.

    Parameters:
        start (pd.Timestamp): Start time.
        end (pd.Timestamp): End time.
        locationCode (str): Label for plot title.
        df (pd.DataFrame): DataFrame with 'timestamp', 'temperature', and 'depth'.
        depth_threshold (int): Minimum depth to include.

    Returns:
        None
    """
    
    # Filter by time and depth
    df_int = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end) &
        (df["depth"] >= depth_threshold)
    ].sort_values("depth")

    if df_int.empty:
        print(f"Skipping empty cast interval: {start} to {end}")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_int["temperature"], df_int["depth"], color="tab:blue", label="CTD Temperature")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    #ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title(
        f"{places[locationCode]}\n"
        f"{start.strftime('%H:%M:%S')} to {end.strftime('%H:%M:%S')} {end.strftime('%b %d, %Y')}",
        fontweight="bold"
    )
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

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

def plot_mount_temp_vs_time(start: pd.Timestamp, end: pd.Timestamp, locationCode: str, df: pd.DataFrame) -> None:
    """
    Plots a temperature time-series with fewer time labels.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df_int = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_int["timestamp"], df_int["temperature"], label="Mount Temperature")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlim([start - pd.Timedelta(seconds=1), end + pd.Timedelta(seconds=1)])

    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    title_str = f"{places[locationCode]}\n{start.strftime('%H:%M:%S')} to {end.strftime('%H:%M:%S')} {end.strftime('%B %d, %Y')}"
    ax.set_title(title_str, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
def cast_and_mount_temp_plot(
    cast_df: pd.DataFrame,
    mount_df: pd.DataFrame,
    mount_depth_m: int,
    locationCode: str,
    title: str
) -> None:
    """
    Plots cast and mount temperature as color gradients vs. time and depth.

    Cast data is plotted at measured depths.
    Mount data is plotted at fixed depth (mount_depth_m), colored by temperature.

    Parameters:
        cast_df (pd.DataFrame): DataFrame with 'timestamp', 'temperature', 'depth'
        mount_df (pd.DataFrame): DataFrame with 'timestamp', 'temperature'
        mount_depth_m (int): Depth of fixed mount sensor
        locationCode (str): ONC location code
        title (str): Plot title
    """
    cast_info = get_cast_info(cast_df=cast_df, depth_threshold=mount_depth_m)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    for i, cur in enumerate(cast_info[:2]):  # limit to 2 casts (2x2 grid)
        cast = cast_df[(cast_df["timestamp"] >= cur["start"]) & (cast_df["timestamp"] <= cur["end"])]
        mount = mount_df[(mount_df["timestamp"] >= cur["start"]) & (mount_df["timestamp"] <= cur["end"])]

        cast_deep = cast_df[(cast_df["timestamp"] >= cur["deep_start"]) & (cast_df["timestamp"] <= cur["deep_end"])]
        mount_deep = mount_df[(mount_df["timestamp"] >= cur["deep_start"]) & (mount_df["timestamp"] <= cur["deep_end"])]

        for j, (c, m, label) in enumerate([(cast, mount, "Entire"), (cast_deep, mount_deep, "Deep")]):
            ax = axes[i * 2 + j]
            if c.empty or m.empty:
                ax.set_title(f"No data | {label} Cast {i}")
                continue

            vmin = min(c["temperature"].min(), m["temperature"].min())
            vmax = max(c["temperature"].max(), m["temperature"].max())

            sc_cast = ax.scatter(
                c["timestamp"], c["depth"],
                c=c["temperature"],
                cmap="viridis",
                s=20,
                edgecolor="none",
                label="Cast",
                vmin=vmin, vmax=vmax
            )

            sc_mount = ax.scatter(
                m["timestamp"], [mount_depth_m] * len(m),
                c=m["temperature"],
                cmap="viridis",
                s=20,
                edgecolor="none",
                marker="s",
                label="Mount",
                vmin=vmin, vmax=vmax
            )

            ax.set_xlabel("Time (UTC)", labelpad=10)
            ax.set_ylabel("Depth (m)", labelpad=10)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.set_title(f"{label} | Cast {i+1}", pad=10)
            ax.legend()
            ax.grid(True)

            # Format time axis with ~5 clean ticks
            duration = (c["timestamp"].max() - c["timestamp"].min()).total_seconds()
            target_step = duration / 5
            nice_step = round_data_tick_size(target_step)
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=nice_step))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

            cbar = plt.colorbar(sc_cast, ax=ax)
            cbar.set_label("Temperature (°C)", labelpad=10)

    # Adjust layout to make space for subtitle
    plt.subplots_adjust(top=0.92, hspace=0.4)
    fig.suptitle(f"{places[locationCode]} - {title}", fontsize=14, fontweight="bold")
    #fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
