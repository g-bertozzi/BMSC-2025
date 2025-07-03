import onc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from typing import List, Tuple
from functools import reduce
import os

# token = os.environ["GRACE_TOKEN"]
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("ONC_TOKEN")

my_onc = onc.ONC(token)

# schema: propertyCode: {label, deviceCategoryCode, color}
sensor_info = {
    "oxygen": {
        "label": "Oxygen Saturation (%)",
        "deviceCategoryCode": "OXYSENSOR",
        "sensorCategoryCode": "oxygen_saturation",
        "color": "royalblue",
    },
    "chlorophyll": {
        "label": "Chlorophyll (µg/l)",
        "deviceCategoryCode": "FLUOROMETER",
        "sensorCategoryCode": "chlorophyll",
        "color": "darkgreen",
    },
    "seawatertemperature": {
        "label": "Temperature (°C)",
        "deviceCategoryCode": "CTD",
        "sensorCategoryCode": "temperature",
        "color": "crimson",
    },
    "salinity": {
        "label": "Salinity (psu)",
        "deviceCategoryCode": "CTD",
        "sensorCategoryCode": "salinity",
        "color": "orange",
    },
    "turbidityntu": {
        "label": "Turbidity (NTU)",
        "deviceCategoryCode": "TURBIDITYMETER",
        "color": "saddlebrown",
    },
    "conductivity": {
        "label": "Conductivity (S/m)",
        "deviceCategoryCode": "CTD",
        "sensorCategoryCode": "conductivity",
        "color": "mediumorchid",
    },
    "density": {
        "label": "Density (kg/m3)",
        "deviceCategoryCode": "CTD",
        "sensorCategoryCode": "density",
        "color": "darkcyan",
    },
    "depth": {
        "label": "Depth (m)",
        "deviceCategoryCode": "CTD",
        "sensorCategoryCode": "depth",
        "color": "pink" # TODO: change
    },
}

place = {
    "FGPPN": {
        "name": "Folger Pinnacle",
        "mountCode": "FGPPN",
        "castCode":"CF341",
        "mountDepth": 23
    },
    "FGPD": {
        "name": "Folger Deep",
        "mountCode": "FGPD",
        "castCode": "CF340",
        "mountDepth": 90
    }
}

def fetch_property_result(start: str, end: str, locationCode: str, propertyCode: str, updates: bool, resample: int = None) -> dict:
    """
    Makes ONC API call to get scalar data for a single propertyCode (for Casts).
    """
    sensor_cat = sensor_info[propertyCode]["sensorCategoryCode"]
    device_cat = sensor_info[propertyCode]["deviceCategoryCode"]
    
    # print(f" sensor cat: {sensor_cat}, dev cat: {device_cat}") # NOTE: debug

    # Default parameters (propertyCode is only used internally, not passed to the API)
    params = {
        "locationCode": locationCode,
        "deviceCategoryCode": device_cat,
        "sensorCategoryCodes": sensor_cat,
        "dateFrom": start,
        "dateTo": end,
    }

    if updates: print(f"API Request: getScalarData{params}")

    result = my_onc.getScalardata(params)

    # NOTE: debug
    # sensorData_df = pd.DataFrame(result['sensorData']) 
    # print(f"Sensor data: {sensorData_df}")

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


def detect_cast_intervals(df: pd.DataFrame, gap_threshold_minutes: int = 10) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Detects individual CTD cast intervals based on time gaps in the data.

    Parameters:
        df (pd.DataFrame): DataFrame indexed by datetime.
        gap_threshold_minutes (int): Time gap threshold to detect breaks between casts.

    Returns:
        List[Tuple[pd.Timestamp, pd.Timestamp]]: List of (start_time, end_time) timestamp pairs.
    """
 
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return []

    df = df.sort_index()

    # Calculate time gaps between index entries
    gaps = df.index.to_series().diff().fillna(pd.Timedelta(seconds=0))
    new_cast_starts = df.index[gaps > pd.Timedelta(minutes=gap_threshold_minutes)]

    # Get positional indices of cast boundaries
    new_cast_indices = df.index.get_indexer(new_cast_starts)
    cast_starts = [0] + new_cast_indices.tolist()
    cast_ends = new_cast_indices.tolist() + [len(df)]

    # Extract (start_time, end_time) for each cast
    intervals = [
        (
            df.index[start_idx],
            df.index[end_idx - 1]
        )
        for start_idx, end_idx in zip(cast_starts, cast_ends)
    ]

    return intervals

def detect_deep_intervals(df: pd.DataFrame, locationCode: str, gap_threshold_seconds: int = 60) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Detects continuous time intervals where depth is within 15m of the mount depth.

    Parameters:
        df (pd.DataFrame): DataFrame indexed by datetime, with a 'depth' column.
        locationCode (str): Location code to look up mount depth from global 'place'.
        gap_threshold_seconds (int): Max allowed time gap between points within an interval.

    Returns:
        List[Tuple[pd.Timestamp, pd.Timestamp]]: List of (start, end) timestamp pairs.
    """

    mount_depth = place[locationCode]["mountDepth"]
    depth_threshold = mount_depth - 10

    df_deep = df[df["Depth (m)"] >= depth_threshold].copy()
    if df_deep.empty:
        return []

    df_deep = df_deep.sort_index()
    df_deep["delta"] = df_deep.index.to_series().diff().dt.total_seconds().fillna(0)

    intervals = []
    start_time = df_deep.index[0]

    for i in range(1, len(df_deep)):
        if df_deep.iloc[i]["delta"] > gap_threshold_seconds:
            end_time = df_deep.index[i - 1]
            intervals.append((start_time, end_time))
            start_time = df_deep.index[i]

    intervals.append((start_time, df_deep.index[-1]))
    return intervals

def get_cast_info(cast_df: pd.DataFrame, locationCode: str) -> List[dict]:
    """
    Extracts information about each cast, including start/end and deep section timing.

    Parameters:
        cast_df (pd.DataFrame): CTD data indexed by datetime, with a 'depth' column.
        depth_threshold (int): Minimum depth to define the 'deep' section of a cast.

    Returns:
        List[dict]: A list of casts with metadata.
    """
    casts = []
    cast_ints = detect_cast_intervals(df=cast_df, gap_threshold_minutes=10)
    deep_ints = detect_deep_intervals(df=cast_df, locationCode=locationCode, gap_threshold_seconds=60)

    for i, (cast, deep) in enumerate(zip(cast_ints, deep_ints)):
        cast_info = {
            "cast_id": f"cast_{i}",
            "start": cast[0],
            "end": cast[1],
            "deep_start": deep[0],
            "deep_end": deep[1]
        }
        casts.append(cast_info)

    return casts

# TODO: fix and integrate for all properties
def subplot_cast_and_mount_property(dataframes: list[pd.DataFrame], locationCode: str, propertyCode: str) -> None:
    """
    Plots cast and mount property (e.g., temperature, oxygen) as color gradients vs. time and depth.
    Cast data must contain depth column; mount is assumed fixed depth.

    Parameters:
        dataframes (list[pd.DataFrame]): [mount_df, cast_df] — cast must contain depth column.
        locationCode (str): ONC location code (e.g., "FGPPN").
        propertyCode (str): ONC property code (e.g., "seawatertemperature").
    """    # Get display label
    label = sensor_info[propertyCode]["label"]

    # Extract input data
    mount_df, cast_df = dataframes
    mount_depth_m = place[locationCode]["mountDepth"]

    # Dynamically locate matching column names
    cast_col = next((col for col in cast_df.columns if label in col), None)
    mount_col = next((col for col in mount_df.columns if label in col), None)
    depth_col = next((col for col in cast_df.columns if "Depth" in col), None)

    if not cast_col or not mount_col or not depth_col:
        raise KeyError(f"Could not find expected columns in cast or mount data for '{label}'")

    # Get cast interval info
    cast_info = get_cast_info(cast_df=cast_df, locationCode=locationCode)
    num_casts = len(cast_info)

    fig, axes = plt.subplots(nrows=num_casts, ncols=2, figsize=(14, 5 * num_casts), sharex=False)
    axes = axes.flatten()

    for i, cur in enumerate(cast_info):
        # Subset for each cast interval
        cast = cast_df[(cast_df.index >= cur["start"]) & (cast_df.index <= cur["end"])]
        mount = mount_df[(mount_df.index >= cur["start"]) & (mount_df.index <= cur["end"])]
        cast_deep = cast_df[(cast_df.index >= cur["deep_start"]) & (cast_df.index <= cur["deep_end"])]
        mount_deep = mount_df[(mount_df.index >= cur["deep_start"]) & (mount_df.index <= cur["deep_end"])]

        vmin = cast_df[cast_col].min()
        vmax = cast_df[cast_col].max()

        for j, (c, m, section_label) in enumerate([(cast, mount, "Entire Cast"), (cast_deep, mount_deep, "Deep Section")]):
            ax = axes[i * 2 + j]

            if c.empty or m.empty:
                ax.set_title(f"No data | {section_label} Cast {i+1}")
                continue

            sc_cast = ax.scatter(
                c.index, c[depth_col],
                c=c[cast_col],
                cmap="turbo",
                s=20,
                edgecolor="none",
                label="Cast",
                vmin=vmin, vmax=vmax
            )

            sc_mount = ax.scatter(
                m.index, [mount_depth_m] * len(m),
                c=m[mount_col],
                cmap="turbo",
                s=20,
                edgecolor="none",
                marker="s",
                label="Mount",
                vmin=vmin, vmax=vmax
            )

            local_min = min(c[cast_col].min(), m[mount_col].min())
            local_max = max(c[cast_col].max(), m[mount_col].max())

            ax.text(
                0.04, 0.15, f"Min: {local_min:.3f}\nMax: {local_max:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(facecolor='lightgrey', edgecolor='grey', boxstyle='round', linewidth=0.8)
            )

            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Depth (m)")
            ax.invert_yaxis()
            ax.set_title(f"{section_label} | Cast {i+1}")
            ax.legend()
            ax.grid(True)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

            ticks = pd.date_range(start=c.index.min(), end=c.index.max(), periods=5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([t.strftime('%H:%M:%S') for t in ticks])

            cbar = plt.colorbar(sc_cast, ax=ax)
            cbar.set_label(label, labelpad=13)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

    plt.subplots_adjust(top=0.89, hspace=0.3, wspace=0.3)
    fig.suptitle(
        f'{place[locationCode]["name"]} - {label}\n{cast_df.index.min().strftime("%B %d, %Y")}',
        y=0.97,
        fontweight="bold"
    )
    plt.show()