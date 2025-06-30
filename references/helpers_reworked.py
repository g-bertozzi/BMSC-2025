import onc
import pandas as pd

from typing import List, Tuple
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


""" GOALS
- provide overview on upwelling and downwelling, hypoxia, winds and currents by seasons 

1. access temp, salinity, chlorophyll OR oxygen at folger pinnacle and deep - over 2021

2. keep the getScalarData call visibile - keep everything as visible as possible?

3. maybe: plot the casts vs mounts? - this would be 2023 though
"""

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

# TODO: make a defualt color for each property to be plotted in
"""
Global metadata dictionary for each device category - e.g. CTD

Useful for naming dataframe columns with clean, capitlzed names and units.

SCHEMA: deviceCategoryCode : [{propertyCode, name (unit)}]
"""
device_info = {
    "OXYSENSOR": {
        "oxygen": {"label": "Oxygen (ml/l)", "unit": "ml/l"}
    },
    "radiometer": {
        "parphotonbased": {"label": "PAR (µmol/m²/s)", "unit": "µmol/m²/s"}
    },
    "FLNTU": {
        "chlorophyll": {"label": "Chlorophyll (µg/l)", "unit": "µg/l"},
        "turbidityntu": {"label": "Turbidity (NTU)", "unit": "NTU"}
    },
    "CTD": {
        "conductivity": {"label": "Conductivity (S/m)", "unit": "S/m"},
        "seawatertemperature": {"label": "Temperature (°C)", "unit": "°C"},
        "density": {"label": "Density (kg/m3)", "unit": "kg/m3"}
    }
}


# FETCHING DATA
def get_device_parameters(start: str, end: str, locationCode: str, deviceCategoryCode: str, resample: int = None) -> dict:
    """
    Constructs the parameters dictionary for querying ONC scalar data for a specific device category, location, and time window.

    Handles differences between devices that use 'propertyCode' (e.g., CTD, FLNTU) 
    and those that use 'sensorCategoryCodes' (e.g., OXYSENSOR). Supports optional resampling.

    Parameters:
        start (str): ISO timestamp of the start time (e.g., "2021-01-01T00:00:00.000Z")
        end (str): ISO timestamp of the end time (e.g., "2021-12-31T23:59:59.999Z")
        locationCode (str): ONC location code (e.g., "FGPD")
        deviceCategoryCode (str): Device type (e.g., "CTD", "OXYSENSOR")
        resample (int, optional): If provided, enables ONC's server-side resampling in seconds

    Returns:
        dict: Parameters dictionary to use in ONC's getScalarData() API call
    """

    # Join all propertyCodes into one comma-separated string
    propertyCode = ",".join(device_info[deviceCategoryCode].keys())


    if resample: 
        # If OXYSENSOR, must use sensorCategoryCodes instead of propertyCode
        if deviceCategoryCode == "OXYSENSOR":
                params = {
                "locationCode": locationCode,
                "deviceCategoryCode": deviceCategoryCode,
                "sensorCategoryCodes": "oxygen_corrected",
                "dateFrom": start,
                "dateTo" : end,
                "metadata": "minimum",
                "qualityControl": "clean",
                "resamplePeriod": resample,
                "resampleType": "avg"
                }
        else:
            # For other devices, use propertyCode list for resampled query
            params = {
                "locationCode": locationCode,
                "deviceCategoryCode": deviceCategoryCode,
                "propertyCode": propertyCode,
                "dateFrom": start,
                "dateTo" : end,
                "metadata": "minimum",
                "qualityControl": "clean",
                "resamplePeriod": resample,
                "resampleType": "avg"
                }
    else:
        # No resampling: same distinction between OXYSENSOR and other devices
        if deviceCategoryCode == "OXYSENSOR":
                params = {
                "locationCode": locationCode,
                "deviceCategoryCode": deviceCategoryCode,
                "sensorCategoryCodes": "oxygen_corrected",
                "dateFrom": start,
                "dateTo" : end,
                }
        else:   
            params = {
                "locationCode": locationCode,
                "deviceCategoryCode": deviceCategoryCode,
                "propertyCode": propertyCode,
                "dateFrom": start,
                "dateTo" : end,
            }

    # NOTE: optional
    # print(f"Parameters: {params}")
    
    return params

# MAKING DATAFRAME
def get_device_dataframe(start: str, end: str, locationCode: str, result: dict) -> pd.DataFrame:
    """ 
    Converts an ONC JSON response into a single merged DataFrame for one location and device.

    For each propertyCode in the response, creates a column labeled with a cleaned property code and unit using `device_info`,
    and merges them on the 'Time' column (which is also set as a datetime index).

    Parameters:
        start (str): Start timestamp (ISO format)
        end (str): End timestamp (ISO format)
        locationCode (str): ONC location code (e.g., "FGPD")
        result (dict): JSON object returned by ONC getScalarData()

    Returns:
        pd.DataFrame: Merged sensor DataFrame indexed by 'Time'
    """
    # NOTE: optional - debugging
    # print(f"Requesting data at {locationCode} from {start} to {end}")

    # Error handle if there is no data returned
    if not result or "sensorData" not in result or result["sensorData"] is None or len(result["sensorData"]) == 0:
        print(f"No data returned for CTD devices at {locationCode} between {start} and {end}.")
        return
        
    dfs = []

    # Extract the sensors from the JSON response
    for sensor in result["sensorData"]:

        # Extract each sensors data fields
        prop = sensor["propertyCode"]
        times = sensor["data"]["sampleTimes"]
        values = sensor["data"]["values"]

        # Default to raw property name if no match
        column_title = prop
        
        # Search for the label in the new device_info structure
        for dev_cat, props in device_info.items():
            if prop in props:
                column_title = props[prop]["label"]
                break


        # Populate dataframe with induvidual sensor property
        df = pd.DataFrame({
            "Time": pd.to_datetime(times), # Convert strings to datetime objects
            column_title: values,
        })
        dfs.append(df)

    # Merge dataframes by joining on Time    
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), dfs)
    df_merged.sort_values("Time", inplace=True)

    # NOTE: optional - debugging
    # start_time = df_merged["Time"].iloc[0]
    # end_time = df_merged["Time"].iloc[-1]
    # print(f"{locationCode} Dataframe start: {start_time} Dataframe end: {end_time}")

    return df_merged

def merge_device_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges a list of device DataFrames on the 'Time' column.
    Assumes all dataframes are valid and contain a 'Time' column.
    Always sets 'Time' as index.

    Parameters:
        dfs (List[pd.DataFrame]): List of valid dataframes to merge.

    Returns:
        pd.DataFrame: Merged dataframe indexed by Time.
    """
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), dfs)
    merged_df.sort_values("Time", inplace=True)
    merged_df.set_index("Time", inplace=True)

    return merged_df


# FURTHER PROCESSING
def smooth_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rolling mean smoothing and rolling z-score outlier filtering 
    to all data (i.e. numeric) columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'Time' and sensor data.

    Returns:
        pd.DataFrame: Smoothed and filtered DataFrame (same shape).
    """
    import numpy as np

    window = 12  # Size of the rolling window for smoothing
    z_thresh = 3.0  # Z-score threshold for outlier detection

    smoothed_df = df.copy()  # Work on a copy to preserve the original

    # Apply rolling smoothing and z-score filtering to each numeric column
    for col in smoothed_df.columns:
        # Compute rolling mean and std deviation using centered window
        roll_mean = smoothed_df[col].rolling(window=window, center=True).mean()
        roll_std = smoothed_df[col].rolling(window=window, center=True).std()

        # Calculate z-scores for detecting outliers
        z_scores = (smoothed_df[col] - roll_mean) / roll_std

        # Replace values with rolling mean where the z-score is within the threshold; otherwise set to NaN
        smoothed_df[col] = roll_mean.where(z_scores.abs() < z_thresh, np.nan)

    return smoothed_df


# PLOTTING FUNCTIONS
def plot_dataframe(df: pd.DataFrame, locationCode: str, columns: str = None, smooth: bool = False) -> None:
    """
    Overlays sensor columns in a single time series plot, indexed by 'Time'.
    Optionally filters by column list and applies smoothing to remove noise/outliers.

    Parameters:
        df (pd.DataFrame): A DataFrame indexed by time, with one or more numeric sensor columns.
        locationCode (str): ONC location code used to label the plot.
        columns (str, optional): Comma-separated list of columns to plot. Defaults to all.
        smooth (bool, optional): Whether to apply rolling smoothing and outlier filtering. Default is False.

    Returns:
        None
    """
    # Define figure and axes for subplots
    fig, ax = plt.subplots(figsize=(16, 10), nrows= 1, ncols= 1)

    # Determine which columns to plot
    if columns:
        selected_cols = [col.strip() for col in columns.split(",")]
        plot_df = df[selected_cols]
    else:
        plot_df = df

    plot_df = smooth_df(plot_df) if smooth else plot_df

    # To format title
    start_time = plot_df.index[0].strftime("%d/%m/%y")
    end_time = plot_df.index[-1].strftime("%d/%m/%y")
    column_list = list(plot_df.columns)
    plot_title = ", ".join(column_list)

    # Plot each column
    for column in plot_df:
        ax.plot(plot_df.index, plot_df[column], label=column, linewidth=0.8) # plot without priority
    
    # Label axes
    ax.set_xlabel("Time", labelpad=13)
    ax.set_ylabel("Sensor Value", labelpad=13)

    ax.set_title(f"{place[locationCode]['name']} -  {plot_title}\n"
                 f"{start_time} to {end_time}", 
                 fontweight="bold", 
                 pad=14,)

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis="x")

    # TODO: make y axis ticks more frequent

    # Add grid and legend
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
    ax.legend(title="Sensors", loc="best")


    # Adjust layout to make space for subtitle
    # plt.subplots_adjust(top=0.93, hspace=0.3)

    plt.show()
    
     
def subplot_dataframe(df: pd.DataFrame, locationCode: str, columns: str = None, smooth: bool = False) -> None:
     """ 
     Subplots each sensor (column in dataframe) against time.

     """
     
