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


""" GOALS
- provide overview on upwelling and downwelling, hypoxia, winds and currents by seasons 

1. access temp, salinity, chlorophyll OR oxygen at folger pinnacle and deep - over 2021

2. keep the getScalarData call visibile - keep everything as visible as possible?

3. maybe: plot the casts vs mounts? - this would be 2023 though
"""

# SCHEMA: deviceCategoryCode : [{propertyCode, name (unit)}]
device_info = {
    "OXYSENSOR": [{"propertyCode": "oxygen", "name": "Oxygen (ml/l)", "sensorCategoryCode": "oxygen_corrected"}],
    "radiometer": [{"propertyCode": "parphotonbased", "name": "PAR (µmol/m²/s)"}],
    "FLNTU": [{"propertyCode": "chlorophyll", "name": "Chlorophyll (µg/l)"}, 
               {"propertyCode": "turbidityntu", "name": "Turbidity (NTU)"}],
    "CTD": [{"propertyCode": "conductivity", "name": "Conductivity (S/m)"}, 
            {"propertyCode": "seawatertemperature", "name": "Temperature (°C)"},
            {"propertyCode": "density", "name": "Density (kg/m3)"}],
    }

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
    propertyCode = ",".join([prop["propertyCode"] for prop in device_info[deviceCategoryCode]])

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

    print(f"Parameters: {params}")
    
    return params

def get_device_dataframe( start: str, end: str, locationCode: str, result: dict) -> pd.DataFrame: # result is JSON
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

        # Look for a matching entry in device_info dictionary
        for devCat in device_info: # go through each deviceCategoryCode
            for dev in device_info[devCat]: # go through each property  
                if dev["propertyCode"] in prop or prop in dev["propertyCode"]: # check if this is the current property
                    column_title = dev["name"]  # already includes unit, e.g., "Oxygen (ml/l)"
                    break

        # Populate dataframe with induvidual sensor property
        df = pd.DataFrame({
            "Time": pd.to_datetime(times), # convert strings to datetime objects
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

def plot_dataframe(df: pd.DataFrame) -> None:
    """
    Overlays plots of each sensor (column in dataframe) against time.

    """
    
     
def subplot_dataframe(df: pd.DataFrame) -> None:
     """ 
     Subplots each sensor (column in dataframe) against time.

     """
     
