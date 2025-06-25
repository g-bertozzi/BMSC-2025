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
1. access temp, salinity, chlorophyll OR oxygen at folger pinnacle and deep - over 2021

2. keep the getScalarData call visibile


- 

to access CTDS use: locationCode and sensorCategoryCodes


to acesss other categories use locationCode, deviceCategoryCode, propertyCode

"""

device_info = {
    "OXYSENSOR": [{"propertyCode": "oxygen", "name": "Oxygen", "unit": "ml/l", "sensorCategoryCode": "oxygen_corrected"}],
    "radiometer": [{"propertyCode": "parphotonbased", "name": "PAR", "unit": "µmol/m²/s" }],
    "FLNTU": [{"propertyCode": "chlorophyll", "name": "Chlorophyll", "unit": "µg/l"}, 
               {"propertyCode": "turbidityntu", "name": "Turbidity", "unit": "NTU"}],
    "CTD": [{"propertyCode": "conductivity", "name": "Conductivity", "unit": "S/m"}, 
            {"propertyCode": "seawatertemperature", "name": "Temperature", "unit": "°C"},
            {"propertyCode": "density", "name": "Density", "unit": "kg/m3"}],
    }

def get_device_parameters(start: str, end: str, locationCode: str, deviceCategoryCode: str, resample: int = None) -> dict: # only pro of this is for resample and dealing with propertyCode vs sensorCatCodes

    # Join all propertyCodes into one comma-separated string
    propertyCode = ",".join([prop["propertyCode"] for prop in device_info[deviceCategoryCode]])

    if resample: 

        if deviceCategoryCode == "OXYSENSOR": # NOTE: sensorCategoryCodes instead of propertyCode for OXYSENSOR
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

def get_device_dataframe(result: dict, start: str, end: str, locationCode: str) -> pd.DataFrame: # result is JSON
    """ 
    Generates dataframe with timestamp and sensor values from JSON response for specific devices in a given location.
    Parameters:

    Returns:
    """
    # print(f"Requesting data at {locationCode} from {start} to {end}") # NOTE: debugging

    # Error handle if there is no data returned
    if not result or "sensorData" not in result or result["sensorData"] is None or len(result["sensorData"]) == 0:
        print(f"No data returned for CTD devices at {locationCode} between {start} and {end}.")
        return
        
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

    # print(f"{locationCode} Dataframe start: {start_time} Dataframe end: {end_time}") # NOTE: debugging

    return df_merged

def merge_device_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges a list of device DataFrames on the 'timestamp' column.
    Assumes all dataframes are valid and contain a 'timestamp' column.
    Always sets 'timestamp' as index.

    Parameters:
        dfs (List[pd.DataFrame]): List of valid dataframes to merge.

    Returns:
        pd.DataFrame: Merged dataframe indexed by timestamp.
    """
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="timestamp", how="outer"), dfs)
    merged_df.sort_values("timestamp", inplace=True)
    merged_df.set_index("timestamp", inplace=True)

    return merged_df

def plot_dataframe(df: pd.DataFrame) -> None:
     
