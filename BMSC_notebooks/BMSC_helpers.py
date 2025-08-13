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

# connect using Deepnote (secure)
# token = os.environ["GRACE_TOKEN"]
# from dotenv import load_dotenv
# load_dotenv()
# token = os.getenv("ONC_TOKEN")

# connect using env var (secure)
# Create ONC client
# my_onc = onc.ONC(token)

# Global variable to hold the ONC client
my_onc = None

# Global variable for hypoxia threshold
hypoxia_threshold = 1.4

# Schema: propertyCode: {label, deviceCategoryCode, #color}
sensor_info = {
    "oxygen": {
        "label": "Oxygen (ml/l)",
        "deviceCategoryCode": "OXYSENSOR",
        # "color": "royalblue",
    },
    "parphotonbased": {
        "label": "PAR (µmol/m²/s)",
        "deviceCategoryCode": "radiometer",
        # "color": "goldenrod",
    },
    "chlorophyll": {
        "label": "Chlorophyll (µg/l)",
        "deviceCategoryCode": "FLNTU",
        # "color": "darkgreen",
    },
    "seawatertemperature": {
        "label": "Temperature (°C)",
        "deviceCategoryCode": "CTD",
        # "color": "crimson",
    },
    "salinity": {
        "label": "Salinity (psu)",
        "deviceCategoryCode": "CTD",
        # "color": "#b1d12e",
    },
    "turbidityntu": {
        "label": "Turbidity (NTU)",
        "deviceCategoryCode": "FLNTU",
        # "color": "saddlebrown",
    },
    "density": {
        "label": "Density (kg/m3)",
        "deviceCategoryCode": "CTD",
        # "color": "darkcyan",
    },
    "sigmat": {
        "label": "Sigma-t (kg/m3)",
        "deviceCategoryCode": "CTD",
        # "color": mcolors.to_hex(cmocean.cm.thermal(0.75)),
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
    "Sigma-t (kg/m3)": mcolors.to_hex(cmocean.cm.thermal(0.75))
}

# Schema: locationCode: {name, mountCode, castCode, mountDepth}
place = {
    "FGPPN": {
        "name": "Folger Pinnacle",
        "mountCode": "FGPPN",
        "castCode":"CF341",
        "mountDepth": 25,
    },
    "FGPD": {
        "name": "Folger Deep",
        "mountCode": "FGPD",
        "castCode": "CF340",
        "mountDepth": 100,
    }
}


# API CALL FUNCTIONS

def create_onc_client(token: str) -> None:
    """
    Initializes the global ONC client with a user-provided token.

    Input: 
        token (str): ONC API token

    Output:
        None
    """
    global my_onc
    my_onc = onc.ONC(token=token)

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

    if updates: print(f"API Request: getScalardata{params}") # NOTE: for clarity 


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
    # Error handle if no API response
    if not result or "sensorData" not in result or not result["sensorData"]:
        print(f"No data for {propertyCode}")
        return None

    # Extract info from API response
    sensor = result["sensorData"][0]
    times = sensor["data"]["sampleTimes"]
    values = sensor["data"]["values"]
    column_title = sensor_info[propertyCode]["label"] if propertyCode in sensor_info else propertyCode

    # Create new dataframe
    df = pd.DataFrame({
        "Time": pd.to_datetime(times),
        column_title: values
    })

    # Set timestamps as the index of the dataframe
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

    # Make API request and dataframe for each property
    for prop in propertyCodes:
        try:
            result = fetch_property_result(start, end, locationCode, prop, updates, resample)
            df = result_to_dataframe(result, prop)
            # if updates: print(f"Creating data frame for {prop}.\nPreview: {df.columns.tolist()}") # NOTE: for clarity
            if df is not None:
                dfs.append(df)
        except Exception as e:
            print(f"Error retrieving {prop}: {e}")

    if not dfs:
        return None

    # Merge property dataframes
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), dfs)
    merged_df.sort_index(inplace=True)

    if updates: print(f"Generated combined data frame for {propertyCodes} at {locationCode}.") # NOTE: for clarity TODO: make actually df preview

    return merged_df

# PLOTTING FUNCTIONS

def plot_dataframe(df: pd.DataFrame, locationCode: str, start: pd.Timestamp = None, end: pd.Timestamp = None) -> None:
    """
    Plots each numeric sensor column in the DataFrame against time, 
    with line priority and unit-labeled legend entries.
    Option to normalize.
    """

    # 1. Data Preparation
    plot_df = df.copy()
    start_time = start or plot_df.index[0]
    end_time = end or plot_df.index[-1]

    # 2. Plot Setup
    fig, ax = plt.subplots(figsize=(16, 9))

    # 3. Plot Sensor Lines
    for col in plot_df.columns:
        color = sensor_colors.get(col, "black")
        z_order = 10 if col == "Oxygen (ml/l)" else 1
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=0.8, color=color, zorder=z_order)

    # 4. Axis Formatting

    ## X-Axis (Date)
    # Calculate number of full months in range
    num_months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
    num_days = (end_time - start_time).days

    # Choose appropriate locator based on range
    if num_days <= 31:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)  # weekly
    elif num_months <= 3:
        locator = mdates.DayLocator(bymonthday=[1, 15])        # twice a month
    elif num_months % 2 != 0 and num_months <= 10:
        locator = mdates.MonthLocator(interval=1, bymonthday=1)
    else:
        locator = mdates.MonthLocator(interval=2, bymonthday=1)

    formatter = mdates.DateFormatter('%b %d, %Y')
    locator.prune = None  # don't cut off start or end ticks

    # Apply x-axis formatting
    time_range = end_time - start_time
    x_padding = time_range * 0.03 
    ax.set_xlim(start_time - x_padding, end_time + x_padding)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ## Y-Axis (Sensor Values)
    y_min = 0
    y_max_data = plot_df.max().max()
    y_padding = (y_max_data - y_min) * 0.01  # Add 1% headroom
    ax.set_ylim(bottom=y_min - y_padding, top=y_max_data + y_padding)

    # 5. Labels and Title
    ax.set_xlabel("Date", labelpad=12)
    ax.set_ylabel("Sensor Value", labelpad=12)
    ax.set_title(
        f"{place[locationCode]['name']}\n"
        f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        pad=15
    )

    # 6. Annotations and Legend
    ax.plot([start_time, end_time], [hypoxia_threshold, hypoxia_threshold], color='red', linestyle='--', linewidth=1, label=f"Hypoxia Threshold ({hypoxia_threshold} ml/l)")
    ax.legend(loc="upper right")

    # 7. Grid and Show
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def subplot_all_with_oxygen(df: pd.DataFrame, locationCode: str, start: pd.Timestamp = None, end: pd.Timestamp = None) -> None:
    """
    Creates subplots showing oxygen vs each other sensor over time.
    Applies consistent styling, color, and a hypoxia threshold line.
    """

    # 1. Data Preparation
    plot_df = df.copy()
    oxygen_col = "Oxygen (ml/l)"

    if oxygen_col not in plot_df.columns:
        raise ValueError(f"'{oxygen_col}' not found in DataFrame columns.")

    start_time = start or plot_df.index[0]
    end_time = end or plot_df.index[-1]
    time_range = end_time - start_time
    x_padding = time_range * 0.03

    sensor_cols = [col for col in plot_df.columns if col != oxygen_col]
    num_sensors = len(sensor_cols)

    # 2. Plot Setup
    fig, axs = plt.subplots(num_sensors, 1, figsize=(16, 3.2 * num_sensors))
    if num_sensors == 1:
        axs = [axs]

    # 3. Plot Each Subplot
    for i, sensor_col in enumerate(sensor_cols):
        ax = axs[i]
        ax_right = ax.twinx()

        # 3.1 Plot Oxygen
        oxy_color = sensor_colors.get(oxygen_col, "royalblue")
        sensor_color = sensor_colors.get(sensor_col, "black")

        ax.plot(plot_df.index, plot_df[oxygen_col], color=oxy_color, linewidth=0.8, label=oxygen_col) # , zorder=10
        ax.set_ylabel(oxygen_col, color=oxy_color, labelpad=12)
        ax.tick_params(axis='y', labelcolor=oxy_color)
        oxy_max = plot_df[oxygen_col].max()
        ax.set_ylim(0, max(oxy_max * 1.03, hypoxia_threshold + 0.5))

        ax.plot([start_time, end_time], [hypoxia_threshold, hypoxia_threshold], color='red', linestyle='--', linewidth=1, label=f"Hypoxia Threshold ({hypoxia_threshold} ml/l)")

        # 3.2 Plot Other Sensor
        ax_right.plot(plot_df.index, plot_df[sensor_col], color=sensor_color, linewidth=0.8, label=sensor_col) # , zorder=1
        ax_right.set_ylabel(sensor_col, color=sensor_color, labelpad=12)
        ax_right.tick_params(axis='y', labelcolor=sensor_color)
        if plot_df[sensor_col].min() <= 1:
            ax_right.set_ylim(bottom=0)

        # 3.3 Match Tick Counts
        from matplotlib.ticker import MaxNLocator
        ax_right.yaxis.set_major_locator(MaxNLocator(nbins=len(ax.get_yticks()), prune=None))

        # 3.4 Subplot Title
        ax.set_title(f"{oxygen_col} vs {sensor_col}", y=1.0, pad=10, fontsize=12)

        # 3.5 Combined Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        l = ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
        l.set_zorder(100)


        # 3.6 Grid
        ax.grid(True, linestyle='--', linewidth=0.5)

    # 4. X-Axis Formatting
    num_months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
    num_days = (end_time - start_time).days

    if num_days <= 31:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
    elif num_months <= 3:
        locator = mdates.DayLocator(bymonthday=[1, 15])
    elif num_months % 2 != 0 and num_months <= 10:
        locator = mdates.MonthLocator(interval=1, bymonthday=1)
    else:
        locator = mdates.MonthLocator(interval=2, bymonthday=1)

    formatter = mdates.DateFormatter('%b %d, %Y')
    locator.prune = None

    for ax in axs:
        ax.set_xlim(start_time - x_padding, end_time + x_padding)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    axs[-1].set_xlabel("Date", labelpad=15)

    # 5. Overall Title and Layout
    fig.suptitle(
        f"{place[locationCode]['name']}\n"
        f"{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        y=0.98, x=0.51
    )
    plt.subplots_adjust(top=0.89, hspace=0.4)
    plt.show()

def subplot_all_with_time(df: pd.DataFrame, locationCode: str, start: pd.Timestamp = None, end: pd.Timestamp = None) -> None:
    """
    Subplots all properties in a data frame against time.

    Inputs:

    Outputs:
    """
    
    # 1. Data Preparation
    start_time = start or df.index[0]
    end_time = end or df.index[-1]
    time_range = end_time - start_time
    padding = time_range * 0.03  # 3% padding on each side
    sensor_cols = df.columns.to_list()

    # 2. Plot Setup
    fig, axes = plt.subplots(figsize=(16, len(sensor_cols) * 4), nrows=len(sensor_cols), ncols=1)
    if len(sensor_cols) == 1:
        axes = [axes]  # ensure iterable

    # 3. Plot Each Sensor Column
    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        color = sensor_colors[col] or "black"
        z_order = 0.8
        label = col

        # Plot the sensor line
        ax.plot(df.index, df[col], color=color, linewidth=0.7, zorder=z_order, label=label)

        # Labels and title
        ax.set_title(f"{place[locationCode]['name']} - {label}")
        ax.set_ylabel(label, labelpad=13)
        ax.set_xlabel("Date", labelpad=13)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

        # 4. X-Axis Formatting
        # Calculate how many full months are in the range
        num_months = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
        num_days = (end_time - start_time).days

        # Choose appropriate locator based on range
        if num_days <= 31:
            locator = mdates.WeekdayLocator(byweekday=mdates.MO)  # weekly
        elif num_months <= 3:
            locator = mdates.DayLocator(bymonthday=[1, 15])        # twice a month
        elif num_months % 2 != 0 and num_months <= 10:
            locator = mdates.MonthLocator(interval=1, bymonthday=1)
        else:
            locator = mdates.MonthLocator(interval=2, bymonthday=1)

        formatter = mdates.DateFormatter('%b %d, %Y')
        locator.prune = None  # don't cut off start or end ticks
        
        ax.xaxis.set_major_formatter(formatter)


        # 5. Add Hypoxia Threshold (if oxygen)
        if "oxygen" in col.lower():
            ax.plot([start_time, end_time], [hypoxia_threshold, hypoxia_threshold], color='red', linestyle='--', linewidth=1, label=f"Hypoxia Threshold ({hypoxia_threshold} ml/l)")
            legend = ax.legend()
            legend.set_zorder(50)

    # 6. Shared X-Ticks Across Subplots
    shared_xticks = axes[0].get_xticks()
    for ax in axes[1:]:
        ax.set_xticks(shared_xticks)

    # 7. Title and Layout
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

    Inputs:
        df1 (pd.DataFrame): First location's data (timestamp as index).
        df2 (pd.DataFrame): Second location's data (timestamp as index).
        sensor_cols (list[str]): List of shared propertyCodes to plot.
        locationCode1 (str): Location code for first dataset (e.g., 'FGPPN').
        locationCode2 (str): Location code for second dataset (e.g., 'FDPP').

    Outputs:
        None
    """
    # Check time period in each dataframe (i.e. for missing data)
    min_time = min(df1.index[0], df2.index[0])
    max_time = max(df1.index[-1], df2.index[-1])

    # Define figure and axes for subplots
    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(16, len(sensor_cols)*4))
    

    if len(sensor_cols) == 1:
        axes = [axes]  # Ensure axes is always iterable

    for ax, col in zip(axes, sensor_cols):

        label = col #subplot title

        # color options: seagreen & tomato, royalblue & crimson, teal & darkorange, Slateblue & firebrick. steelblue & indianred
        ax.plot(df2.index, df2[col], label=f"{place[locationCode2]['name']} - {place[locationCode2]['mountDepth']}m", linewidth=0.8, color="firebrick")
        ax.plot(df1.index, df1[col], label=f"{place[locationCode1]['name']} - {place[locationCode1]['mountDepth']}m", linewidth=0.8, color="slateblue")

        ax.set_title(f"{label}", fontsize=12)
        ax.set_ylabel(col, labelpad=12)
        ax.set_xlabel("Time", labelpad=12)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()

        # Compute dynamic padding (e.g. 3% of full time range)
        time_range = max_time - min_time
        padding = time_range * 0.03  # 3% padding on each side
        ax.set_xlim(min_time - padding, max_time + padding) # Make sure all axes span same range

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'),)

    start_time = df1.index[0]
    end_time = df1.index[-1]

    fig.suptitle(
        f"{place[locationCode2]['name']} vs {place[locationCode1]['name']}\n{start_time.strftime('%B %d, %Y')} to {end_time.strftime('%B %d, %Y')}",
        fontweight='bold',
        y=0.97,
        x=0.51
    )

    plt.subplots_adjust(top=0.89, hspace=0.45)
    plt.show()

# TODO: set y padding
def plot_with_twin_y_axis_for_outlier(df: pd.DataFrame, locationCode: str) -> None:
    """
    Plots time series data using a twin y-axis. The highest-magnitude parameter is plotted on the
    right y-axis using its defined color and label; all others are on the left.

    Inputs:
        df (pd.DataFrame): Time-indexed DataFrame of numeric sensor columns.
        locationCode (str): Optional label to include in title.
        title (str): Optional custom title.
        sensor_info (dict): Maps propertyCode -> {label, color}.

    Outputs:
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
    fig, ax_left = plt.subplots(figsize=(16, 9))
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

def plot_dataframe_plotly(df: pd.DataFrame, locationCode: str) -> None:
    """
    Creates an interactive Plotly line chart for all numeric sensor columns in the DataFrame.

    Inputs:
        df (pd.DataFrame): DataFrame with datetime index and sensor columns.
        locationCode (str): Optional identifier used in the title.

    Outputs:
        None
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
            # 'xanchor': 'center',
            # 'yanchor': 'top'
        }  
    )

    fig.show()
