import requests
import os
import time
import re
import json
import csv
import datetime
import pandas as pd
import numpy as np
import random
import math

base_folder = r'C:\Users\georg\OneDrive\Documents\MIS581\Data'
usage_folder = os.path.join(base_folder, 'usage_data')
usage_folder_contents = [os.path.join(usage_folder, file) for file in os.listdir(usage_folder)]
url_json_location = os.path.join(base_folder, 'urls.json')
timestamp_format = '%Y-%m-%dT%H:%M'

# Send get request without auth, but with automatic retry logic.
def request_get(url, retries = 0):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception('Bad status code')
        return r
    except Exception as e:
        if retries > 5:
            print(f'Unable to get data for {url}')
            return None
        else:
            print(f'request failed for url: {url}. Retry {retries} of 5')
            time.sleep(10)
            return request_get(url, retries = retries + 1)

# Get all NC CSV URLs
def reload_csv_urls():
    def folder_filter(x):
        return x['type'] == 'folder'

    def nc_csv_filter(x):
        return x['type'] == 'csv' and x['name'].startswith('NC') and x['name'] not in usage_folder_contents
    
    def get_relevant_items(base_url, filter_function):
        r = request_get(base_url)
        if r == None:
            return []
        payload = r.json()
        payload_folders = list(filter(filter_function, payload))
        folder_names = list(map(lambda x: x['name'], payload_folders))
        folder_names = sorted(folder_names)
        return folder_names

    base_url = 'https://sciduct.bii.virginia.edu/fs/file/resources/residential_energy_profiles/'
    current_urls = [base_url]
    for level in [0, 2]:
        new_urls = []
        print(f'Level {level}')
        print(f'Number of urls at level: {len(current_urls)}')
        filter_function = folder_filter if level < 2 else nc_csv_filter
        # Within each date folder, there will be a folder marked 37. This is the North Carolina state FIPS code,
        # and it is where all state-level data will be located (United States Census Bureau, 2023).
        # For each csv, actually downloading the file rather than retrieving metadata can be performed by appending
        # this query param, per instructions by D. Machi (personal communication, January 5, 2024).
        end_of_url = '/37/' if level < 2 else '?http_accept=application/octet-stream'
        for url_num, url in enumerate(current_urls):
            if (url_num % 10 == 0):
                print(f'Processing url {url_num}')
            new_urls.extend([os.path.join(url, folder) + end_of_url for folder in get_relevant_items(url, filter_function)])
        current_urls = new_urls
    # Write URLs to location
    with open(url_json_location, 'w') as f:
        f.write(json.dumps(new_urls, indent=2))

# Generate county --> weather station table
def get_counties_per_weather_station():
    def normalized_position(latitude, longitude):
        latitude *= (2 * np.pi / 360)
        longitude *= (2 * np.pi / 360)
        return {   
            "X": np.cos(longitude) * np.cos(latitude), 
            "Y": np.sin(longitude) * np.cos(latitude), 
            "Z": np.sin(latitude)
        }
    # Weather station coordinates
    weather_data_raw_folder = os.path.join(base_folder, 'weather_data_raw')
    weather_station_positions = {}
    for path in [os.path.join(weather_data_raw_folder, file) for file in os.listdir(weather_data_raw_folder)]:
        df = pd.read_csv(path)
        weather_station_positions[df.loc[0, 'STATION']] = normalized_position(df.loc[0, 'LATITUDE'], df.loc[0, 'LONGITUDE'])
        weather_station_positions[df.loc[0, 'STATION']]['NAME'] = df.loc[0, 'NAME']
    weather_station_positions = pd.DataFrame(weather_station_positions).transpose()

    county_coordinates = pd.read_csv(os.path.join(base_folder, 'county_coordinates.csv'))
    nc_counties = county_coordinates.loc[county_coordinates.USPS == 'NC', ('GEOID', 'NAME', 'INTPTLAT', 'INTPTLONG')]
    nc_counties = nc_counties.rename(columns = {'GEOID': 'FIPS'})
    def add_absolute_position(row):
        absolute_position = normalized_position(row.loc['INTPTLAT'], row.loc['INTPTLONG'])
        row['X'] = absolute_position['X']
        row['Y'] = absolute_position['Y']
        row['Z'] = absolute_position['Z']
        print(row)
        return row

    nc_counties = nc_counties.apply(lambda row: add_absolute_position(row), axis=1)
    
    def find_closest_station(county):
        def find_distance_to_county(station):
            return np.sum(np.square([
                county.loc['X'] - station.loc['X'],
                county.loc['Y'] - station.loc['Y'],
                county.loc['Z'] - station.loc['Z']
            ]))
        return weather_station_positions.apply(lambda station: find_distance_to_county(station), axis=1).idxmin()
    
    nc_counties['STATION'] = nc_counties.apply(lambda county: find_closest_station(county), axis=1)

    # Demonstrate 
    for station in list(set(nc_counties.loc[:, 'STATION'])):
        print(f'Station: {weather_station_positions.loc[station, "NAME"]}')
        print('Counties:')
        print(list(nc_counties.loc[nc_counties.STATION == station, 'NAME']))

    nc_counties.index = nc_counties.FIPS
    nc_counties = nc_counties.loc[:, ('NAME', 'STATION')]
    nc_counties.to_csv(os.path.join(base_folder, 'nc_county_info.csv'))
        
# Generate weather data table
def process_weather_data():
    weather_data_raw_folder = os.path.join(base_folder, 'weather_data_raw')
    hour_timestamps = [(datetime.datetime(2014, 1, 1, 0, 0) + datetime.timedelta(hours=n))
                    for n in range(24*365)]
    all_weather_data = pd.DataFrame()
    for file in os.listdir(weather_data_raw_folder):
        print(file)
        path = os.path.join(weather_data_raw_folder, file)
        weather_data = pd.read_csv(path)
        station_id = int(weather_data.loc[0, 'STATION'])
        weather_data = weather_data.loc[(weather_data.REPORT_TYPE == 'FM-15') | (weather_data.REPORT_TYPE == 'FM-16'), 
                                        ('DATE', 'HourlyDryBulbTemperature', 'HourlyWetBulbTemperature',
                                         'HourlyRelativeHumidity', 'HourlyPrecipitation',
                                         'HourlyStationPressure', 'HourlyWindSpeed', 'HourlySkyConditions'
                                         )]
        weather_data = weather_data.rename(columns = {
            'DATE': 'timestamp',
            'HourlyDryBulbTemperature': 'dry_bulb_temperature',
            'HourlyWetBulbTemperature': 'wet_bulb_temperature',
            'HourlyRelativeHumidity': 'humidity',
            'HourlyPrecipitation': 'precipitation',
            'HourlyStationPressure': 'air_pressure',
            'HourlyWindSpeed': 'wind_speed',
            'HourlySkyConditions': 'sky_conditions'
        })
        
        observation_indices = {k.strftime(timestamp_format): {'ROW_NUM': None} for k in hour_timestamps}
        for row_num in range(weather_data.shape[0]):
            timestamp = datetime.datetime.strptime(
                weather_data['timestamp'].iloc[row_num], timestamp_format + ':%S')
            rounded_timestamp = datetime.datetime(timestamp.year, 
                                                  timestamp.month, 
                                                  timestamp.day, 
                                                  timestamp.hour, 
                                                  0).strftime(timestamp_format)
            if observation_indices[rounded_timestamp]['ROW_NUM'] == None:
                observation_indices[rounded_timestamp]['ROW_NUM'] = row_num

        weather_hourly = pd.DataFrame(observation_indices).transpose()
        weather_hourly = weather_hourly.apply(lambda row: weather_data.iloc[row.loc['ROW_NUM']] 
                             if (row.loc['ROW_NUM'] != None)
                             else pd.Series(), axis=1)
        weather_hourly.index = [f'{station_id}_{i}' for i in list(weather_hourly.index)]
        all_weather_data = pd.concat([all_weather_data, weather_hourly])

    all_weather_data.index.name = 'station_hour'
    all_weather_data.to_csv(os.path.join(base_folder, 'weather_data.csv'))


def get_selected_home_ids():
    with open(url_json_location) as f:
        csv_urls = json.load(f)
    csv_urls = list(filter(lambda x: x.__contains__('2014_01_01'), csv_urls))
    home_ids_dict = {}
    for csv_url_num, csv_url in enumerate(csv_urls):
        print(f'Processing url {csv_url_num}')
        r = request_get(csv_url)
        home_ids = [row.split(',')[0] for row in r.text.split('\n')][1:]
        fips_code = int('37' + re.findall(r'(?<=NC)\d+(?=-)', csv_url)[0])
        home_ids_dict[fips_code] = home_ids

    home_ids_flattened = [[fips, home_id] for fips in home_ids_dict.keys() for home_id in home_ids_dict[fips]]
    random.shuffle(home_ids_flattened)
    selected_home_ids = home_ids_flattened[0:10000]
    selected_home_ids = {fips: [x[1] for x in selected_home_ids if x[0] == fips] 
                        for fips 
                        in list(set([x[0] for x in home_ids_flattened]))}
    with open(os.path.join(base_folder, 'selected_home_ids.json'), 'w') as f:
        f.write(json.dumps(selected_home_ids, indent=2))


# Get CSV data, combine it with matching weather data, and dump it into usage_data folder.
def download_and_process_csvs():

    def process_row(row_text, csv_columns_dict, rename_dict, fips_code):
        row_values = row_text.split(',')
        if len(row_values) < len(csv_columns_dict.keys()):
            return []
        row_hourly_list = [None for n in range(24)]
        for hour in range(24):
            # home id, timestamp
            row_hourly_list[hour] = [
                row_values[csv_columns_dict['hid']],
                fips_code,
                datetime.datetime(2014, 1, 1, hour, 0).strftime(timestamp_format)
            ]
            for key in rename_dict.keys():
                row_hourly_list[hour].append(float(row_values[csv_columns_dict[f'{key}_{hour + 1}']]))
                if key.endswith('_wh'):
                    row_hourly_list[hour][-1] /= 1000
        
        return row_hourly_list

    #reload_csv_urls()
    with open(url_json_location) as f:
        csv_urls = json.load(f)

    with open(os.path.join(base_folder, 'selected_home_ids.json')) as f:
        home_ids = json.load(f)

    # Extract data for each NC CSV
    print(f'Number of CSV URLs: {len(csv_urls)}')

    rename_dict = {
        'total_kwh': 'total_kw',
        'hvac_kwh': 'hvac_kw',
        'hoth2o_kwh': 'hot_water_kw',
        'refr_kwh': 'refrigerator_kw',
        'light_kwh': 'light_kw',
        'misc_kwh': 'misc_kw',
        'dw_wh': 'dishwasher_kw',
        'laundry_wh': 'laundry_kw',
        'cook_wh': 'cooking_kw'
    }
    usage_csv_headers = ['home_id', 'fips_code', 'timestamp'] + list(rename_dict.values())
    csv_dump_location = os.path.join(base_folder, 'usage_data', 'hourly_usage.csv')

    for csv_url_num, csv_url in enumerate(csv_urls[7000:]):
        fips_code = '37' + re.findall(r'(?<=NC)\d+(?=-)', csv_url)[0]
        url_date = re.findall(r'(?<=residential_energy_profiles/).+(?=/37)', csv_url)[0]
        csv_dump_location = os.path.join(base_folder, 'usage_data', f'hourly_usage_{url_date}.csv')

        print(f'Processing url {csv_url_num}, Date {url_date}, FIPS {fips_code}, at time {datetime.datetime.now().strftime(timestamp_format + ":%S.%f")}')
        r = request_get(csv_url)
        # Process and filter response lines
        response_lines = r.text.split('\n')
        csv_columns = list(filter(lambda x: x != '', response_lines[0].split(',')))
        csv_columns_dict = {key: index for index, key in enumerate(csv_columns)}
        data_lines = [row for row in response_lines[1:] if row.split(',')[csv_columns_dict['hid']] in home_ids[fips_code]]
        # Convert into hourly rows
        hourly_rows = []
        for row in data_lines:
            hourly_rows.extend(process_row(row, csv_columns_dict, rename_dict, fips_code))

        with open(csv_dump_location, 'a+', newline='') as obj:
            csvwriter = csv.writer(obj)
            csvwriter.writerow(usage_csv_headers)
            csvwriter.writerows(hourly_rows)

def home_to_group_ids():
    with open(os.path.join(base_folder, 'selected_home_ids.json')) as f:
        home_ids = json.load(f)
    home_to_group_dict = {}
    for fips_code in home_ids.keys():
        ids_in_county = home_ids[fips_code]
        random.shuffle(ids_in_county)
        home_to_group_dict.update({ids_in_county[n]: f'{fips_code}_{math.floor(n/50)}' for n in range(len(ids_in_county))})
    with open(os.path.join(base_folder, 'home_to_group_dict.json'), 'w') as f:
        f.write(json.dumps(home_to_group_dict, indent=2))
            
# Join usage data with weather data, and dump it into the usage_data_joined folder.
def join_weather():
        
        def get_oktas_from_sky_conditions(sky_conditions):
            if not isinstance(sky_conditions, str):
                return None
            oktas = re.findall(r'(?<=:)\d+', sky_conditions)
            return int(oktas[0]) if (len(oktas) > 0) else None

        nc_counties = pd.read_csv(os.path.join(base_folder, 'nc_county_info.csv'))
        nc_counties.FIPS = nc_counties.FIPS.astype('str')
        nc_counties = nc_counties.assign(station = nc_counties.STATION.astype('str'))

        with open(os.path.join(base_folder, 'home_to_group_dict.json')) as f:
            home_to_group_dict = json.load(f)
            valid_home_ids = list(home_to_group_dict.keys())

        weather_data = pd.read_csv(os.path.join(base_folder, 'weather_data.csv'))
        weather_data.index = weather_data.station_hour

        for file in [f for f in os.listdir(os.path.join(base_folder, 'usage_data')) if f.startswith('hourly_usage')]:
            print(f'Processing file {file} at time {datetime.datetime.now().strftime(timestamp_format + ":%S.%f")}')
            # Get dataframe for weather data by county for matching day
            file_date = file.replace('.csv', '').replace('hourly_usage_', '').replace('_', '-')
            weather_data_for_day = weather_data.loc[list(map(lambda x: x.__contains__(file_date), weather_data['station_hour'])), ]
            weather_data_for_day = weather_data_for_day.assign(station = list(map(lambda x: x.split('_')[0], weather_data_for_day['station_hour'])))
            weather_data_by_county = pd.merge(weather_data_for_day, nc_counties, how='left', on='station')
            weather_data_by_county = weather_data_by_county.assign(
                timestamp = list(map(lambda x: x.split('_')[1], weather_data_by_county['station_hour'])),                          
                oktas = list(map(lambda x: get_oktas_from_sky_conditions(x), weather_data_by_county['sky_conditions'])),
                fips_code = weather_data_by_county['FIPS']
            )
            weather_data_by_county.drop(labels=['station_hour', 'NAME', 'sky_conditions', 'station', 'STATION', 'FIPS'], axis=1, inplace=True)
            

            # Processed usage data
            usage_df = pd.read_csv(os.path.join(base_folder, 'usage_data', file))
            usage_df.fips_code = usage_df.fips_code.astype('str')
            usage_df = usage_df.loc[usage_df['fips_code'].str.startswith('37'), ]
            usage_df = usage_df.loc[usage_df['home_id'].isin(valid_home_ids), ]
            # Timestamp was set wrong during parsing code; this corrects that code
            usage_df.timestamp = usage_df['timestamp'].str.replace('2014-01-01', file_date)
            usage_df = usage_df.assign(
                group_id = list(map(lambda home_id: home_to_group_dict[str(home_id)], usage_df.home_id)),
                hour_of_day = list(map(lambda timestamp: re.findall(r'(?<=T)\d+(?=:)', timestamp)[0], usage_df.timestamp)) # Jankier but faster than datetime parsing
            )
            usage_df = usage_df.sort_values(by=['hour_of_day', 'home_id'])
            # Join tables, dump to folder
            all_df = pd.merge(usage_df, weather_data_by_county, how='left', on=['fips_code', 'timestamp'])
            all_df.to_csv(os.path.join(base_folder, 'usage_data_joined', file), index=False)


def merge_csvs():
    for file_num, file in enumerate(os.listdir(os.path.join(base_folder, 'usage_data_joined'))):
        print(f'Processing file {file} at time {datetime.datetime.now().strftime(timestamp_format + ":%S.%f")}')
        if file_num < 90:
            continue
        with open(os.path.join(base_folder, 'usage_data_joined', file), mode ='r') as obj:
            file_lines = obj.read().splitlines()
            file_lines = list(map(lambda x: x.split(','), file_lines))

        with open(os.path.join(base_folder, 'MIS581_final_project_data_raw.csv'), 'a+', newline='') as obj:
            csvwriter = csv.writer(obj)
            if file_num == 0:
                csvwriter.writerow(file_lines[0])
            csvwriter.writerows(file_lines[1:])

def filter_sort_csv():
    usage_df = pd.read_csv(os.path.join(base_folder, 'MIS581_final_project_data_raw.csv'))
    usage_df = usage_df.dropna()

    usage_df = usage_df.loc[
        ~usage_df['air_pressure'].astype('str').str.contains('s')
        & ~usage_df['dry_bulb_temperature'].astype('str').str.contains('s')
        & ~usage_df['humidity'].astype('str').str.contains('s')
        & ~usage_df['precipitation'].astype('str').str.contains('s')
        & ~usage_df['precipitation'].astype('str').str.contains('T')
        & ~usage_df['wet_bulb_temperature'].astype('str').str.contains('s')
        & ~usage_df['wind_speed'].astype('str').str.contains('s')
        & ~usage_df['oktas'].astype('str').str.contains('s')
    ]
    usage_df = usage_df.sort_values(by=['group_id', 'home_id', 'timestamp'])

    usage_df.to_csv(os.path.join(base_folder, 'MIS581_final_project_data_filtered.csv'))
# Main/execution section
#join_weather()
filter_sort_csv()