import pandas as pd
import os
import numpy as np
import random
import keras.api._v2.keras as keras
from keras import layers
from keras.utils import to_categorical
import datetime
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from IPython.display import display


independent_vars_numeric = [
    'air_pressure', 'dry_bulb_temperature', 'humidity', 
    'precipitation_normal', 'wet_bulb_temperature', 'wind_speed', 'oktas'
]

def current_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

print(f'Loaded libraries! {current_time()}')


# Takes dataframe & splits it into I/O formats that the neural network can understand.
def split_input_output(df_all, df_numpy, usage_column_index_dict, downsample, model_categories):
    power_categories = [
        'hvac_kw', 'hot_water_kw', 'refrigerator_kw', 'light_kw',
        'misc_kw', 'dishwasher_kw', 'laundry_kw', 'cooking_kw'
    ] if model_categories else ['total_kw']
    input_list = np.concatenate(
        (
            df_numpy[:, [usage_column_index_dict[col] for col in independent_vars_numeric]],
            to_categorical(df_numpy[:, usage_column_index_dict['hour_of_day']])
        ), axis = 1
    )
    output_list = (df_numpy[:, [usage_column_index_dict[col] for col in power_categories]])


    for column_num, column in enumerate(independent_vars_numeric):
        input_list[:, column_num] = (input_list[:, column_num] - np.mean(df_all.loc[:, column])) / np.std(df_all.loc[:, column])

    # Bounds output to [0, 1], which makes sense for a sigmoid activation function.
    # Also, normalizing on total_kw rather than on the column's own maximum keeps the loss function from getting distorted.
    for column_num, column in enumerate(power_categories):
        output_list[:, column_num] = output_list[:, column_num] / np.max(df_all.loc[:, 'total_kw'])

    input_list = np.asarray(input_list).astype(np.float32)
    output_list = np.asarray(output_list).astype(np.float32)

    if downsample != 1:
        input_list = input_list[range(0, len(input_list), downsample), ]
        output_list = output_list[range(0, len(output_list), downsample), ]


    return input_list, output_list

# Evaluate data on FFNN model.
def train_model(train_in, train_out, test_in, test_out, df_in, df_all, epochs):
    activation = 'sigmoid'
    vb = 2
    batch_size = 32
    model = keras.Sequential()
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(64, activation=activation))
    model.add(layers.Dense(32, activation=activation))
    model.add(layers.Dense(train_out.shape[1], activation=activation))
    model.compile(
            loss = 'mean_squared_error',
            optimizer = keras.optimizers.Adam(learning_rate=0.01),
            metrics=['mean_squared_error']
        )

    # Train model on train/test data
    model.fit(train_in, train_out, 
        batch_size=batch_size, epochs=epochs, verbose=vb,
        validation_data=(test_in, test_out))
    # Get predictions for dataframe, assign into dataframe
    predictions = model.predict(df_in)
    predictions = predictions * np.max(df_all.total_kw)
    predictions = np.sum(predictions, axis=1)
    df_all = df_all.assign(total_kw_prediction = predictions)
    df_all = df_all.astype({'total_kw': np.float64})
    # Group by timestamp
    df_by_timestamp = df_all.loc[:, ['total_kw', 'total_kw_prediction', 'timestamp']].groupby(by=['timestamp']).mean()
    df_by_timestamp = df_by_timestamp.assign(timestamp = df_by_timestamp.index)
    # Build top-level regression model
    df_by_timestamp_train = df_by_timestamp.loc[df_by_timestamp.timestamp < '2014-09-01T00:00', ]
    regression = smf.ols('total_kw ~ total_kw_prediction - 1', data=df_by_timestamp_train).fit()
    df_by_timestamp = df_by_timestamp.assign(total_kw_fitted_aggregate = regression.predict(df_by_timestamp.total_kw_prediction))

    train_r2 = df_by_timestamp.loc[df_by_timestamp.timestamp < '2014-09-01T00:00', ].corr().loc['total_kw', 'total_kw_fitted_aggregate']**2
    valid_r2 = df_by_timestamp.loc[df_by_timestamp.timestamp >= '2014-09-01T00:00', ].corr().loc['total_kw', 'total_kw_fitted_aggregate']**2
    rmse = np.sqrt(np.mean((df_by_timestamp.loc[df_by_timestamp.timestamp < '2014-09-01T00:00', 'total_kw'] - 
                            df_by_timestamp.loc[df_by_timestamp.timestamp < '2014-09-01T00:00', 'total_kw_fitted_aggregate'])**2))
    return {
        "model": model, 
        "regression": regression,
        "df_by_timestamp": df_by_timestamp,
        "train_r2": train_r2,
        "valid_r2": valid_r2,
        "rmse": rmse
    }

def main(reload_home_data=False, reload_group_data=False, execute_home_models=False, execute_group_models=False):
    # Load data
    if reload_home_data:
        base_folder = r'C:\Users\georg\OneDrive\Documents\MIS581\Data'
        usage_df = pd.read_csv(os.path.join(base_folder, 'MIS581_final_project_data_filtered.csv'))
        print(f'Loaded usage_df! {current_time()}')
        usage_df = usage_df.assign(precipitation_normal = np.arcsinh(usage_df.precipitation / np.mean(usage_df.precipitation)))
        usage_df_sampled = usage_df.sample(frac=0.05) # using full dataframe causes computer to crash due to lack of RAM, so acceptably large sample (~3 million rows) was used
        usage_df_sampled_numpy = np.array(usage_df_sampled)
    if reload_group_data:
        #usage_columns = list(usage_df.columns)
        usage_columns = ['Unnamed: 0', 'home_id', 'fips_code', 'timestamp', 'total_kw', 'hvac_kw', 'hot_water_kw', 'refrigerator_kw', 'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw', 'cooking_kw', 'group_id', 'hour_of_day', 'air_pressure', 'dry_bulb_temperature', 'humidity', 'precipitation', 'wet_bulb_temperature', 'wind_speed', 'oktas', 'precipitation_normal']
        usage_column_index_dict = {usage_columns[n]: n for n in range(len(usage_columns))}
        usage_df_numpy = np.array(usage_df)
        np.save(r'C:\Users\georg\OneDrive\Documents\MIS581\Data\usage_df_numpy.npy', usage_df_numpy)
        usage_df_numpy_split_by_group = np.array_split(usage_df_numpy, np.where(np.diff(list(map(lambda x: int(x.replace('_', '')), usage_df_numpy[:, usage_column_index_dict['group_id']]))) != 0)[0] + 1)
        for group_array_num, group_array in enumerate(usage_df_numpy_split_by_group):
            group_df = pd.DataFrame(group_array, columns=usage_columns)
            group_df_by_timestamp = group_df.groupby(by=['timestamp']).mean()
            group_df_by_timestamp = group_df_by_timestamp.assign(
                timestamp = group_df_by_timestamp.index
            )
            group_df_by_timestamp.group_id = group_df.loc[0, 'group_id']
            print(f'Group num: {group_array_num}, {current_time()}, group id: { group_df.loc[0, "group_id"]}')
            group_df_by_timestamp = group_df_by_timestamp.loc[:, usage_columns]
            if group_array_num == 0:
                usage_df_numpy_by_group = np.array(group_df_by_timestamp)
            else:
                usage_df_numpy_by_group = np.concatenate((usage_df_numpy_by_group,  np.array(group_df_by_timestamp)), axis=0)
        usage_df_by_group = pd.DataFrame(usage_df_numpy_by_group, columns=usage_columns)
        usage_df_by_group.to_pickle(r'C:\Users\georg\OneDrive\Documents\MIS581\Data\usage_df_by_group.pkl')
    else:
        usage_columns = ['Unnamed: 0', 'home_id', 'fips_code', 'timestamp', 'total_kw', 'hvac_kw', 'hot_water_kw', 'refrigerator_kw', 'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw', 'cooking_kw', 'group_id', 'hour_of_day', 'air_pressure', 'dry_bulb_temperature', 'humidity', 'precipitation', 'wet_bulb_temperature', 'wind_speed', 'oktas', 'precipitation_normal']
        usage_column_index_dict = {usage_columns[n]: n for n in range(len(usage_columns))}
        usage_df_by_group = pd.read_pickle(r'C:\Users\georg\OneDrive\Documents\MIS581\Data\usage_df_by_group.pkl')
        usage_df_numpy_by_group = np.array(usage_df_by_group)
        

    #usage_df_by_group = usage_df.loc[['group_id', 'timestamp', 'hour_of_day'] + independent_vars_numeric + dependent_vars].groupby(by=["group_id", "timestamp"]).mean()

    # Get train & test home IDs
    timestamp_split = '2014-09-01T00:00'

    # Orchestrator for split_input_output
    def get_model_data(df, df_numpy, id_field, usage_column_index_dict, downsample, model_categories):
        ids_unique = list(set(df.loc[:, id_field]))
        random.shuffle(ids_unique)
        train_ids = ids_unique[0:math.floor(0.7 * len(ids_unique))]
        test_ids = ids_unique[math.floor(0.7 * len(ids_unique)):]
        # Get train & test datasets
        non_validation_df = df.loc[df.timestamp < timestamp_split, ]
        train = np.array(non_validation_df.loc[non_validation_df[id_field].isin(train_ids), ])
        test = np.array(non_validation_df.loc[non_validation_df[id_field].isin(test_ids), ])
        del non_validation_df

        # Get train & test numpy arrays
        train_in, train_out = split_input_output(df,
                                                    train, 
                                                    usage_column_index_dict=usage_column_index_dict,
                                                    downsample=downsample, 
                                                    model_categories=model_categories)
        test_in, test_out = split_input_output(df,
                                                    test, 
                                                    usage_column_index_dict=usage_column_index_dict, 
                                                    downsample=downsample,
                                                    model_categories=model_categories)
        
        df_in, df_out = split_input_output(df,
                                            df_numpy, 
                                            usage_column_index_dict=usage_column_index_dict,
                                            downsample = 1, 
                                            model_categories=model_categories)
        
        return train_in, train_out, test_in, test_out, df_in

    if execute_group_models:
        # Group, total usage model
        train_in, train_out, test_in, test_out, df_in = get_model_data(
            usage_df_by_group,
            usage_df_numpy_by_group, 
            'group_id', 
            usage_column_index_dict, 
            downsample=20, 
            model_categories=False
        )
        group_total_model_results = train_model(train_in, 
            train_out, 
            test_in, 
            test_out,
            df_in, 
            usage_df_by_group,
            epochs=20
        )
        
        # Group, per-category model
        train_in, train_out, test_in, test_out, df_in = get_model_data(
            usage_df_by_group,
            usage_df_numpy_by_group, 
            'group_id', 
            usage_column_index_dict, 
            downsample=20, 
            model_categories=True
        )
        group_category_model_results = train_model(train_in, 
            train_out, 
            test_in, 
            test_out,
            df_in, 
            usage_df_by_group,
            epochs=20
        )

    if execute_home_models:
        # Home, total model
        train_in, train_out, test_in, test_out, df_in = get_model_data(
            usage_df_sampled,
            usage_df_sampled_numpy, 
            'home_id', 
            usage_column_index_dict, 
            downsample=20, 
            model_categories=True
        )
        home_total_model_results = train_model(train_in, 
            train_out, 
            test_in, 
            test_out,
            df_in, 
            usage_df_sampled,
            epochs=20
        )

        # Home, per-category model
        train_in, train_out, test_in, test_out, df_in = get_model_data(
            usage_df_sampled,
            usage_df_sampled_numpy, 
            'home_id', 
            usage_column_index_dict, 
            downsample=20, 
            model_categories=False
        )
        home_category_model_results = train_model(train_in, 
            train_out, 
            test_in, 
            test_out,
            df_in, 
            usage_df_sampled,
            epochs=20
        )



        model_results = [group_total_model_results, group_category_model_results, home_total_model_results, home_category_model_results]
        granularities = ['group_total', 'group_category', 'home_total', 'home_category']
        
        # Metrics
        summary_stats = pd.DataFrame([[m['train_r2'], m['valid_r2'], m['rmse']] for m in model_results], columns = ['train_r2', 'valid_r2', 'rmse'])
        summary_stats.index = granularities
        print(summary_stats)

        model_dfs = [m['df_by_timestamp'].loc[m['df_by_timestamp'].timestamp.isin(home_category_model_results['df_by_timestamp'].timestamp), 'total_kw_fitted_aggregate'] for m in model_results]

        # Correlation matrix
        comparison_frame = pd.DataFrame.from_dict({
            "group_total": model_dfs[0],
            "group_category": model_dfs[1],
            "home_total": model_dfs[2],
            "home_category": model_dfs[3]
            })
        print(comparison_frame.corr())

        # Regression summaries
        for model_num, model in enumerate(model_results):
            print(f'\n\nTopline regression for {granularities[model_num]}\n\n')
            print(model['regression'].summary())

        # Scatterplots
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(7.5, 7.5)
        fig.tight_layout(pad = 3) # stops text from overlapping
        for axis_num, axis in enumerate(axes.flatten()):
            axis.scatter(model_results[axis_num]['df_by_timestamp'].total_kw_fitted_aggregate, 
                         model_results[axis_num]['df_by_timestamp'].total_kw,
                         marker='x')
            axis.set_title(granularities[axis_num])
            axis.set_xlabel('Fitted average power (kW)')
            axis.set_ylabel('Real average power (kW)')
        plt.show()




    print(5)




        #return model
    
    
    #model_train(usage_df, usage_df_numpy, 'home_id', usage_columns, 1024)
    #model_train(usage_df_by_group, usage_df_numpy_by_group, 'group_id', usage_columns, 32)
    

main(reload_home_data=True, reload_group_data=False, execute_group_models=True, execute_home_models=True)
#print(5)