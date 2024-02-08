#############
# General functions & libraries
#############

library(PerformanceAnalytics)
library(ggplot2)
require(MASS)
library(dplyr)
library(stats)

remove_zeros <- function(value) {
  return(ifelse(value == 0, NA, value))
}

ratio_zeros <- function(value) {
  return(mean(value == 0))
}

normalize <- function(df_column) {
  avg <- as.numeric(sapply(df_column %>% mutate_all(remove_zeros), mean, na.rm=TRUE))
  return (asinh(df_column / avg))
}

#############
# Load usage data frame
#############

weather_variables <- c("air_pressure", "dry_bulb_temperature", "humidity", 
                       "precipitation_normal", "wet_bulb_temperature",
                       "wind_speed", "oktas")

numeric_columns <- c('total_kw', 'hvac_kw', 'hot_water_kw', 'refrigerator_kw', 
                     'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                     'cooking_kw', 'air_pressure', 'dry_bulb_temperature',
                     'humidity', 'precipitation', 'precipitation_normal',
                     'wet_bulb_temperature', 'wind_speed', 'oktas')

usage_df <- read.csv("C:/Users/georg/OneDrive/Documents/MIS581/Data/MIS581_final_project_data_filtered.csv")
usage_df$X <- NULL
head(usage_df)

# Shows homoskedasticity
plot_usage_model <- function(df, target) {
  df[paste(target, 'log', sep = '_')] <- log(df[target] %>% mutate_all(remove_zeros))
  usage_model_weather_log <- lm(eval(parse(text = paste(
    target,
    "_log ~ ",
    'hour_of_day + ', 
    paste(weather_variables, collapse = " + "),
    sep = ""
  ))), data = df)
  
  model_df <- data.frame(
    residuals = usage_model_weather_log$residuals,
    fitted_buckets = ntile(usage_model_weather_log$fitted.values, 10)
  )
  ggplot(model_df, aes(x = fitted_buckets, y = residuals)) + ggtitle(target) + geom_boxplot()
}

usage_df$precipitation_normal <- normalize(usage_df['precipitation'])$precipitation
summary_usage <- data.frame(
  mean=                    sapply(usage_df[, numeric_columns], mean, na.rm=TRUE),
  sd=                      sapply(usage_df[, numeric_columns], sd, na.rm=TRUE),
  min=                     sapply(usage_df[, numeric_columns], min, na.rm=TRUE),
  max=                     sapply(usage_df[, numeric_columns], max, na.rm=TRUE),
  median=                  sapply(usage_df[, numeric_columns], median, na.rm=TRUE),
  length=                  sapply(usage_df[, numeric_columns], length),
  skewness=                sapply(usage_df[, numeric_columns], skewness, na.rm=TRUE),
  excess_kurtosis=         sapply(usage_df[, numeric_columns], kurtosis, na.rm=TRUE, method = "moment") - 3,
  zero_ratio=              sapply(usage_df[, numeric_columns], ratio_zeros),
  skewness_log=            sapply(log(usage_df[, numeric_columns] %>% mutate_all(remove_zeros)), skewness, na.rm=TRUE),
  excess_kurtosis_log=     sapply(log(usage_df[, numeric_columns] %>% mutate_all(remove_zeros)), kurtosis, na.rm=TRUE, method = "moment") - 3,
  miss_val=                sapply(usage_df[, numeric_columns], function(x) sum(length(which(is.na(x)))))
)
plot_usage_model(usage_df, 'total_kw')
usage_df$precipitation <- NULL;
numeric_columns <- numeric_columns[numeric_columns != 'precipitation'];


#############
# Get per-home data
#############

usage_df_by_hour <- usage_df[as.character(usage_df$timestamp) < '2014-09-01T00:00' 
                             & usage_df$total_kw > 0, ] %>% group_by(home_id) %>% summarize(
  home_id = first(home_id),
  hours_of_day = length(unique(hour_of_day))
)
usage_df <- usage_df[usage_df$home_id %in% (
  usage_df_by_hour[usage_df_by_hour$hours_of_day == 24, ]$home_id
), ]

usage_df$hour_of_day <- as.factor(usage_df$hour_of_day)
usage_df$home_id <- as.factor(usage_df$home_id)
usage_df$group_id <- as.factor(usage_df$group_id)
usage_df$timestamp <- as.factor(usage_df$timestamp)

#############
# Model total kw based on home data
#############

model_power_for_home <- function(df, target) {
  
  df[paste(target, 'log', sep = '_')] <- log(df[target] %>% mutate_all(remove_zeros))
  df[paste(target, 'nonzero', sep = '_')] <- 1 * (df[[target]] > 0)
  usage_model_weather_log <- lm(eval(parse(text = paste(
    target,
    "_log ~ ",
    'hour_of_day + ', 
    paste(weather_variables, collapse = " + "),
    sep = ""
  ))), data = df)
      
  usage_model_weather_nonzero <- glm(eval(parse(text = paste(
    target,
    "_nonzero ~ ",
    'hour_of_day + ', 
    paste(weather_variables, collapse = " + "),
    sep = ""
  ))), data = df)
  
  usage_predictions <- exp(predict(usage_model_weather_log, df)) * usage_model_weather_nonzero$fitted.values
  # Errors in lognormal distribution are asymmetric, yielding systematic underestimates.
  # As long as there is homoskedasticity, the multiplicative difference between
  # mean(prediction) and exp(mean(log(prediction)) can be applied equally.
  multiplier <- mean(df[[target]]) / mean(usage_predictions)

  return (list(
    log_model = usage_model_weather_log,
    nonzero_model = usage_model_weather_nonzero,
    multiplier = multiplier))
}

for (target in c('total_kw')) {
  usage_df[paste(target, 'fitted', sep = '_')] <- NA
}

home_id_indexes <- which(diff(as.numeric(usage_df$home_id)) != 0);
home_id_indexes <- c(home_id_indexes, home_id_indexes[length(home_id_indexes)] + 1)

usage_df_by_home_id <- split(usage_df, f = usage_df$home_id);
home_id_num <- 0;
all_usage_predictions <- data.frame();

for (target in c('hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                 'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                 'cooking_kw')) {
  usage_df[1:dim(all_usage_predictions)[1], paste(target, 'fitted', sep = '_')] <- all_usage_predictions[, paste(target, 'fitted', sep = '_')];
}

previous_filled_index <- 0
for (home_id in usage_df$home_id[home_id_indexes][1846:length(home_id_indexes)]) {
  home_id_num <- home_id_num + 1
  print(home_id_num)
  # periodically clearing out all_usage_predictions speeds up execution by minimizing number of long-length rbinds,
  # while also minimizing index calls to usage_df. This compromise adds a few lines of code while saving many hours of execution time.
  if (home_id_num %% 1000 == 0) {
    for (target in c('hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                     'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                     'cooking_kw')) {
      usage_df[(previous_filled_index + 1):(previous_filled_index + dim(all_usage_predictions)[1]), 
               paste(target, 'fitted', sep = '_')] <- all_usage_predictions[, paste(target, 'fitted', sep = '_')];
    }
    previous_filled_index <- previous_filled_index + dim(all_usage_predictions)[1];
    all_usage_predictions <- data.frame();
  }
  user_df <- usage_df_by_home_id[[home_id]];
  user_df_train <- user_df[as.character(user_df$timestamp) < '2014-09-01T00:00', ];
  for (target in c('hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                       'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                       'cooking_kw')) {
    usage_model_weather <- lm(eval(parse(text = paste(
      target,
      " ~ ",
      'hour_of_day + ', 
      paste(weather_variables, collapse = " + "),
      sep = ""
    ))), data = user_df_train);
    usage_predictions <- predict(usage_model_weather, user_df);
    if (target == 'hvac_kw') { usage_predictions_df <- data.frame(hvac_kw_fitted = usage_predictions); }
    else { usage_predictions_df[, paste(target, 'fitted', sep = '_')] <- usage_predictions; }
  }
  all_usage_predictions <- rbind(all_usage_predictions, usage_predictions_df);
}
for (target in c('hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                 'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                 'cooking_kw')) {
  usage_df[(previous_filled_index + 1):(previous_filled_index + dim(all_usage_predictions)[1]), 
           paste(target, 'fitted', sep = '_')] <- all_usage_predictions[, paste(target, 'fitted', sep = '_')];
}

for (target in c('total_kw')) {
  home_id_num <- 0
  all_usage_predictions <- NULL
  for (home_id in usage_df$home_id[home_id_indexes]) {
    home_id_num <- home_id_num + 1
    print(home_id_num)
    
    user_df <- usage_df_by_home_id[[home_id]]
    user_models <- model_power_for_home(user_df[as.character(user_df$timestamp) < '2014-09-01T00:00', ], target)
    usage_predictions <- exp(predict(user_models[['log_model']], user_df)) * predict(user_models[['nonzero_model']], user_df) * user_models[['multiplier']];
    all_usage_predictions <- c(all_usage_predictions, usage_predictions);
  }
  usage_df[paste(target, 'fitted', sep = '_')] <- all_usage_predictions
}
usage_df$total_kw_fitted_category_sum <- 0
for (target in c('hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                 'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                 'cooking_kw')) {
  usage_df$total_kw_fitted_category_sum <- usage_df$total_kw_fitted_category_sum + usage_df[, paste(target, 'fitted', sep='_')];
}

# Evaluate home model, total usage
home_data_by_timestamp <- usage_df %>% group_by(timestamp) %>% summarize(
  total_kw        = mean(total_kw),
  total_kw_fitted = mean(total_kw_fitted),
  total_kw_fitted_category_sum = mean(total_kw_fitted_category_sum),
  hvac_kw_fitted         = mean(hvac_kw_fitted),
  hot_water_kw_fitted    = mean(hot_water_kw_fitted),
  refrigerator_kw_fitted = mean(refrigerator_kw_fitted),
  light_kw_fitted        = mean(light_kw_fitted),
  misc_kw_fitted         = mean(misc_kw_fitted),
  dishwasher_kw_fitted   = mean(dishwasher_kw_fitted),
  laundry_kw_fitted      = mean(laundry_kw_fitted),
  cooking_kw_fitted      = mean(cooking_kw_fitted),
  home_num               = length(home_id));

# Evaluate home model, total usage
aggregate_model_home <- lm(total_kw ~ 0 + total_kw_fitted, 
                      data = home_data_by_timestamp[as.character(home_data_by_timestamp$timestamp) < '2014-09-01T00:00', ])
summary(aggregate_model_home)
home_data_by_timestamp$total_kw_fitted_aggregate <- predict(aggregate_model_home, home_data_by_timestamp)
plot(home_data_by_timestamp$total_kw_fitted_aggregate, 
     home_data_by_timestamp$total_kw)

evaluate_model <- function(df, col1, col2) {
  return(list(
    train_r2 = cor(
      df[as.character(df$timestamp) < '2014-09-01T00:00', ][[col1]], 
      df[as.character(df$timestamp) < '2014-09-01T00:00', ][[col2]]
    )^2,
    validation_r2 = cor(
      df[as.character(df$timestamp) >= '2014-09-01T00:00', ][[col1]], 
      df[as.character(df$timestamp) >= '2014-09-01T00:00', ][[col2]]
    )^2,
    rms = mean((
      df[as.character(df$timestamp) >= '2014-09-01T00:00', ][[col1]] - 
        df[as.character(df$timestamp) >= '2014-09-01T00:00', ][[col2]]
    )^2)^0.5
  ))
}

printouts_model <- function(df, col1, col2, model_description) {
  model_evaluation <- evaluate_model(df, col1, col2)
  print(paste('Train R^2, ', model_evaluation[['train_r2']], sep=""))
  print(paste('Validation R^2 ', model_evaluation[['validation_r2']], sep=""))
  print(paste('RMS, ', model_evaluation[['rms']], sep=""))
}

printouts_model(home_data_by_timestamp, 'total_kw_fitted_aggregate', 'total_kw', 'home/total usage');

# Evaluate home model, category sum
aggregate_model_home_category_sum <- lm(total_kw ~ 0 + total_kw_fitted_category_sum, 
                           data = home_data_by_timestamp[as.character(home_data_by_timestamp$timestamp) < '2014-09-01T00:00', ])
summary(aggregate_model_home_category_sum)
home_data_by_timestamp$total_kw_fitted_category_sum_aggregate <- predict(aggregate_model_home_category_sum, home_data_by_timestamp)
plot(home_data_by_timestamp$total_kw_fitted_category_sum_aggregate, 
     home_data_by_timestamp$total_kw)

printouts_model(home_data_by_timestamp, 'total_kw_fitted_category_sum_aggregate', 'total_kw', 'home/category sum');

# Evaluate home model, category separate
aggregate_model_home_category_separate <- lm(total_kw ~ 0 + hvac_kw_fitted
                                    + hot_water_kw_fitted + refrigerator_kw_fitted + light_kw_fitted
                                    + misc_kw_fitted + dishwasher_kw_fitted + laundry_kw_fitted + cooking_kw_fitted,
                           data = home_data_by_timestamp[as.character(home_data_by_timestamp$timestamp) < '2014-09-01T00:00', ])
summary(aggregate_model_home_category_separate)
home_data_by_timestamp$total_kw_fitted_category_separate_aggregate <- predict(aggregate_model_home_category_separate, home_data_by_timestamp)
plot(home_data_by_timestamp$total_kw_fitted_category_separate_aggregate, 
     home_data_by_timestamp$total_kw)

printouts_model(home_data_by_timestamp, 'total_kw_fitted_category_separate_aggregate', 'total_kw', 'home/category');


#############
# Get grouped dataset
#############

usage_df$group_timestamp <- paste(usage_df$group_id, usage_df$timestamp, sep = "_")
usage_df_by_group <- usage_df %>% group_by(group_timestamp) %>% summarize(
  fips_code = first(fips_code),
  timestamp = first(timestamp),
  group_id = first(group_id),
  total_kw = mean(total_kw),
  hvac_kw = mean(hvac_kw),
  hot_water_kw = mean(hot_water_kw),
  refrigerator_kw = mean(refrigerator_kw),
  light_kw = mean(light_kw),
  misc_kw = mean(misc_kw),
  dishwasher_kw = mean(dishwasher_kw),
  laundry_kw = mean(laundry_kw),
  cooking_kw = mean(cooking_kw),
  hour_of_day = first(hour_of_day),
  air_pressure = mean(air_pressure),
  dry_bulb_temperature = mean(dry_bulb_temperature),
  humidity = mean(humidity),
  wet_bulb_temperature = mean(wet_bulb_temperature),
  wind_speed = mean(wind_speed),
  oktas = mean(oktas),
  precipitation_normal = mean(precipitation_normal),
  home_num = length(home_id)
)
usage_df_by_group$group_timestamp <- NULL
usage_df_by_group <- usage_df_by_group[usage_df_by_group$home_num >= 15, ]

groups_by_hour <- usage_df_by_group[as.character(usage_df_by_group$timestamp) < '2014-09-01T00:00', ] %>% group_by(group_id) %>% summarize(
  group_id = first(group_id),
  hours_of_day = length(unique(hour_of_day))
)
usage_df_by_group <- usage_df_by_group[usage_df_by_group$group_id %in% (
  groups_by_hour[groups_by_hour$hours_of_day == 24, ]$group_id
), ]
summary_usage_by_group <- data.frame(
  mean=                    sapply(usage_df_by_group[, numeric_columns], mean, na.rm=TRUE),
  sd=                      sapply(usage_df_by_group[, numeric_columns], sd, na.rm=TRUE),
  min=                     sapply(usage_df_by_group[, numeric_columns], min, na.rm=TRUE),
  max=                     sapply(usage_df_by_group[, numeric_columns], max, na.rm=TRUE),
  median=                  sapply(usage_df_by_group[, numeric_columns], median, na.rm=TRUE),
  length=                  sapply(usage_df_by_group[, numeric_columns], length),
  skewness=                sapply(usage_df_by_group[, numeric_columns], skewness, na.rm=TRUE),
  excess_kurtosis=         sapply(usage_df_by_group[, numeric_columns], kurtosis, na.rm=TRUE, method = "moment") - 3,
  zero_ratio=              sapply(usage_df_by_group[, numeric_columns], ratio_zeros),
  skewness_log=            sapply(log(usage_df_by_group[, numeric_columns] %>% mutate_all(remove_zeros)), skewness, na.rm=TRUE),
  excess_kurtosis_log=     sapply(log(usage_df_by_group[, numeric_columns] %>% mutate_all(remove_zeros)), kurtosis, na.rm=TRUE, method = "moment") - 3,
  miss_val=                sapply(usage_df_by_group[, numeric_columns], function(x) sum(length(which(is.na(x)))))
)
print(summary_usage_by_group)

#############
# Model power based on group data
#############

group_id_indexes <- which(diff(as.numeric(usage_df_by_group$group_id)) != 0)

for (target in c('total_kw', 'hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                 'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                 'cooking_kw')) {
  usage_df_by_group[paste(target, 'fitted', sep = '_')] <- NA
}

for (group_id_num in 1:(length(group_id_indexes) + 1)) {
  print(group_id_num)
  group_row_start <- ifelse(group_id_num != 1, 
                            group_id_indexes[group_id_num - 1] + 1,
                            1)
  group_row_end <- ifelse(group_id_num != (length(group_id_indexes) + 1), 
                          group_id_indexes[group_id_num], 
                          dim(usage_df_by_group)[1])
  
  user_indices <- group_row_start:group_row_end
  user_df <- usage_df_by_group[user_indices, ]
  group_id <- lapply(user_df$group_id, as.character)[[1]]
  
  for (target in c('total_kw', 'hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                   'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                   'cooking_kw')) {
    group_model <- lm(eval(parse(text = paste(
      target,
      " ~ ",
      'hour_of_day + ', 
      paste(weather_variables, collapse = " + "),
      sep = ""
    ))), data = user_df[as.character(user_df$timestamp) < '2014-09-01T00:00', ]
    , weights = home_num)
    usage_predictions <- predict(group_model, user_df);
    usage_df_by_group[user_indices, paste(target, 'fitted', sep = '_')] <- usage_predictions
  }
}

group_data_by_timestamp <- usage_df_by_group %>% group_by(timestamp) %>% summarize(
  total_kw               = weighted.mean(total_kw, home_num),
  hvac_kw                = weighted.mean(hvac_kw, home_num),
  hot_water_kw           = weighted.mean(hot_water_kw, home_num),
  refrigerator_kw        = weighted.mean(refrigerator_kw, home_num),
  light_kw               = weighted.mean(light_kw, home_num),
  misc_kw                = weighted.mean(misc_kw, home_num),
  dishwasher_kw          = weighted.mean(dishwasher_kw, home_num),
  laundry_kw             = weighted.mean(laundry_kw, home_num),
  cooking_kw             = weighted.mean(cooking_kw, home_num),
  total_kw_fitted        = weighted.mean(total_kw_fitted, home_num),
  hvac_kw_fitted         = weighted.mean(hvac_kw_fitted, home_num),
  hot_water_kw_fitted    = weighted.mean(hot_water_kw_fitted, home_num),
  refrigerator_kw_fitted = weighted.mean(refrigerator_kw_fitted, home_num),
  light_kw_fitted        = weighted.mean(light_kw_fitted, home_num),
  misc_kw_fitted         = weighted.mean(misc_kw_fitted, home_num),
  dishwasher_kw_fitted   = weighted.mean(dishwasher_kw_fitted, home_num),
  laundry_kw_fitted      = weighted.mean(laundry_kw_fitted, home_num),
  cooking_kw_fitted      = weighted.mean(cooking_kw_fitted, home_num),
  home_num               = sum(home_num))

group_data_by_timestamp$total_kw_fitted_category_sum <- 0
for (target in c('hvac_kw', 'hot_water_kw', 'refrigerator_kw',
                 'light_kw', 'misc_kw', 'dishwasher_kw', 'laundry_kw',
                 'cooking_kw')) {
  group_data_by_timestamp$total_kw_fitted_category_sum <- group_data_by_timestamp$total_kw_fitted_category_sum + group_data_by_timestamp[[paste(target, 'fitted', sep='_')]];
}

aggregate_model_group <- lm(total_kw ~ 0 + total_kw_fitted,
                                              data = group_data_by_timestamp[as.character(group_data_by_timestamp$timestamp) < '2014-09-01T00:00', ])
summary(aggregate_model_group)
group_data_by_timestamp$total_kw_fitted_aggregate <- predict(aggregate_model_group, group_data_by_timestamp)
plot(group_data_by_timestamp$total_kw_fitted_aggregate, group_data_by_timestamp$total_kw)
printouts_model(group_data_by_timestamp, 'total_kw', 'total_kw_fitted_aggregate', 'group/total')


aggregate_model_group_category_sum <- lm(total_kw ~ 0 + total_kw_fitted_category_sum,
                            data = group_data_by_timestamp[as.character(group_data_by_timestamp$timestamp) < '2014-09-01T00:00', ])
summary(aggregate_model_group_category_sum)
group_data_by_timestamp$total_kw_fitted_category_sum_aggregate <- predict(aggregate_model_group_category_sum, group_data_by_timestamp)
plot(group_data_by_timestamp$total_kw_fitted_category_sum_aggregate, group_data_by_timestamp$total_kw)
printouts_model(group_data_by_timestamp, 'total_kw', 'total_kw_fitted_category_sum_aggregate', 'group/total')


aggregate_model_group_category_separate <- lm(total_kw ~ 0 + hvac_kw_fitted + hot_water_kw_fitted
                      + refrigerator_kw_fitted + light_kw_fitted + misc_kw_fitted
                      + dishwasher_kw_fitted + laundry_kw_fitted + cooking_kw_fitted,
  data = group_data_by_timestamp[as.character(group_data_by_timestamp$timestamp) < '2014-09-01T00:00', ])
summary(aggregate_model_group_category_separate)
group_data_by_timestamp$total_kw_fitted_category_separate_aggregate <- predict(aggregate_model_group_category_separate, group_data_by_timestamp)
plot(group_data_by_timestamp$total_kw_fitted_category_separate_aggregate, group_data_by_timestamp$total_kw)
printouts_model(group_data_by_timestamp, 'total_kw', 'total_kw_fitted_category_separate_aggregate', 'group/category separate')

#############
# Get correlation matrix & model evaluations
#############

valid_timestamps <- home_data_by_timestamp$timestamp
valid_timestamps <- valid_timestamps[valid_timestamps %in% group_data_by_timestamp$timestamp]

model_comparison_df <- data.frame(timestamp = valid_timestamps)
home_data_by_timestamp_valid_ts <- home_data_by_timestamp[home_data_by_timestamp$timestamp %in% valid_timestamps, ]
model_comparison_df[, 'home_total'] <- home_data_by_timestamp_valid_ts$total_kw_fitted_aggregate
model_comparison_df[, 'home_cat_sum'] <- home_data_by_timestamp_valid_ts$total_kw_fitted_category_sum_aggregate
model_comparison_df[, 'home_cat_sep'] <- home_data_by_timestamp_valid_ts$total_kw_fitted_category_separate_aggregate
group_data_by_timestamp_valid_ts <- group_data_by_timestamp[group_data_by_timestamp$timestamp %in% valid_timestamps, ]
model_comparison_df[, 'group_total'] <- group_data_by_timestamp_valid_ts$total_kw_fitted_aggregate
model_comparison_df[, 'group_cat_sum'] <- group_data_by_timestamp_valid_ts$total_kw_fitted_category_sum_aggregate
model_comparison_df[, 'group_cat_sep'] <- group_data_by_timestamp_valid_ts$total_kw_fitted_category_separate_aggregate

cor(model_comparison_df[, c('home_total', 'home_cat_sum', 'home_cat_sep',
                            'group_total', 'group_cat_sum', 'group_cat_sep')])

model_evaluation_df <- data.frame()
model_evaluation_df <- rbind(model_evaluation_df, evaluate_model(home_data_by_timestamp, 'total_kw', 'total_kw_fitted_aggregate'))
model_evaluation_df <- rbind(model_evaluation_df, evaluate_model(home_data_by_timestamp, 'total_kw', 'total_kw_fitted_category_sum_aggregate'))
model_evaluation_df <- rbind(model_evaluation_df, evaluate_model(home_data_by_timestamp, 'total_kw', 'total_kw_fitted_category_separate_aggregate'))
model_evaluation_df <- rbind(model_evaluation_df, evaluate_model(group_data_by_timestamp, 'total_kw', 'total_kw_fitted_aggregate'))
model_evaluation_df <- rbind(model_evaluation_df, evaluate_model(group_data_by_timestamp, 'total_kw', 'total_kw_fitted_category_sum_aggregate'))
model_evaluation_df <- rbind(model_evaluation_df, evaluate_model(group_data_by_timestamp, 'total_kw', 'total_kw_fitted_category_separate_aggregate'))
row.names(model_evaluation_df) <- c('home_total', 'home_cat_sum', 'home_cat_sep',
                                    'group_total', 'group_cat_sum', 'group_cat_sep')
print(model_evaluation_df)

save.image("C:/Users/georg/OneDrive/Documents/MIS581/analysis_workspace.RData")
load("C:/Users/georg/OneDrive/Documents/MIS581/analysis_workspace.RData")
