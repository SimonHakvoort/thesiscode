import os
# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from src.loading_data.get_data import load_cv_data
from src.linreg_emos.emos import LinearEMOS
import pickle as pkl

# Load the training and test data.
all_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

location_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

scale_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in all_features}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

train_data, test_data, data_info = load_cv_data(0, features_names_dict)


# Prepare the training data, by adding a constant weight and batching and shuffling the data set.
def addweight(X, y):
    return X, y, tf.constant(1, dtype=tf.float32)

train_data = train_data.map(addweight)

train_data = train_data.shuffle(train_data.cardinality())

train_data = train_data.batch(256)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data.batch(test_data.cardinality())

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)


# Choose the loss function
loss = "loss_twCRPS_sample"
#loss = "loss_cPIT"
samples = 250

# Select the chaining function in case the twCRPS is used. Also select the parameters.
chain_function = "chain_function_indicator"
chain_function_threshold = 14

### Sharp sigmoid parameters
# chain_function_mean = 8.830960273742676
# chain_function_std = 1.0684260129928589
# chain_function_constant = 0.015800999477505684

### Sigmoid parameters
# chain_function_mean = 7.050563812255859
# chain_function_std = 2.405172109603882
# chain_function_constant = 0.06170300021767616		

### Best CNN parameters
chain_function_mean = 5.419507
chain_function_std = 7.822199
chain_function_constant = 0.919453
				

# possible optimizers: 'SGD', 'Adam'
optimizer = "Adam"
learning_rate = 0.01

# Select the parametric distribution. The variables distribution_1 and distribution_2 only get used in case an (adaptive) mixture distibution is selected. 
forecast_distribution = "distr_trunc_normal"

distribution_1 = "distr_trunc_normal"
distribution_2 = "distr_log_normal"

random_init = False
printing = True
subset_size = None

setup = {'loss': loss,
         'samples': samples, 
         'optimizer': optimizer, 
         'learning_rate': learning_rate, 
         'forecast_distribution': forecast_distribution,
         'chain_function': chain_function,
         'chain_function_threshold': chain_function_threshold,
         'distribution_1': distribution_1,
         'distribution_2': distribution_2,
         'chain_function_mean': chain_function_mean,
         'chain_function_std': chain_function_std,
         'chain_function_constant': chain_function_constant,
         'all_features': all_features,
         'location_features': location_features,
         'scale_features': scale_features,
         'random_init': random_init,
         'subset_size': subset_size,
        'printing': printing,
         }

# In case we a mixture distribution we pretrain the seperate models.
if forecast_distribution == 'distr_mixture_linear' or forecast_distribution == 'distr_mixture':
    setup['forecast_distribution'] = distribution_1

    emos1 = LinearEMOS(setup)

    setup['forecast_distribution'] = distribution_2

    emos2 = LinearEMOS(setup)

    emos1.fit(train_data, 75, False)
    emos2.fit(train_data, 75, False)

    setup['forecast_distribution'] = forecast_distribution
    print(setup['forecast_distribution'])

    setup['parameters'] = {**emos1.get_parameters(), **emos2.get_parameters()}


epochs = 450

batch_size = None

emos = LinearEMOS(setup)

my_dict = emos.fit(train_data, epochs)

# Saving the model.
mydict = emos.to_dict()

filepath = '/net/pc200239/nobackup/users/hakvoort/models/final_models/linregemos/emos_tn_indictor_weight_14'

with open(filepath, 'wb') as f:
    pkl.dump(mydict, f)
