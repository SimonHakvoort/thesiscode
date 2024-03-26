import tensorflow as tf

from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten
from tensorflow.keras.models import Model

class NN_Forecast():
    def __init__(self):
        self.model = self.create_model()

    def get_distribution(self, X, variance):
            mu, sigma = self.model([X, variance])
            mu = tf.squeeze(mu)
            sigma = tf.squeeze(sigma)
            return mu, sigma
    
    def create_model(self):
        # create a neural network model for the truncated normal distribution which receives X and variances
        # as input and outputs the mean and standard deviation of the distribution

        features_input = tf.keras.Input(shape=(self.num_features,), name="features_input")
        variance_input = tf.keras.Input(shape=(1,), name="variance_input")

        # two dense layers with 64 and 32 units respectively
        dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')(features_input)
        dense2 = tf.keras.layers.Dense(32, activation='relu', name='dense2')(dense1)

        mu_output = tf.keras.layers.Dense(1, name='mu_output')(dense2)
        sigma_output = tf.keras.layers.Dense(1, activation='softplus', name='sigma_output')(dense2)

        model = Model(inputs=[features_input, variance_input], outputs=[mu_output, sigma_output])

        return model
    
    def create_model_conv(self):
        grid_input = tf.keras.Input(shape=(50, 50, 1), name="grid_input")
        features_input = tf.keras.Input(shape=(self.num_features,), name="features_input")
        variance_input = tf.Keras.Input(shape=(1,), name="variance_input")

        # three convolutional layers with 7x7, 5x5 and 3x3 filters with 4, 8 and 16 filters respectively
        # max pooling after each convolutional layer and batch normalization after each max pooling layer

        conv1 = Conv2D(filters=2, kernel_size=(7, 7), activation='relu', name='conv1')(grid_input)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
        norm1 = tf.keras.layers.BatchNormalization(name='norm1')(pool1)

        conv2 = Conv2D(filters=4, kernel_size=(5, 5), activation='relu', name='conv2')(norm1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
        norm2 = tf.keras.layers.BatchNormalization(name='norm2')(pool2)

        conv3 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv3')(norm2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
        norm3 = tf.keras.layers.BatchNormalization(name='norm3')(pool3)

        # flatten the output of the last convolutional layer
        flat = tf.keras.layers.Flatten(name='flat')(norm3)

        # concatenate the flattened output of the last convolutional layer with the features input
        concat = tf.keras.layers.Concatenate(name='concat')([flat, features_input])

        # two dense layers with 64 and 32 units respectively
        dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense1')(concat)
        dense2 = tf.keras.layers.Dense(32, activation='relu', name='dense2')(dense1)

        mu_output = tf.keras.layers.Dense(1, name='mu_output')(dense2)
        sigma_output = tf.keras.layers.Dense(1, name='sigma_output')(dense2)

        model = Model(inputs=[grid_input, features_input, variance_input], outputs=[mu_output, sigma_output])

        return model



    def NN_CRPS_wrapper(self):    
        def NN_CRPS(y_true, y_pred):
            """
            The loss function for the CRPS, based on the forecast distribution and observations.
            We use a sample based approach to estimate the expected value of the CRPS.

            Arguments:
            - X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
            - y (tf.Tensor): the observations of shape (n,).
            - variance (tf.Tensor): the variance of the forecast distribution around the grid point of shape (n,).
            - samples (int): the amount of samples used to estimate the expected value of the CRPS.

            Returns:
            - the loss value.
            """
            mu, sigma = y_pred[0], y_pred[1]

            forecast_distribution = tfpd.TruncatedNormal(loc=mu, scale=sigma, low=0.0, high=100.0)

            #X_1 has shape (samples, n), where n is the number of observations
            X_1 = forecast_distribution.sample(100)
            X_2 = forecast_distribution.sample(100)

            tf.print("X_1 shape: ", tf.shape(X_1))
            tf.print("X_2 shape: ", tf.shape(X_2))
            tf.print("y_true shape: ", tf.shape(y_true))

            tf.debugging.assert_shapes([(X_1, (100, None)), (X_2, (100, None))])
            

            # y will be broadcasted to the shape of X_1 and X_2
            E_1 = tf.reduce_mean(tf.abs(X_1 - y_true), axis=0)
            E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

            tf.print("E_1 shape: ", tf.shape(E_1))
            tf.print("E_2 shape: ", tf.shape(E_2))

            return tf.reduce_mean(E_1 - 0.5 * E_2)
        return NN_CRPS