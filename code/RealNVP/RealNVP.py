import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

""" ___________________________________________ RealNVP ___________________________________________ """


class RealNVPSimulator:
    """ 
    Creates a generative model that learns the distribution of a given continuous dataset, based on the RealNVP technique, introduced in:
    https://doi.org/10.48550/arXiv.1605.08803. Implementation is derived from: https://keras.io/examples/generative/real_nvp/#real-nvp, 
    the Keras offical example.  

    This class works as a wrapper around the Real NVP model, bringing its usage closer to the already existing simulators.
    To instantiate the model, the following arguments should or may be provided:

    Args
    ----
    dataset (pandas.DataFrame) : the data to be simulated. 
    output_dim (int): the hidden / latent space dimension of the Coupling MLPs; defaults to 256.
    reg (float): the regularization paramater of the L2 norm layer; defaults to 0.01. 
    """
    def __init__(self, dataset, output_dim=256, reg=0.01):

        self.dataset = dataset
        self.data = dataset.values

        self.norm = tf.keras.layers.Normalization()
        self.norm.adapt(self.data)
        self.normalized_data = self.norm(self.data)

        self.inverse_norm = tf.keras.layers.Normalization(invert=True)
        self.inverse_norm.adapt(self.data)

        self.output_dim = output_dim
        self.reg = reg

        self.model = RealNVP(num_coupling_layers=6, data=self.data, output_dim=output_dim, reg=reg)
    
    
    def fit(self, epochs=100, batch_size=256, learning_rate=0.0001, verbose=0):
        """ 
        Trains the model, following the typical TensorFlow pipeline. Returns nothing.

        Args
        ----
        epochs (int) : the number of training epochs; defaults to 100.
        batch_size (int) : the training batch_size; defaults to 256.
        learning_rate (float) : the training learning_rate; defaults to 1e-4.
        """
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        self.history = self.model.fit(
            self.normalized_data, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=0.2, callbacks=[early_stopping]
        )
    

    def simulate(self):
        """ 
        Simulates data, once the model has been fitted. No arguments needed.
        
        Return
        ------
        df (pandas.dataframe) : the simulated data, inverse-transformed
        """
        # From data to latent space.
        z, _ = self.model(self.normalized_data)

        # From latent space to data.
        samples = self.model.distribution.sample(len(self.dataset))
        x, _ = self.model.predict(samples, verbose=0)

        return pd.DataFrame(data=self.inverse_norm(x), columns=self.dataset.columns)


    def evaluate(self):
        """ 
        Visualizations that help in the model's evaluation. 
        Only available for 1-D data as KDE plots and 2-D data as scatter plots 
        """
        # Evaluation 
        plt.figure(figsize=(12, 7))
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("model loss")
        plt.legend(["train", "validation"], loc="upper right")
        plt.ylabel("loss")
        plt.xlabel("epoch")

        # From data to latent space.
        z, _ = self.model(self.normalized_data)

        # From latent space to data.
        samples = self.model.distribution.sample(len(self.dataset))
        x, _ = self.model.predict(samples, verbose=0)

        if self.data.shape[1]==1:
            
            f, axes = plt.subplots(ncols=2)
            f.set_size_inches(12, 7)

            sns.kdeplot(self.dataset.values, fill=True, palette="crest", label='true', ax=axes[0])
            sns.kdeplot(x, fill=True, color='r', palette="flare", label='sim', ax=axes[0])
            axes[0].set(title="True vs Simulated")
            axes[0].legend()

            sns.kdeplot(z, fill=True, palette="Pastel1", label='true', ax=axes[1])
            sns.kdeplot(samples, fill=True, palette="Pastel2", label='sim', ax=axes[1])
            axes[1].set(title="Inference vs Generated latent space Z")
            axes[1].legend()

        elif self.data.shape[1]==2:

            f, axes = plt.subplots(2, 2)
            f.set_size_inches(20, 15)

            axes[0, 0].scatter(self.normalized_data[:, 0], self.normalized_data[:, 1], color="r")
            axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
            axes[0, 1].scatter(z[:, 0], z[:, 1], color="r")
            axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
            axes[0, 1].set_xlim([-3.5, 4])
            axes[0, 1].set_ylim([-4, 4])
            axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
            axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")
            axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
            axes[1, 1].set(title="Generated data space X", label="x", ylabel="y")
            axes[1, 1].set_xlim([-2, 2])
            axes[1, 1].set_ylim([-2, 2])

        else:
            print('No visual evaluation offered for data of dimensions > 2.')


def Coupling(input_shape, output_dim=256, reg=0.01):
    """
    """
    input = tf.keras.layers.Input(shape=input_shape)

    t_layer_1 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(input)
    t_layer_2 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = tf.keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(input)
    s_layer_2 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = tf.keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = tf.keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer_4)

    return tf.keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])


class RealNVP(tf.keras.Model):
    """
    """
    def __init__(self, num_coupling_layers, data=None, output_dim=256, reg=0.01):
        super().__init__()

        self.num_coupling_layers = num_coupling_layers
        
        if data is not None:
            # Adjusting the number of layers as a temporary solution to dimension errors / difficulties
            self.num_coupling_layers = 2 * data.shape[1] if (2 * data.shape[1] < 12) else data.shape[1]
        
        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(shape=[data.shape[1]]).tolist(),     
            scale_diag=np.ones(shape=[data.shape[1]]).tolist() 
        )
        self.masks = np.array(
            # [[0, 1], [1, 0]] * (num_coupling_layers // 2),
            np.identity(n=data.shape[1], dtype='int')[::-1].tolist() * (self.num_coupling_layers // data.shape[1]), 
            dtype="float32"
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # self.layers_list = [Coupling(2) for i in range(self.num_coupling_layers)]
        self.layers_list = [Coupling(data.shape[1], output_dim=output_dim, reg=reg) for i in range(self.num_coupling_layers)]


    @property
    def metrics(self):
        """
        """
        return [self.loss_tracker]


    def call(self, x, training=True):
        """
        """
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])
        return x, log_det_inv


    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        """
        """
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)


    def train_step(self, data):
        """
        """
        with tf.GradientTape() as tape:
            loss = self.log_loss(data)
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


    def test_step(self, data):
        """
        """
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    
