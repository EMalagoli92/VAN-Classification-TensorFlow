import tensorflow as tf
import tensorflow_addons as tfa

from van_classification_tensorflow.models.layers.lka import LKA
from van_classification_tensorflow.models.layers.utils import (
    Conv2d_,
    CustomNormalInitializer,
)


@tf.keras.utils.register_keras_serializable(package="van")
class Attention(tf.keras.layers.Layer):
    """Basic attention module in VAN Block."""

    def __init__(self, d_model: int, **kwargs):
        """
        Parameters
        ----------
        d_model : int
            Number of input channels.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.proj_1 = Conv2d_(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=1,
            kernel_initializer=CustomNormalInitializer(
                kernel_size=1, out_channels=self.d_model
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="proj_1",
        )
        self.activation = tfa.layers.GELU(approximate=False, name="activation")
        self.spatial_gating_unit = LKA(self.d_model, name="spatial_gating_unit")
        self.proj_2 = Conv2d_(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=1,
            kernel_initializer=CustomNormalInitializer(
                kernel_size=1, out_channels=self.d_model
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="proj_2",
        )
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        shortcut = tf.identity(inputs)
        x = self.proj_1(inputs)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config
