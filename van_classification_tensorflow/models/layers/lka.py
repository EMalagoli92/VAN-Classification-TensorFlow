import tensorflow as tf

from van_classification_tensorflow.models.layers.utils import (
    Conv2d_,
    CustomNormalInitializer,
)


@tf.keras.utils.register_keras_serializable(package="van")
class LKA(tf.keras.layers.Layer):
    """Large Kernel Attention(LKA) of VAN."""

    def __init__(self, dim: int, **kwargs):
        """
        Parameters
        ----------
        dim : int
            Number of input channels.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.conv0 = Conv2d_(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=5,
            padding=2,
            groups=self.dim,
            kernel_initializer=CustomNormalInitializer(
                kernel_size=5, out_channels=self.dim, groups=self.dim
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="conv0",
        )
        self.conv_spatial = Conv2d_(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=7,
            stride=1,
            padding=9,
            groups=self.dim,
            dilation=3,
            kernel_initializer=CustomNormalInitializer(
                kernel_size=7, out_channels=self.dim, groups=self.dim
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="conv_spatial",
        )
        self.conv1 = Conv2d_(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=1,
            kernel_initializer=CustomNormalInitializer(
                kernel_size=1, out_channels=self.dim
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="conv1",
        )
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        u = tf.identity(inputs)
        attn = self.conv0(inputs)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config
