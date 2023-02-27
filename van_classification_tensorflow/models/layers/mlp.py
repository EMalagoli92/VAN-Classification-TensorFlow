from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa

from van_classification_tensorflow.models.layers.dwconv import DWConv
from van_classification_tensorflow.models.layers.utils import (
    Conv2d_,
    CustomNormalInitializer,
)


@tf.keras.utils.register_keras_serializable(package="van")
class Mlp(tf.keras.layers.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = "gelu",
        drop: float = 0.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        in_features : int
            Input features dimension.
        hidden_features : Optional[int], optional
            Hidden features dimension.
            The default is None.
        out_features : Optional[int], optional
            Output features dimension.
            The default is None.
        act_layer : str, optional
            Name of activation layer.
            The default is "gelu".
        drop : float, optional
            Dropout rate.
            The default is 0.0.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop = drop

    def build(self, input_shape):
        self._out_features = self.out_features or self.in_features
        self._hidden_features = self.hidden_features or self.in_features
        self.fc1 = Conv2d_(
            in_channels=self.in_features,
            out_channels=self._hidden_features,
            kernel_size=1,
            kernel_initializer=CustomNormalInitializer(
                kernel_size=1, out_channels=self._hidden_features
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="fc1",
        )
        self.dwconv = DWConv(dim=self._hidden_features, name="dwconv")
        if self.act_layer == "gelu":
            self.act = tfa.layers.GELU(approximate=False, name="act")
        else:
            self.act = tf.keras.layers.Activation(
                self.act_layer, dtype=self.dtype, name="act"
            )

        self.fc2 = Conv2d_(
            in_channels=self._hidden_features,
            out_channels=self._out_features,
            kernel_size=1,
            kernel_initializer=CustomNormalInitializer(
                kernel_size=1, out_channels=self._out_features
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="fc2",
        )
        self.drop = tf.keras.layers.Dropout(rate=self.drop, name="drop")
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.fc1(inputs)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_features": self.in_features,
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "act_layer": self.act_layer,
                "drop": self.drop,
            }
        )
        return config
