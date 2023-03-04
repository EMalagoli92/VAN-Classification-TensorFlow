from typing import List, Literal, Optional, Tuple, Union

import tensorflow as tf

from van_classification_tensorflow import __version__
from van_classification_tensorflow.models.config import MODELS_CONFIG, TF_WEIGHTS_URL
from van_classification_tensorflow.models.layers.block import Block
from van_classification_tensorflow.models.layers.overlap_patch_embed import (
    OverlapPatchEmbed,
)
from van_classification_tensorflow.models.layers.utils import (
    LayerNorm_,
    Linear_,
    TruncNormalInitializer_,
)
from van_classification_tensorflow.models.utils import _imgr2tuple, _to_channel_first


@tf.keras.utils.register_keras_serializable(package="van")
class VAN_(tf.keras.Model):
    """Visual Attention Network."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: List[int] = [64, 128, 256, 512],
        mlp_ratios: List[int] = [4, 4, 4, 4],
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        depths: List[int] = [3, 4, 6, 3],
        num_stages: int = 4,
        include_top: bool = True,
        classifier_activation: Optional[str] = None,
        data_format: Literal[
            "channels_first", "channels_last"
        ] = tf.keras.backend.image_data_format(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        in_chans : int, optional
            Number of input channels.
            The default is 3.
        num_classes : int, optional
            Number of classes.
            The default is 1000.
        embed_dims : List[int], optional
            Dimensions of embedding.
            The default is [64, 128, 256, 512].
        mlp_ratios : List[int], optional
            MLP ratios.
            The default is [4, 4, 4, 4].
        drop_rate : float, optional
            Dropout rate after embedding.
            The default is 0.0.
        drop_path_rate : float, optional
            Drop path rate.
            The default is 0.0.
        depths : List[int], optional
            Number of layers in each stage.
            The default is [3, 4, 6, 3].
        num_stages : int, optional
            Number of stages.
            The default is 4.
        include_top : bool, optional
            Whether to include the fully-connected layer at the top, as the
            last layer of the network.
            The default is True.
        classifier_activation : Optional[str], optional
            String name for a tf.keras.layers.Activation layer.
            The default is None.
        data_format : Literal["channels_first", "channels_last"], optional
            A string, one of "channels_last" or "channels_first".
            The ordering of the dimensions in the inputs.
            "channels_last" corresponds to inputs with shape:
            (batch_size, height, width, channels)
            while "channels_first" corresponds to inputs with shape
            (batch_size, channels, height, width).
            The default is tf.keras.backend.image_data_format().
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.num_stages = num_stages
        self.include_top = include_top
        self.classifier_activation = classifier_activation
        self.data_format = data_format

        dpr = [
            i * self.drop_path_rate / (sum(self.depths) - 1)
            for i in range(sum(self.depths))
        ]
        cur = 0

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=self.in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                name=f"patch_embed{i+1}",
            )
            block = [
                Block(
                    dim=self.embed_dims[i],
                    mlp_ratio=self.mlp_ratios[i],
                    drop=self.drop_rate,
                    drop_path=dpr[cur + j],
                    name=f"block{i+1}/{j}",
                )
                for j in range(self.depths[i])
            ]
            norm = LayerNorm_(self.embed_dims[i], epsilon=1e-6, name=f"norm{i+1}")
            cur += self.depths[i]

            setattr(self, f"patch_embed{i+1}", patch_embed)
            setattr(self, f"block{i+1}", block)
            setattr(self, f"norm{i+1}", norm)

        if self.include_top:
            self.head = Linear_(
                in_features=self.embed_dims[3],
                units=self.num_classes,
                kernel_initializer=TruncNormalInitializer_(std=0.02),
                bias_initializer=tf.keras.initializers.Zeros(),
                name="head",
            )

            if self.classifier_activation is not None:
                self.classifier_activation_ = tf.keras.layers.Activation(
                    self.classifier_activation, dtype=self.dtype, name="pred"
                )

    def forward_features(self, x):
        x_shape = tf.shape(x)
        B = x_shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i+1}")
            block = getattr(self, f"block{i+1}")
            norm = getattr(self, f"norm{i+1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            dim1 = tf.shape(x)[1]
            dim2 = tf.math.multiply(H, W)
            x = tf.reshape(x, [tf.shape(x)[0], dim1, dim2])
            x = tf.transpose(x, [0, 2, 1])
            x = norm(x)
            if i != self.num_stages - 1:
                x = tf.reshape(x, [B, H, W, dim1])
                x = tf.transpose(x, perm=[0, 3, 1, 2])

        return tf.math.reduce_mean(x, axis=1)

    def call(self, inputs, *args, **kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_first(inputs)
        x = self.forward_features(inputs)
        if self.include_top:
            x = self.head(x)
            if hasattr(self, "classifier_activation_"):
                x = self.classifier_activation_(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)

    def __to_functional(self):
        if self.built:
            x = tf.keras.layers.Input(shape=(self._build_input_shape[1:]))
            model = tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)
        else:
            raise ValueError(
                "This model has not yet been built. "
                "Build the model first by calling build() or "
                "by calling the model on a batch of data."
            )
        return model

    def summary(self, *args, **kwargs):
        self.__to_functional()
        super().summary(*args, **kwargs)

    def plot_model(self, *args, **kwargs):
        tf.keras.utils.plot_model(model=self.__to_functional(), *args, **kwargs)

    def save(self, *args, **kwargs):
        self.__to_functional().save(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_chans": self.in_chans,
                "num_classes": self.num_classes,
                "embed_dims": self.embed_dims,
                "mlp_ratios": self.mlp_ratios,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "depths": self.depths,
                "num_stages": self.num_stages,
                "include_top": self.include_top,
                "classifier_activation": self.classifier_activation,
                "data_format": self.data_format,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def VAN(
    configuration: Optional[
        Literal["van_b0", "van_b1", "van_b2", "van_b3", "van_b4", "van_b5", "van_b6"]
    ] = None,
    pretrained: bool = False,
    img_resolution: Optional[Union[int, Tuple[int, int], List[int]]] = None,
    **kwargs,
) -> tf.keras.Model:
    """Wrapper function for VAN model.

    Parameters
    ----------
    configuration : Optional[Literal["van_b0", "van_b1", "van_b2", "van_b3",
                                     "van_b4", "van_b5", "van_b6"]], optional
        Name of VAN predefined configuration.
        Possible values are: "van_b0", "van_b1", "van_b2", "van_b3", "van_b4",
        "van_b5", "van_b6".
        The default is None.
    pretrained : bool, optional
        Whether to use ImageNet pretrained weights.
        The default is False.
    img_resolution : Optional[Union[int, Tuple[int, int], List[int]]], optional
        Input image resolution (H,W).
        The default is None.
    **kwargs
        Additional keyword arguments.

    Raises
    ------
    ValueError
        If choosen configuration is not among those with ImageNet pretrained
        weights, i.e. not in:
        ["van_b0","van_b1","van_b2","van_b3"]
    KeyError
        If choosen configuration not in:
        ["van_b0","van_b1","van_b2","van_b3","van_b4","van_b5","van_b6"]

    Returns
    -------
    tf.keras.Model
        VAN model.
    """
    if configuration is not None:
        if configuration in MODELS_CONFIG.keys():
            model = VAN_(**MODELS_CONFIG[configuration]["spec"], **kwargs)
            if pretrained:
                pretrained_img_resolution = MODELS_CONFIG[configuration][
                    "pretrained_img_resolution"
                ]
                if pretrained_img_resolution is not None:
                    img_resolution = (
                        img_resolution
                        if img_resolution is not None
                        else pretrained_img_resolution
                    )
                    img_resolution = _imgr2tuple(img_resolution)
                    if model.data_format == "channels_last":
                        model.build((None, img_resolution[0], img_resolution[1], 3))
                    elif model.data_format == "channels_first":
                        model.build((None, 3, img_resolution[0], img_resolution[1]))
                    weights_path = "{}/{}/{}.h5".format(
                        TF_WEIGHTS_URL, __version__, configuration
                    )
                    model_weights = tf.keras.utils.get_file(
                        fname="{}.h5".format(configuration),
                        origin=weights_path,
                        cache_subdir="datasets/van_classification_tensorflow",
                    )
                    model.load_weights(model_weights, by_name=(not model.include_top))
                else:
                    pretrained_available = [
                        conf
                        for conf in MODELS_CONFIG.keys()
                        if MODELS_CONFIG[conf]["pretrained_img_resolution"] is not None
                    ]
                    raise ValueError(
                        "Pretrained weights only available for the "
                        f"following configurations: {pretrained_available}"
                    )
            return model
        else:
            raise KeyError(
                f"{configuration} configuration not found. "
                f"Valid values are: {list(MODELS_CONFIG.keys())}"
            )
    else:
        return VAN_(**kwargs)
