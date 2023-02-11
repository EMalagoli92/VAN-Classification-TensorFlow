import tensorflow as tf
from van_classification_tensorflow import __version__
import tensorflow.experimental.numpy as tnp
from van_classification_tensorflow.models.config import MODELS_CONFIG, TF_WEIGHTS_URL, PRETRAINED_AVAILABLE
from van_classification_tensorflow.models.utils import _to_channel_first
from van_classification_tensorflow.models.layers.utils import LayerNorm_, Linear_, Identity_, TruncNormalInitializer_
from van_classification_tensorflow.models.layers.overlap_patch_embed import OverlapPatchEmbed
from van_classification_tensorflow.models.layers.block import Block

@tf.keras.utils.register_keras_serializable(package="van")
class VAN_(tf.keras.Model):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64,128,256,512],
                 mlp_ratios=[4,4,4,4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3,4,6,3],
                 num_stages=4,
                 flag=False,
                 classifier_activation = None,
                 data_format = tf.keras.backend.image_data_format(),
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.depths = depths
        self.num_stages = num_stages
        self.flag = flag
        self.classifier_activation = classifier_activation
        self.data_format = data_format
        
        if self.flag == False:
            self.num_classes_ = num_classes
        
        dpr = [i * self.drop_path_rate / (sum(self.depths) - 1)
               for i in range(sum(self.depths))
               ]
        cur = 0
        
        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size = self.img_size if i == 0 else self.img_size // (2 **(i+1)),
                                            patch_size = 7 if i==0 else 3,
                                            stride = 4 if i==0 else 2,
                                            in_chans = self.in_chans if i==0 else self.embed_dims[i -1],
                                            embed_dim = self.embed_dims[i],
                                            name=f"patch_embed{i+1}"
                                            )
            block = [Block(dim=self.embed_dims[i], mlp_ratio=self.mlp_ratios[i], drop=self.drop_rate, drop_path=dpr[cur +j],name=f'block{i+1}/{j}') for j in range(self.depths[i])]
            norm = LayerNorm_(self.embed_dims[i],epsilon=1e-6,name=f"norm{i+1}")
            cur += self.depths[i]
            
            setattr(self, f"patch_embed{i+1}", patch_embed)
            setattr(self, f"block{i+1}", block)
            setattr(self, f"norm{i+1}", norm)
            
        self.head = Linear_(in_features = self.embed_dims[3], 
                            units = self.num_classes,
                            kernel_initializer = TruncNormalInitializer_(std = .02),
                            bias_initializer = tf.keras.initializers.Zeros(),
                            name = "head"
                            ) if self.num_classes > 0 else Identity_(name='head')
        
        if self.classifier_activation is not None:
            self.classifier_activation_ = tf.keras.layers.Activation(self.classifier_activation, dtype=self.dtype, name="pred")
           
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
            dim2 = tf.shape(x)[2] * tf.shape(x)[3]
            x = tf.reshape(x,[tf.shape(x)[0],dim1,dim2])
            x = tnp.swapaxes(x,1,2)
            x = norm(x)
            if i != self.num_stages -1:
                x = tf.reshape(x,[B,H,W,tf.cast((dim1*dim2)/(H*W),tf.int32)])
                x = tf.transpose(x,perm=[0, 3, 1, 2])
        
        return tf.math.reduce_mean(x, axis=1)    
            
    def call(self, inputs, *args, **kwargs):
        if self.data_format == "channels_last":
            inputs = _to_channel_first(inputs)
        x = self.forward_features(inputs)
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
        config.update({"img_size": self.img_size,
                       "in_chans": self.in_chans,
                       "num_classes": self.num_classes,
                       "embed_dims": self.embed_dims,
                       "mlp_ratios": self.mlp_ratios,
                       "drop_rate": self.drop_rate,
                       "drop_path_rate": self.drop_path_rate,
                       "depths": self.depths,
                       "num_stages": self.num_stages,
                       "flag": self.flag,
                       "classifier_activation": self.classifier_activation,
                       "data_format": self.data_format
                       })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
def VAN(configuration = None,
        pretrained = False,
        **kwargs):
    if configuration is not None:
        if configuration in MODELS_CONFIG.keys():
            model = VAN_(**MODELS_CONFIG[configuration], **kwargs)
            if pretrained:
                if configuration in PRETRAINED_AVAILABLE: 
                    if model.data_format == "channels_last":
                        model.build((None, 224, 224, 3))
                    elif model.data_format == "channels_first":
                        model.build((None, 3, 224, 224))
                    weights_path = "{}/{}/{}.h5".format(
                        TF_WEIGHTS_URL, __version__, configuration
                    )
                    model_weights = tf.keras.utils.get_file(
                        fname="{}.h5".format(configuration),
                        origin=weights_path,
                        cache_subdir="datasets/van_classification_tensorflow",
                    )
                    model.load_weights(model_weights)
                else:
                    raise ValueError("Pretrained weights only available for the "\
                                     f"following configurations: {PRETRAINED_AVAILABLE}"
                                     )
            return model
        else:
            raise KeyError(
                f"{configuration} configuration not found. "
                f"Valid values are: {list(MODELS_CONFIG.keys())}"
            )
    else:
        return VAN_(**kwargs)