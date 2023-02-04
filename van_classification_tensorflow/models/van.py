import tensorflow as tf
from van_classification_tensorflow.layers.utils import LayerNorm_

@tf.keras.utils.register_keras_serializable(package="van")
class VAN(tf.keras.Model):
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
                       "flag": self.flag})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)    