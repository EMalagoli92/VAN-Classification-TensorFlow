import tensorflow as tf
from van_classification_tensorflow.models.layers.utils import BatchNorm2d_, DropPath_, Identity_
from van_classification_tensorflow.models.layers.attention import Attention
from van_classification_tensorflow.models.layers.mlp import Mlp


@tf.keras.utils.register_keras_serializable(package="van")
class Block(tf.keras.layers.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio = 4.,
                 drop = 0.,
                 drop_path = 0.,
                 act_layer = "gelu",
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop = drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        
    def build(self,input_shape):
        self.norm1 = BatchNorm2d_(self.dim, name = "norm1")
        self.attn = Attention(self.dim, name = "attn")
        self.drop_path = DropPath_(self.drop_path, name = "drop_path") if self.drop_path > 0. else Identity_(name = "drop_path")
        self.norm2 = BatchNorm2d_(self.dim, name = "norm2")
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(in_features = self.dim,
                       hidden_features = mlp_hidden_dim,
                       act_layer = self.act_layer,
                       drop = self.drop,
                       name = "mlp"
                       )
        layer_scale_init_values = 1e-2
        self.layer_scale_1 = self.add_weight(name = "layer_scale_1",
                                             shape = self.dim,
                                             initializer = tf.keras.initializers.Constant(layer_scale_init_values),
                                             trainable = True,
                                             dtype = self.dtype
                                             )
        self.layer_scale_2 = self.add_weight(name = "layer_scale_2",
                                             shape = self.dim,
                                             initializer = tf.keras.initializers.Constant(layer_scale_init_values),
                                             trainable = True,
                                             dtype = self.dtype
                                             )
        super().build(input_shape)
        
    
    def call(self,inputs,*args,**kwargs):
        x = inputs + self.drop_path(tf.expand_dims(tf.expand_dims(self.layer_scale_1,-1),-1) * self.attn(self.norm1(inputs)))
        x = x + self.drop_path(tf.expand_dims(tf.expand_dims(self.layer_scale_2,-1),-1) * self.attn(self.norm1(x)))
        return x
        
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim,
                       "mlp_ratio": self.mlp_ratio,
                       "drop": self.drop,
                       "drop_path": self.drop_path,
                       "act_layer": self.act_layer
                       })
        return config