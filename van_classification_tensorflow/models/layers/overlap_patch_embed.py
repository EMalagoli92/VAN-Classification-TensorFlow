import tensorflow as tf
from van_classification_tensorflow.models.layers.utils import BatchNorm2d_, Conv2d_
from van_classification_tensorflow.models.utils import _ntuple


@tf.keras.utils.register_keras_serializable(package="van")
class OverlapPatchEmbed(tf.keras.layers.Layer):
    def __init__(self, 
                 img_size = 224,
                 patch_size = 7,
                 stride = 4,
                 in_chans = 3,
                 embed_dim = 768,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
    def build(self,input_shape):
        _patch_size = _ntuple(2)(self.patch_size)
        self.proj = Conv2d_(in_channels = self.in_chans,
                            out_channels = self.embed_dim,
                            kernel_size = _patch_size,
                            stride = self.stride,
                            padding = (_patch_size[0] // 2, _patch_size[1] // 2),
                            name = "proj"
                            )
        self.norm = BatchNorm2d_(self.embed_dim,name="norm")
        super().build(input_shape)
        
    def call(self,inputs,*args,**kwargs):
        x = self.proj(inputs)
        x_shape = tf.shape(x)
        H = x_shape[2]
        W = x_shape[3]
        x = self.norm(x)
        return x, H, W
        
    def get_config(self):
        config = super().get_config()
        config.update({"img_size": self.img_size,
                       "patch_size": self.patch_size,
                       "stride": self.stride,
                       "in_chans": self.in_chans,
                       "embed_dim": self.embed_dim})
        return config