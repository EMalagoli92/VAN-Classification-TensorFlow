import tensorflow as tf

from van_classification_tensorflow.models.layers.utils import Conv2d_, CustomNormalInitializer

@tf.keras.utils.register_keras_serializable(package="van")
class DWConv(tf.keras.layers.Layer):
    def __init__(self,dim=768,**kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        
    def build(self,input_shape):
        self.dwconv = Conv2d_(in_channels = self.dim, 
                              out_channels = self.dim, 
                              kernel_size = 3, 
                              stride = 1, 
                              padding = 1, 
                              bias=True, 
                              groups=self.dim,
                              kernel_initializer = CustomNormalInitializer(kernel_size = 3, out_channels = self.dim, groups = self.dim),
                              bias_initializer = tf.keras.initializers.Zeros()
                              )
        super().build(input_shape)
        
    def call(self, inputs, *args, **kwargs):
        return self.dwconv(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config
        
        