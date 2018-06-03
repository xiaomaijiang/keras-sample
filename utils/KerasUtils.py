import keras
from keras.callbacks import EarlyStopping


class KerasUtils(object):
    """
    构建TensorBoard回调，生成后可以使用
    tensorboard --logdir=./logs
    查看训练过程
    """

    def buildTensorflowCallback(self, log_dir='./logs'):
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                           histogram_freq=1,
                                                           write_graph=True,
                                                           write_images=False,
                                                           embeddings_freq=0,
                                                           embeddings_layer_names=None,
                                                           embeddings_metadata=None)
        earlystopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.09, patience=5, verbose=0, mode='auto')
        callbacks = [tensorboard_callback, earlystopping_callback];
        return callbacks
