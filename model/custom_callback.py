import wandb
import tensorflow as tf

class LRLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        wandb.log({'lr': lr}, commit=False)