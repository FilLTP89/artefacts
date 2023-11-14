import tensorflow as tf
from transformers import TFSegformerForSemanticSegmentation
from metrics import accuracy, precision, recall, f1_score




class SegFormer(tf.keras.Model):
    def __init__(
        self,
        input_shape=(512, 512, 1),
        nb_class=1,
        learning_rate=3e-4,
    ):
        super().__init__()
        self.shape = input_shape
        self.learning_rate = learning_rate


    def build_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model_checkpoint = "nvidia/mit-b0"
        id2label = {0: "good", 1: "artefacts"}
        label2id = {label: id for id, label in id2label.items()}
        num_labels = len(id2label)
        model = TFSegformerForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.MeanIoU(num_classes=2, sparse_y_pred=False),
                     accuracy, precision, recall,],
            )
        return model