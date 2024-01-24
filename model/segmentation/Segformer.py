import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor,SegformerImageProcessor




class SegFormer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        id2label = {0: "background", 1: "dental artefact"}
        label2id = {v : k for k, v in id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)

    def extract_shape(self,x):
        if x.ndim == 3:
            shape_x = (x.shape[1], x.shape[2])
        if x.ndim == 4:
            shape_x = (x.shape[2], x.shape[3])
        return shape_x
    
    def forward(self,pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=self.extract_shape(x), # (height, width)
            mode='bilinear',
            align_corners=False
        )
        return upsampled_logits
    
    def predict_mask(self,x):
        logits = self.forward(x)
        mask = torch.argmax(logits, dim=1)
        return mask
    
if __name__ == "__main__":
    x = torch.rand(1,3,1024,1024)/255 
    model = SegFormer()
    from transformers import SegformerImageProcessor
    processor = SegformerImageProcessor()
    processed = processor(x, return_tensors="pt")
    print(processed["pixel_values"].shape)
    logits = model(**processed)
    print(logits.shape)