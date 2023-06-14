import segmentation_models as sm
# Segmentation Models: using `keras` framework.



def segmentation_Unet(input_shape, learning_rate, **kwargs):
    model = sm.Unet('resnet34',
                    input_shape=input_shape,
                    classes=1, 
                    activation='sigmoid')
    
    
    model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
    )

    return model    