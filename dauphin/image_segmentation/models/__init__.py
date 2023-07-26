from dauphin.image_segmentation.models.architectures import GadgetronResUnet18, UNet_model, fcn_resnet50, FCNHead

ALL_MODELS = {
    "UNet": UNet_model,
    "GadgetronResUnet18": GadgetronResUnet18,
    "Res50": fcn_resnet50,
    "FCNHead": FCNHead,
}

MODEL_DIM = {"GadgetronResUnet18": "2d", 
             "UNet": "2d",
             "Res50": "2d",
}
