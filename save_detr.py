import torch

print("=== Downloading pretrained DETR-ResNet50 model from PyTorch Hub ===")
try:
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    
    # Wrap model inside checkpoint dictionary compatible with attempt_load
    ckpt = {
        'model': model.eval(),
        'ema': None
    }
    
    torch.save(ckpt, 'detr_resnet50.pt')
    print("=== Successfully saved pretrained DETR model to 'detr_resnet50.pt' ===")
except Exception as e:
    print(f"=== Error downloading/saving DETR model: {e} ===")
