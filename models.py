import torch
import torch.nn as nn
import timm

class HierarchicalNet(nn.Module):
    def __init__(self, model_name, num_variants, num_families, num_makers, pretrained=True):
        super(HierarchicalNet, self).__init__()
        
        # Load backbone without the final classifier
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the feature dimension of the backbone
        # We can pass a dummy tensor to get the feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
            
        # Define the three separate classification heads
        self.head_variant = nn.Linear(feature_dim, num_variants)
        self.head_family = nn.Linear(feature_dim, num_families)
        self.head_maker = nn.Linear(feature_dim, num_makers)
        
    def forward(self, x):
        features = self.backbone(x)
        
        out_variant = self.head_variant(features)
        out_family = self.head_family(features)
        out_maker = self.head_maker(features)
        
        return out_variant, out_family, out_maker

if __name__ == '__main__':
    # Test the model
    model = HierarchicalNet('convnext_tiny', num_variants=100, num_families=70, num_makers=30)
    print("Model created successfully.")
    
    dummy_input = torch.randn(2, 3, 224, 224)
    out_v, out_f, out_m = model(dummy_input)
    print(f"Variant shape: {out_v.shape}")
    print(f"Family shape: {out_f.shape}")
    print(f"Maker shape: {out_m.shape}")
