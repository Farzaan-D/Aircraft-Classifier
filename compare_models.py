import torch
import time
from models import HierarchicalNet
from dataset import FGVCAircraft
from torch.utils.data import DataLoader
from torchvision import transforms

def benchmark_model(model_name, dataloader, device):
    print(f"\n--- Benchmarking {model_name} ---")
    
    # We use dummy sizes for the classes
    model = HierarchicalNet(model_name, num_variants=100, num_families=70, num_makers=30)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    
    # Measure inference speed
    model.eval()
    dummy_input = torch.randn(16, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    # Timing
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end = time.time()
    
    avg_batch_time = (end - start) / 10
    fps = 16 / avg_batch_time
    print(f"Inference Speed: {fps:.2f} images/second (Batch size 16)")
    
    return total_params, fps

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")
    
    # CNN Model: ConvNeXt-Tiny
    cnn_params, cnn_fps = benchmark_model('convnext_tiny', None, device)
    
    # Transformer Model: ViT-Tiny or Swin
    vit_params, vit_fps = benchmark_model('vit_tiny_patch16_224', None, device)
    
    print("\n=== SUMMARY ===")
    print(f"CNN (ConvNeXt-Tiny)   : {cnn_params/1e6:.2f}M Params | {cnn_fps:.2f} FPS")
    print(f"Transformer (ViT-Tiny): {vit_params/1e6:.2f}M Params | {vit_fps:.2f} FPS")

if __name__ == '__main__':
    main()
