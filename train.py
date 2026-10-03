import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import FGVCAircraft
from models import HierarchicalNet

def train_one_epoch(model, dataloader, optimizer, criterion_v, criterion_f, criterion_m, device, alpha, beta, gamma):
    model.train()
    running_loss = 0.0
    correct_v = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        labels_v = batch['variant'].to(device)
        labels_f = batch['family'].to(device)
        labels_m = batch['maker'].to(device)
        
        optimizer.zero_grad()
        
        out_v, out_f, out_m = model(images)
        
        loss_v = criterion_v(out_v, labels_v)
        loss_f = criterion_f(out_f, labels_f)
        loss_m = criterion_m(out_m, labels_m)
        
        # Combined loss
        loss = gamma * loss_v + beta * loss_f + alpha * loss_m
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, predicted_v = torch.max(out_v.data, 1)
        total += labels_v.size(0)
        correct_v += (predicted_v == labels_v).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct_v / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion_v, criterion_f, criterion_m, device, alpha, beta, gamma):
    model.eval()
    running_loss = 0.0
    correct_v, correct_f, correct_m = 0, 0, 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels_v = batch['variant'].to(device)
            labels_f = batch['family'].to(device)
            labels_m = batch['maker'].to(device)
            
            out_v, out_f, out_m = model(images)
            
            loss_v = criterion_v(out_v, labels_v)
            loss_f = criterion_f(out_f, labels_f)
            loss_m = criterion_m(out_m, labels_m)
            
            loss = gamma * loss_v + beta * loss_f + alpha * loss_m
            running_loss += loss.item() * images.size(0)
            total += labels_v.size(0)
            
            _, predicted_v = torch.max(out_v.data, 1)
            _, predicted_f = torch.max(out_f.data, 1)
            _, predicted_m = torch.max(out_m.data, 1)
            
            correct_v += (predicted_v == labels_v).sum().item()
            correct_f += (predicted_f == labels_f).sum().item()
            correct_m += (predicted_m == labels_m).sum().item()
            
    epoch_loss = running_loss / total
    return epoch_loss, correct_v / total, correct_f / total, correct_m / total

def main():
    parser = argparse.ArgumentParser(description="Train Aircraft Classifier")
    parser.add_argument('--model', type=str, default='convnext_tiny', help='Backbone model name from timm')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for Maker loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for Family loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for Variant loss')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_dataset = FGVCAircraft(root_dir="./data", split='train', transform=transform, download=True)
    val_dataset = FGVCAircraft(root_dir="./data", split='val', transform=transform, download=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    num_variants = train_dataset.get_num_classes('variant')
    num_families = train_dataset.get_num_classes('family')
    num_makers = train_dataset.get_num_classes('maker')
    
    print(f"Classes - Variants: {num_variants}, Families: {num_families}, Makers: {num_makers}")
    
    model = HierarchicalNet(args.model, num_variants, num_families, num_makers, pretrained=True)
    model = model.to(device)
    
    criterion_v = nn.CrossEntropyLoss()
    criterion_f = nn.CrossEntropyLoss()
    criterion_m = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc_v = train_one_epoch(
            model, train_loader, optimizer, criterion_v, criterion_f, criterion_m, 
            device, args.alpha, args.beta, args.gamma
        )
        
        val_loss, val_acc_v, val_acc_f, val_acc_m = evaluate(
            model, val_loader, criterion_v, criterion_f, criterion_m, 
            device, args.alpha, args.beta, args.gamma
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Variant Acc: {train_acc_v:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc (V/F/M): {val_acc_v:.4f} / {val_acc_f:.4f} / {val_acc_m:.4f}")
        
        if val_acc_v > best_val_acc:
            best_val_acc = val_acc_v
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model.pth'))
            print("Saved new best model!")

if __name__ == '__main__':
    main()
