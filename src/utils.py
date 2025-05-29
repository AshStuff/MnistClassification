import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({'Loss': loss.item()})

def evaluate(model, device, data_loader, set_name="Test"):
    model.eval()
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            
            # Debug prints
            print(f"Predictions: {pred.view(-1)[:10]}")
            print(f"Targets: {target[:10]}")
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    
    print(f'\n{set_name} set: Average loss: {loss:.4f}, '
          f'Accuracy: {correct}/{len(data_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
