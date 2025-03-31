import torch
import os

##############################################
# Define a Trainer Class for Model Training
##############################################

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, targets in self.train_loader:
            # Move data to device and ensure it has a channel dimension
            data, targets = data.to(self.device), targets.to(self.device)
            if len(data.shape) == 3:  # add channel dimension if missing
                data = data.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                if len(data.shape) == 3:
                    data = data.unsqueeze(1)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self, num_epochs, save_path=None, resume=False, load_path=None):
        start_epoch = 0
        best_val_acc = 0.0
        
        if resume and load_path and os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Resuming training from epoch {start_epoch}")
            
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save checkpoint if there is an improvement in validation accuracy.
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_acc': best_val_acc
                }, save_path)
                print(f"Saved best model at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}")