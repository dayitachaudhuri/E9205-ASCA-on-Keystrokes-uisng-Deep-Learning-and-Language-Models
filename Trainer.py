import torch
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None, early_stopping_patience=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler 
        self.early_stopping_patience = early_stopping_patience

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, targets, device_ids in self.train_loader:
            data, targets, device_ids = data.to(self.device), targets.to(self.device), device_ids.to(self.device)
            if len(data.shape) == 3:
                data = data.unsqueeze(1)
            self.optimizer.zero_grad()
            outputs = self.model(data, device_ids)
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
            for data, targets, device_ids in self.val_loader:
                data, targets, device_ids = data.to(self.device), targets.to(self.device), device_ids.to(self.device)
                if len(data.shape) == 3:
                    data = data.unsqueeze(1)
                outputs = self.model(data, device_ids)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train(self, num_epochs, save_path=None, best_save_path=None, resume=False, load_path=None):
        start_epoch = 0
        best_val_acc = 0.0
        patience_counter = 0

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        if resume and load_path and os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Store metrics
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if self.scheduler:
                self.scheduler.step(val_acc)

            if save_path:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_acc': best_val_acc
                }, save_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if best_save_path:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'best_val_acc': best_val_acc
                    }, best_save_path)
                    print(f"Saved best model at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. No improvement in validation accuracy for {self.early_stopping_patience} epochs.")
                    break

        return history