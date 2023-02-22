import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from IPython import display
import matplotlib.pyplot as plt


def unpack_retriever_tfidf(batch):
    questions = batch['question']
    documents = batch['paragraph']
    scores = batch['scores'].to(torch.int64)
    return (questions, documents), scores

def plot_train_process(train_loss, val_loss, title_suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].set_title(' '.join(['Loss', title_suffix]))
    axes[0].plot(train_loss, label='train')
    axes[0].plot(val_loss, label='validation')
    axes[0].legend()

    plt.show()
    

def train_loop(model, train_dataloader, optimizer, criterion, batch_transform):
    losses = []
    for batch in train_dataloader:
        X_batch, y_batch = batch_transform(batch)
        
        output = model(X_batch)
        loss = criterion(output, y_batch)
        losses.append(loss)
        loss.backward()
        
        optimizer.step()
    return torch.Tensor(losses).mean()

def val_loop(model, dataloader, criterion, batch_transform):
    losses = []
    for batch in dataloader:
        X_batch, y_batch = batch_transform(batch)
        output = model(X_batch)
        loss = criterion(output, y_batch)
        losses.append(loss)
    return torch.Tensor(losses).mean()

def test_loop(model, dataloader, batch_transform):
    y_pred = []
    y_gt = []
    for batch in dataloader:
        X_batch, y_batch = batch_transform(batch)
        output = model(X_batch)
        y_pred.append(output)
        y_gt.append(y_batch)
    return y_pred, y_gt

def train(model, train_dataloader, val_dataloader, optimizer, criterion, batch_transform, 
          epochs = 100, plot_loss = True, every_batch = 5):
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    for e in tqdm(range(1, epochs+1)):
        loss = train_loop(model, train_dataloader, optimizer, criterion, batch_transform)
        if plot_loss and e % every_batch == 0:
            with torch.no_grad():
                val_loss = val_loop(model, val_dataloader, criterion, batch_transform)
            
            train_loss_per_epoch.append(loss)
            val_loss_per_epoch.append(val_loss)

            display.clear_output(wait=True)
            plot_train_process(train_loss_per_epoch, val_loss_per_epoch)
    output = {}
    output['model'] = model
    output['train_loss'] = train_loss_per_epoch
    output['val_loss'] = val_loss_per_epoch
    return output

def train_tfidf_linear_score(model, train_dataloader, val_dataloader, optimizer, criterion, 
                             epochs = 100, every_batch = 5, plot_loss = True):
    return train(model, train_dataloader, val_dataloader, optimizer, criterion, unpack_retriever_tfidf, 
                 epochs=epochs, every_batch = every_batch, plot_loss = plot_loss)