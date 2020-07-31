import sys
import math


import config
from metrics import batch_average_precision


def train_one_step(model, images, targets, optimizer):
    images = list(image.to(config.DEVICE) for image in images)
    targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
    
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()

    if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        print(loss_dict)
        sys.exit(1)
    
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    
    return loss_value


def train_one_epoch(model, data_loader, optimizer, epoch, print_freq=5):
    print(f"Starting epoch {epoch}/{config.EPOCHS-1}")
    model.train()

    step = 0
    epoch_loss = 0

    for images, targets in data_loader:
        loss = train_one_step(model, images, targets, optimizer)
                
        if step % print_freq == 0:
            print(f"Step {step}/{len(data_loader)-1}, loss: {loss}")
                
        step = step + 1
        epoch_loss = epoch_loss + loss

    epoch_loss /= len(data_loader)

    print(f"Epoch {epoch}/{config.EPOCHS-1}, loss: {epoch_loss}")
        
    return epoch_loss


def evaluate(model, data_loader, epoch):
    print(f"Starting evaluation")
    model.eval()

    m_ap = 0

    for images, targets in data_loader:
        images = list(image.to(config.DEVICE) for image in images)
        outputs = model(images)
        outputs = [{k: v for k, v in t.items()} for t in outputs]
        targets = [{k: v for k, v in t.items()} for t in targets]

        m_ap += batch_average_precision(outputs, targets)

    m_ap /= len(data_loader)

    print(f"Epoch {epoch}/{config.EPOCHS-1}, mAP: {m_ap}")

    return m_ap