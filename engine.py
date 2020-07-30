import sys
import math


import config


def train_one_step(model, images, targets, optimizer):
    images = list(image.to(config.DEVICE) for image in images)
    targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
    
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()

    if not math.isfinite(loss_value):
        print('Loss is {}, stopping training'.format(loss_value))
        print(loss_dict)
        sys.exit(1)
    
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    
    return loss_value


def train_one_epoch(model, data_loader, optimizer, print_freq=5):
    step = 0
    epoch_loss = 0

    for images, targets in data_loader:
        loss = train_one_step(model, images, targets, optimizer)
                
        if step % print_freq == 0:
            print(f'Step {step}, loss: {loss}')
                
        step = step + 1
        epoch_loss = epoch_loss + loss
        
    return epoch_loss