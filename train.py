from time import time
from tqdm import tqdm

def train(model, epoch, loader, optim, device, CONFIG, loss_func):
    log_interval = CONFIG['log_interval']
    model.train()
    start = time()
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, batch in pbar:
        modelout = model(batch.to(device))
        pred, c_loss = loss_func(modelout, batch_size=loader.batch_size)
        bpr_loss = loss_func(pred, batch_size=loader.batch_size)
        loss = bpr_loss + 0.04 * c_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % log_interval == 0:
            print('U-B Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i+1) * loader.batch_size, len(loader.dataset),
                100. * (i+1) / len(loader), loss))
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time()-start)))
    return loss

