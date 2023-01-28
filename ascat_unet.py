import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import time

TRAIN_DIR = os.path.join(os.getcwd(), "data", "ATE_IMG", "train")
TEST_DIR = os.path.join(os.getcwd(), "data", "ATE_IMG", "test")
RAIN_CUTOFF = 50

assert torch.cuda.is_available()

# Used as a building block of the U-Net
class ConvBlock(Module):
  def __init__(self, in_channels, out_channels, up:bool=False):
    super(ConvBlock, self).__init__()

    self.c1 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=(3,3), padding=(1,1))
    self.c2 = Conv2d(in_channels=out_channels, out_channels=out_channels,
                        kernel_size=(3,3), padding=(1,1))
    self.up = up
    if self.up:
      self.up_conv = ConvTranspose2d(out_channels, out_channels//2, kernel_size=2,
                                   stride=2)
  def forward(self, x):
    out1 = F.relu(self.c1(x))
    out2 = F.relu(self.c2(out1))
    if self.up:
      out2 = self.up_conv(out2)
    return out2

# The Network is used to detect the rain
class UNet(Module):
  def __init__(self, in_channels: int):
    super(UNet, self).__init__()
    self.conv1 = ConvBlock(in_channels,64)
    self.conv2 = ConvBlock(64,128)
    self.conv3 = ConvBlock(128,256)
    self.conv4 = ConvBlock(256,512)
    self.up_conv1 = ConvBlock(512,1024,up=True)
    self.up_conv2 = ConvBlock(1024,512,up=True)
    self.up_conv3 = ConvBlock(512,256,up=True)
    self.up_conv4 = ConvBlock(256,128,up=True)
    self.out_conv1 = Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
    self.out_conv2 = Conv2d(64, 64, kernel_size=(3,3), padding=(1,1))
    self.out_conv3 = Conv2d(64, 2, kernel_size=(1,1), padding=(0,0))

    self.down = MaxPool2d(2)
  def forward(self, input):
    # Downsampled Portion
    out1 = self.conv1(input)
    out2 = self.conv2(self.down(out1))
    out3 = self.conv3(self.down(out2))
    out4 = self.conv4(self.down(out3))

    # Middle Portion
    out5 = self.up_conv1(self.down(out4))

    # Upconvolution Portion
    out6 = self.up_conv2(torch.cat((out4, out5), 1))
    out7 = self.up_conv3(torch.cat((out3, out6), 1))
    out8 = self.up_conv4(torch.cat((out2, out7), 1))

    #output stuff
    out9=F.relu(self.out_conv1(torch.cat((out1, out8), 1)))
    out10 = F.relu(self.out_conv2(out9))
    return self.out_conv3(out10)

# Accuracy Measurement Functions
def pixelwise_accuracy(y, y_truth):
    b = y.size(0)
    return (y.argmax(1).squeeze() == y_truth.squeeze()).float().view(b, -1).mean(1)
  
def IoU(y, y_truth):
    y[:,1] = y[:,1] > .5
    intersection = torch.sum(y[:,1] * y_truth).item()
    union = torch.sum((y[:,1] + y_truth) >= 1).item()
    # Handle div by 0
    if union == 0:
        return -1
    else:
        return intersection/union

train_dataset = ATE_dataset(TRAIN_DIR, oversample=True, rain_cutoff=RAIN_CUTOFF)
val_dataset = ATE_dataset(TEST_DIR)

test_im, test_gt = val_dataset[172]
test_im = test_im.float().unsqueeze(0).cuda()
test_sigma = test_im[:3]

model = UNet(in_channels=9)
model = model.cuda()

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_accs_pw = []
val_accs_pw = []
test_preds = []

def scope():
    #your code for calling dataset and dataloader
    gc.collect()
    print(torch.cuda.memory_allocated(0) / 1e9)

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=6, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=6, pin_memory=True, shuffle=True)

    # Call your model, figure out loss and accuracy

    # get the mean of losses from running the validation loader
    vals = []
    val_a = []
    val_a_pw = 0
    with torch.no_grad():
      for x, y_truth in val_loader:
        gc.collect()
        y_truth = y_truth[:,2]
        # print(y_truth.shape)
        x, y_truth = x.float().cuda(), y_truth.cuda() 
        y_hat = model(x.cuda())
        vals.append(objective(y_hat, y_truth.long()).item())
        iou = IoU(torch.softmax(y_hat,1), y_truth.float())
        val_a_pw += pixelwise_accuracy(y_hat, y_truth).sum().item()
        if iou != -1:
          val_a.append(iou)
      val_losses.append((len(train_losses), np.mean(vals)))
      val_accs.append((len(train_losses), np.mean(val_a)))
      val_accs_pw.append(val_a_pw / len(val_dataset))

    num_epochs = 20
    for epoch in range(num_epochs):
      current_step = 0
      total_steps = len(train_loader)

      for batch, (x, y_truth) in enumerate(train_loader):
        gc.collect()
        y_truth = y_truth[:,2]
        x, y_truth = x.float().cuda(), y_truth.cuda() 

        optimizer.zero_grad()
        y_hat = model(x)
        train_acc = IoU(torch.softmax(y_hat, 1), y_truth.float())

        loss = objective(y_hat, y_truth.long()) / (10 * max(.0001, train_acc))
        
        loss.backward()

        train_losses.append(loss.item())
        if train_acc != -1:
          train_accs.append(train_acc)
          train_accs_pw.append(pixelwise_accuracy(y_hat, y_truth).mean())
        print('epoch: {0}, loss: {1:04.4f}, accuracy: {2:.5f}, completion: {3} / {4}'.format(epoch, loss.item(), train_acc, current_step, total_steps))

        optimizer.step()
        current_step += 1

      # get the mean of losses from running the validation loader
      vals = []
      val_a = []
      val_a_pw = 0
      with torch.no_grad():
        for x, y_truth in val_loader:
          gc.collect()
          y_truth = y_truth[:,2]
          x, y_truth = x.float().cuda(), y_truth.cuda() 
          y_hat = model(x.cuda())
          vals.append(objective(y_hat, y_truth.long()).item())
          iou = IoU(torch.softmax(y_hat,1), y_truth.float())
          val_a_pw += pixelwise_accuracy(y_hat, y_truth).sum().item()
          if iou != -1:
            val_a.append(iou)
        val_losses.append((len(train_losses), np.mean(vals)))
        val_accs.append((len(train_losses), np.mean(val_a)))
        val_accs_pw.append(val_a_pw / len(val_dataset))

        # Collect predictions on the test image
        test_preds.append(model(test_im).cpu())
    
begin = time.time()
scope()
print("Runtime =", time.time()-begin)

# Your plotting code here
x, val = zip(*val_losses) # y is validations, zip just splits the tuples in validations

plt.figure()
plt.plot(train_losses, label='train')
plt.plot(x, val, label='val')
plt.title("Cross Entropy Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()

x, val_a = zip(*val_accs)
plt.figure()
plt.plot(train_accs, label="train")
plt.plot(x, val_a, label="val")
plt.title("IoU Accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.legend()

val_a_pw = val_accs_pw
plt.figure()
# plt.plot(train_accs_pw, label="train")
plt.plot(x, val_a_pw, label="val")
plt.title("Pixelwise Accuracy")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# Code for testing prediction on an image

# Original image and ground-truth segmentation map
# Change from channel first (required by PyTorch functions) to channel last (required by PyPlot)
plt.figure()
plt.imshow(np.rollaxis(test_sigma.cpu().squeeze(0).numpy(),0,3))
plt.title("Original")
plt.figure()
plt.imshow(test_gt[-1].numpy(), cmap="gray")
plt.title("Ground-Truth")

# Predictions performed at the end of each epoch
for epoch, test_pred in enumerate(test_preds):
  test_pred_np = np.rollaxis(test_pred.cpu().detach().squeeze().numpy(), 0, 3)
  # Argmax along channel dim to turn raw class scores per pixel into class prediction per pixel
  plt.figure()
  plt.imshow(test_pred_np.argmax(2), cmap="gray") 
  plt.title("Prediction at epoch {} w/ {}% accuracy".format(epoch, int(100*val_accs_pw[epoch+1])))

plt.show()