# BATCH_SIZE must larger than 1
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
import torchvision.datasets.cifar
import torchvision.transforms as transforms
import tqdm

BATCH_SIZE = 4
EPOCH = 2
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,

                                        download=True)


class image_title_dataset(Dataset):
    def __init__(self):
        # self.tokens = clip.tokenize([f'image of a {c}' for c in classes])
        self.tokens = clip.tokenize(classes)

    def __len__(self):
        return len(trainset)

    def __getitem__(self, idx):
        image = preprocess(trainset[idx][0])  # Image from PIL module
        title = self.tokens[trainset[idx][1]]
        return image, title


dataset = image_title_dataset()
train_dataloader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)  # Define your own dataloader


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=5e-5,
                       betas=(0.9, 0.98),
                       eps=1e-6,
                       weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.

for epoch in range(EPOCH):
    pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), bar_format=TQDM_BAR_FORMAT)
    running_loss = 0
    # print(f'epochhhhh: {epoch}')
    for i, batch in pbar:
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        running_loss += total_loss.item()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        # log
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        pbar.set_description(
            ('%11s' * 2 + '%11.4g') %
            (f'{epoch + 1}/{EPOCH}', mem, running_loss / (i+1)))


def save():
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, f"weights/model_10.pt")  # just change to your preferred folder/filename


def load():
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
    checkpoint = torch.load("model_checkpoint/model_10.pt")

    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
    checkpoint['model_state_dict']["input_resolution"] = model.input_resolution  # default is 224
    checkpoint['model_state_dict']["context_length"] = model.context_length  # default is 77
    checkpoint['model_state_dict']["vocab_size"] = model.vocab_size

    model.load_state_dict(checkpoint['model_state_dict'])
