"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import sys
import pandas as pd
from argparse import ArgumentParser
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
from models import *
from torch import nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=16, type=int)
    parser.add_argument("--frames-cnt", "-f", default=32, type=int)
    parser.add_argument("--model-type", "-m", default="cnn-avg")
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--predict", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for inputs, labels in tqdm.tqdm(loader, total=len(loader), desc="training...", position=0, leave=True):
        videos = inputs.to(device)

        pred_labels = model(videos).cpu()
        loss = loss_fn(pred_labels, labels)
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for inputs, labels in tqdm.tqdm(loader, total=len(loader), desc="validation...", position=0, leave=True):
        videos = inputs.to(device)

        with torch.no_grad():
            pred_labels = model(videos).cpu()
        loss = loss_fn(pred_labels, labels)
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset),))
    labels = np.zeros((len(loader.dataset),))
    for i, (inputs, label) in enumerate(
            tqdm.tqdm(loader, total=len(loader), desc="test prediction...", position=0, leave=True)):
        videos = inputs.to(device)

        with torch.no_grad():
            pred_labels = model(videos).cpu()
        prediction = pred_labels.numpy().argmax()  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction.reshape(-1)
        labels[i * loader.batch_size: (i + 1) * loader.batch_size] = label
    return predictions, labels


def main(args):
    # 1. prepare data & models
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Creating model...")
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    if args.model_type == "cnn-rnn":
        model = CNNtoRNNModel(frames_cnt=args.frames_cnt)
    else:
        model = AvgCNNModel(frames_cnt=args.frames_cnt)

    model.to(device)
    set_frames_cnt(args.frames_cnt)

    if not args.predict:
        # 1. prepare data & models
        print("Reading data...")
        train_dataset = VideoDataset(args.data, train_transforms, split="train")
        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True,
                                           shuffle=True, drop_last=True, collate_fn=collate_fn)
        val_dataset = VideoDataset(args.data, train_transforms, split="val")
        val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True,
                                         shuffle=False, drop_last=False, collate_fn=collate_fn)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=6, factor=0.3)
        criterion = nn.CrossEntropyLoss()

        # 2. train & validate
        print("Ready for training...")
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            train_loss = train(model, train_dataloader, criterion, optimizer, device=device)
            val_loss = validate(model, val_dataloader, criterion, device=device)
            lr_scheduler.step(val_loss)
            print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with open(f"{args.name}_best.pth", "wb") as fp:
                    torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = VideoDataset(args.data, train_transforms, split="test")
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1, pin_memory=True,
                                      shuffle=False, drop_last=False, collate_fn=collate_fn)

    with open(f"{args.name}_best.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions, test_labels = predict(model, test_dataloader, device)
    pd.DataFrame({"video_names": test_dataset.video_names,
                  "predictions": test_predictions,
                  "labels": test_labels}).to_csv(f"{args.name}_test_predictions.csv", index=False)


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
