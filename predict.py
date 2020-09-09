import sys
from argparse import ArgumentParser
from torchvision import transforms
import torchvision.models as models
from utils import *
from models import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--model-path", "-m", help="Path to a trained model.")
    parser.add_argument("--video-path", "-v", help="Path to a video file.")
    parser.add_argument("--frames-cnt", "-f", help="Number of video frames for random selection.", default=16, type=int)
    parser.add_argument("--model-type", help="Model to run. Two options: 'cnn-avg' or 'cnn-rnn'.",
                        default="cnn-avg")
    parser.add_argument("--bilstm", action="store_true", help="Whether the LSTM is bidirectional")
    parser.add_argument("--cnn-model", help="CNN pretrained Model to use. Two options: 'resnet18' or 'resnet34'.",
                        default="resnet18")
    parser.add_argument("--gpu", action="store_true", help="Whether to run using GPU or not.")
    return parser.parse_args()


def predict(model, video, device):
    model.eval()
    inputs, _ = collate_fn([[video, torch.tensor([0])]])
    videos = inputs.to(device)

    with torch.no_grad():
        pred_labels = model(videos).cpu()
    prediction = pred_labels.numpy().argmax()
    return prediction


def main(args):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    cnn_model = models.resnet34(pretrained=True) if args.cnn_model == "resnet34" else models.resnet18(pretrained=True)
    if args.model_type == "cnn-rnn":
        model = CNNtoRNNModel(cnn_model, frames_cnt=args.frames_cnt, bidirectional=args.bilstm)
    else:
        model = AvgCNNModel(cnn_model, frames_cnt=args.frames_cnt)
    with open(args.model_path, "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)
    model.to(device)
    set_frames_cnt(args.frames_cnt)

    video = read_video(args.video_path, train_transforms)

    prediction = predict(model, video, device)
    print(prediction)


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
