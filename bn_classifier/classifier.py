import torch
import torch.nn as nn
from importlib_resources import files
from skimage import io
import torchvision.transforms as transforms
import os
import numpy as np

class CellModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 3)  # flatten all dimensions except batch
        return x

PATH_LOAD = files('bn_classifier').joinpath('weights/best-augment.pth')
net = CellModel()
net.load_state_dict(torch.load(PATH_LOAD, map_location=torch.device('cpu')))
net.double()

_best_thresholds = torch.Tensor([0.37, 0.29, 0.62, 0.66, 0.29, 0.55])
_radius = 5
_binary_classes_readable = [
        "empty",
        "egg",
        "larva",
        "young pupa",
        "old pupa",
        "bee head"
    ]
_scale_factor = 64/400
class BNClassifier:
    
    @staticmethod
    def feed_image(img_path: str):
        """Feed one image to the network and return the raw output

        Args:
            img_path (str): The path to the image

        Returns:
            torch.Tensor: The prediction of the network
        """
        image = torch.Tensor(io.imread(img_path)).double()[:, :, :3].permute(2, 0, 1)
        
        image = transforms.functional.resize(image, (int(image.shape[1]*_scale_factor),int(image.shape[2]*_scale_factor)))
        image = transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))(image)

        pred_scores = net(image.unsqueeze(0)).detach().squeeze(0).permute(1,2,0)
        return pred_scores
    
    @staticmethod
    def feed_images(img_dir: str):
        """Feed all images from a directory to the network and return the predictions

        Args:
            img_dir (str): The directory path

        Returns:
            dict: A dictionary with the filenames as keys and the predictions as values
        """
        out = {}
        for f in os.listdir(img_dir):
            out[f] = BNClassifier.feed_image(os.path.join(img_dir,f))
        return out

    @staticmethod
    def get_prediction_points(img_path: str):
        """Feed one image to the network and return the prediction points (local maxima filtered)

        Args:
            img_path (str): The path to the image

        Returns:
            tuple: Tuple of (points, class_values)
        """
        pred_scores = BNClassifier.feed_image(img_path)
        
        scores_filtered = torch.zeros((pred_scores.shape[0],pred_scores.shape[1]))
        score_classes = []
        for y in range(pred_scores.shape[0]):
                score_classes.append([])
                for x in range(pred_scores.shape[1]):
                    if len(pred_scores[y,x][pred_scores[y,x] >= _best_thresholds]) > 0:
                        scores_filtered[y,x] = pred_scores[y,x][pred_scores[y,x] >= _best_thresholds].mean()
                        score_classes[y].append(pred_scores[y,x])
                    else:
                        scores_filtered[y,x] = 0
                        score_classes[y].append(torch.zeros(len(pred_scores[y,x])))
        
        scores_copy = scores_filtered.clone()
        for y in range(scores_filtered.shape[0]):
            for x in range(scores_filtered.shape[1]):
                scores_copy[y:y+_radius+1,x:x+_radius+1] = torch.max(scores_copy[y:y+_radius+1,x:x+_radius+1], scores_filtered[y:y+_radius+1,x:x+_radius+1].max())
        score_classes = [[BNClassifier.multi_score_to_readable(t, _best_thresholds.numpy()) for t in l] for l in score_classes]
        
        scores_copy[scores_copy != scores_filtered] = 0
        
        org_pad = 0
        pad = 31
        scale = 4
        point_pad = 0
        pred_points = []
        pred_classes = []
        # scores_copy[scores_copy < best_thresholds[i]] = 0
        for y in range(point_pad, scores_copy.shape[0]-point_pad):
            for x in range(point_pad, scores_copy.shape[1]-point_pad):
                if scores_copy[y,x] > 0:
                    pred_points.append(((1/_scale_factor)*(scale*x+pad)+org_pad,(1/_scale_factor)*(scale*y+pad)+org_pad))
                    pred_classes.append(score_classes[y][x])
        pred_points = np.array(pred_points)
        return pred_points, pred_classes

    @staticmethod
    def multi_score_to_readable(multi_score, thresholds=np.array([0.5] * len(_binary_classes_readable))):
        if type(multi_score) == torch.Tensor:
            multi_score = multi_score.cpu().detach().numpy()
        if (multi_score - thresholds).max() < 0:
            return "(unknown)"
        if np.argmax(multi_score) == _binary_classes_readable.index("empty"):
            return "(empty)"
        return (
            "(" + ", ".join([_binary_classes_readable[i] for i, x in enumerate(multi_score) if x-thresholds[i] >= 0 and i!=_binary_classes_readable.index("empty")]) + ")"
        )