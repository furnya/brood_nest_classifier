from typing import Union
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


class BNClassifier:
    PATH_LOAD = files('bn_classifier').joinpath('weights/best-augment.pth')
    net = CellModel()
    net.load_state_dict(torch.load(PATH_LOAD, map_location=torch.device('cpu')))
    net.double()

    _best_thresholds = torch.Tensor([0.37, 0.29, 0.62, 0.66, 0.29, 0.55])
    _radius = 5
    binary_classes_readable = [
        "empty",
        "egg",
        "larva",
        "young pupa",
        "old pupa",
        "bee head"
    ]
    _scale_factor = 64/400

    @staticmethod
    def get_trained_model() -> nn.Module:
        """Return the model with the trained weights applied

        Returns:
            nn.Module: The model
        """
        return BNClassifier.net

    @staticmethod
    def get_untrained_model() -> nn.Module:
        """Return the model with the default weights

        Returns:
            nn.Module: The model
        """
        return CellModel()

    @staticmethod
    def feed_image(image: Union[str, torch.Tensor]) -> torch.Tensor:
        """Feed one image to the network and return the raw output

        Args:
            image (str|Tensor): The path to the image or the image tensor

        Returns:
            torch.Tensor: The prediction of the network
        """
        if type(image) == str:
            image = torch.Tensor(io.imread(image)).double()[:, :, :3].permute(2, 0, 1)
        elif type(image) == torch.Tensor:
            assert len(image.shape) == 3
            if image.shape[2] == 3 or image.shape[2] == 4:
                image = image.permute(2, 0, 1)
            assert image.shape[0] == 3 or image.shape[0] == 4
            image = image.double()[:, :, :3]
        else:
            raise TypeError(f"Image type {type(image)} not supported")

        image = transforms.functional.resize(
            image, (int(image.shape[1]*BNClassifier._scale_factor), int(image.shape[2]*BNClassifier._scale_factor)))
        image = transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))(image)

        pred_scores = BNClassifier.net(image.unsqueeze(0)).detach().squeeze(0).permute(1, 2, 0)
        return pred_scores

    @staticmethod
    def feed_images_from_dir(img_dir: str) -> dict[str, torch.Tensor]:
        """Feed all images from a directory to the network and return the predictions

        Args:
            image (str|Tensor): The path to the image or the image tensor

        Returns:
            dict: A dictionary with the filenames as keys and the predictions as values
        """
        out = {}
        for f in os.listdir(img_dir):
            out[f] = BNClassifier.feed_image(os.path.join(img_dir, f))
        return out

    @staticmethod
    def get_prediction_points(image: Union[str, torch.Tensor], classes_as_strings=False) -> tuple[list[list[float]], list[Union[int, str]], list[str]]:
        """Feed one image to the network and return the prediction points (local maxima filtered)

        Args:
            image (str|Tensor): The path to the image or the image tensor

        Returns:
            tuple: Tuple of (points, classes [as indices or strings], class values)
        """
        pred_scores = BNClassifier.feed_image(image)

        scores_filtered = torch.zeros((pred_scores.shape[0], pred_scores.shape[1]))
        score_classes = []
        for y in range(pred_scores.shape[0]):
            score_classes.append([])
            for x in range(pred_scores.shape[1]):
                if len(pred_scores[y, x][pred_scores[y, x] >= BNClassifier._best_thresholds]) > 0:
                    scores_filtered[y, x] = pred_scores[y, x][pred_scores[y, x] >= BNClassifier._best_thresholds].mean()
                    score_classes[y].append(pred_scores[y, x])
                else:
                    scores_filtered[y, x] = 0
                    score_classes[y].append(torch.zeros(len(pred_scores[y, x])))

        scores_copy = scores_filtered.clone()
        for y in range(scores_filtered.shape[0]):
            for x in range(scores_filtered.shape[1]):
                scores_copy[y:y+BNClassifier._radius+1, x:x+BNClassifier._radius+1] = torch.max(
                    scores_copy[y:y+BNClassifier._radius+1, x:x+BNClassifier._radius+1], scores_filtered[y:y+BNClassifier._radius+1, x:x+BNClassifier._radius+1].max())
        score_classes = [[BNClassifier.multi_score_to_readable(
            t, BNClassifier._best_thresholds.numpy()) for t in l] for l in score_classes]

        scores_copy[scores_copy != scores_filtered] = 0

        org_pad = 0
        pad = 31
        scale = 4
        point_pad = 0
        pred_points = []
        pred_class_indices = []
        pred_classes = []
        class_values = []
        for y in range(point_pad, scores_copy.shape[0]-point_pad):
            for x in range(point_pad, scores_copy.shape[1]-point_pad):
                if scores_copy[y, x] > 0:
                    pred_points.append(((1/BNClassifier._scale_factor)*(scale*x+pad)+org_pad,
                                       (1/BNClassifier._scale_factor)*(scale*y+pad)+org_pad))
                    if score_classes[y][x] not in class_values:
                        class_values.append(score_classes[y][x])
                    pred_class_indices.append(class_values.index(score_classes[y][x]))
                    pred_classes.append(score_classes[y][x])
        pred_points = np.array(pred_points)
        return pred_points, pred_classes if classes_as_strings else pred_class_indices, class_values

    @staticmethod
    def multi_score_to_readable(multi_score: Union[np.ndarray, torch.Tensor], thresholds=np.array([0.5] * len(binary_classes_readable))) -> str:
        """Convert a multi-label score to a readable string

        Args:
            multi_score (array-like): The multi-label score (length 6)

        Returns:
            str: The converted readable label
        """
        if type(multi_score) == torch.Tensor:
            multi_score = multi_score.cpu().detach().numpy()
        if (multi_score - thresholds).max() < 0:
            return "(unknown)"
        if np.argmax(multi_score - thresholds) == BNClassifier.binary_classes_readable.index("empty"):
            return "(empty)"
        return (
            "(" + ", ".join([BNClassifier.binary_classes_readable[i] for i, x in enumerate(multi_score)
                             if x-thresholds[i] >= 0 and i != BNClassifier.binary_classes_readable.index("empty")]) + ")"
        )
