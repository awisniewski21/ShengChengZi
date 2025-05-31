
import numpy as np
import torch
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.inception import inception_v3


def mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """ Computes the Mean Absolute Error (L1 loss) """
    return F.l1_loss(pred, true)


def inception_score(image_dataset: Dataset, cuda: bool = True, batch_size: int = 32, resize: bool = False, splits: int = 1):
    """
    Computes the inception score of the generated images.

    Args:
        image_dataset: Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1].
        cuda: Whether or not to run on GPU.
        batch_size: Batch size for feeding into Inception v3.
        resize: Whether to resize images to 299x299.
        splits: Number of splits.

    Returns:
        Tuple of (mean, std) inception scores.
    """
    N = len(image_dataset)
    bs = batch_size
    assert bs > 0, "Batch size must be positive."
    assert N > bs, "Dataset size must be larger than batch size."

    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if torch.cuda.is_available() and not cuda:
        print(f"WARNING: CUDA is available, but cuda=False")

    dataloader = DataLoader(image_dataset, batch_size=bs)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode="bilinear").type(dtype)

    def get_pred(x: torch.Tensor) -> np.ndarray:
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batch_var = Variable(batch)
        bs_i = batch.size()[0]
        preds[i*bs : i*bs+bs_i] = get_pred(batch_var)

    # Compute the mean KL-divergence
    split_scores = []
    for k in range(splits):
        part = preds[k*(N//splits) : (k+1)*(N//splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)