import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd
from colour import Color
import itertools
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from PIL import Image
import requests


NUM_INPUT_CHANNELS = 3
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 50
VAL_INTERVAL = LOG_INTERVAL
NUM_TEST_TASKS = 0

rng = np.random.default_rng()
mode = 'train'


def get_data(task_idx, inp, mode='train'):
    if (mode == 'train'):
        task_samples = rng.choice(84, 16)
        task_slice = task_idx[task_samples]
    else:
        task_slice = [(9, 10, 11)]

    task_data = []
    # print(len(task_slice))
    for task in task_slice:
        support_set = []
        query_set = []
        support_label = []
        query_label = []

        for i, entry in enumerate(task):

            label = i
            idx = rng.choice(100, 32)
            for id in idx[0:24]:
                query_set.append(inp[entry, id])
                query_label.append(label)

            shot = 0
            for id in idx[24:32]:
                shot = shot + 1

                support_set.append(inp[entry, id])
                support_label.append(label)
                if (mode != 'train' and shot > 1):
                    break

        c = list(zip(support_set, support_label))
        random.shuffle(c)
        support_set, support_label = zip(*c)

        d = list(zip(query_set, query_label))
        random.shuffle(d)
        query_set, query_label = zip(*d)

        support_set = np.array(support_set)
        support_set = torch.from_numpy(support_set).permute(0, 3, 1, 2).float()
        query_set = np.array(query_set)
        query_set = torch.from_numpy(query_set).permute(0, 3, 1, 2).float()

        support_label = np.array(support_label)
        support_label = torch.from_numpy(support_label).long()
        query_label = np.array(query_label)
        query_label = torch.from_numpy(query_label).long()

        task_data.append((support_set, support_label, query_set, query_label))

    return task_data
def util_score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()

class MAML(nn.Module):
    """Trains and assesses a MAML."""

    def __init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
        """
        super(MAML, self).__init__()
        meta_parameters = {}

        # construct feature extractor
        in_channels = NUM_INPUT_CHANNELS
        for i in range(NUM_CONV_LAYERS):
            meta_parameters[f'conv{i}'] = nn.init.xavier_uniform_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    in_channels,
                    KERNEL_SIZE,
                    KERNEL_SIZE,
                    requires_grad=True,
                    device=DEVICE
                )
            )
            meta_parameters[f'b{i}'] = nn.init.zeros_(
                torch.empty(
                    NUM_HIDDEN_CHANNELS,
                    requires_grad=True,
                    device=DEVICE
                )
            )
            in_channels = NUM_HIDDEN_CHANNELS

        # construct linear head layer
        meta_parameters[f'w{NUM_CONV_LAYERS}'] = nn.init.xavier_uniform_(
            torch.empty(
                num_outputs,
                NUM_HIDDEN_CHANNELS,
                requires_grad=True,
                device=DEVICE
            )
        )
        meta_parameters[f'b{NUM_CONV_LAYERS}'] = nn.init.zeros_(
            torch.empty(
                num_outputs,
                requires_grad=True,
                device=DEVICE
            )
        )

        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        # os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _forward(self, images, parameters):
        """Computes predicted classification logits.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)
            parameters (dict[str, Tensor]): parameters to use for
                the computation

        Returns:
            a Tensor consisting of a batch of logits
                shape (num_images, classes)
        """
        x = images
        for i in range(NUM_CONV_LAYERS):
            x = F.conv2d(
                input=x,
                weight=parameters[f'conv{i}'],
                bias=parameters[f'b{i}'],
                stride=1,
                padding='same'
            )
            x = F.batch_norm(x, None, None, training=True)
            x = F.relu(x)
        x = torch.mean(x, dim=[2, 3])
        return F.linear(
            input=x,
            weight=parameters[f'w{NUM_CONV_LAYERS}'],
            bias=parameters[f'b{NUM_CONV_LAYERS}']
        )

    def _inner_loop(self, images, labels, train):
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Tensor): task support set outputs
                shape (num_images,)
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
        """
        accuracies = []
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        # TODO: finish implementing this method.
        # This method computes the inner loop (adaptation) procedure
        # over the course of _num_inner_steps steps for one
        # task. It also scores the model along the way.
        # Make sure to populate accuracies and update parameters.
        # Use F.cross_entropy to compute classification losses.
        # Use util.score to compute accuracies.

        if (train == True):
            graph = True
        else:
            graph = False

        for i in range(self._num_inner_steps):

            logits = self._forward(images, parameters)
            loss_train = F.cross_entropy(logits, labels)

            score = util_score(logits, labels)

            accuracies.append(score)

            gradient = torch.autograd.grad(loss_train, parameters.values(), create_graph=graph)

            for key, grad in zip(parameters, gradient):
                value = parameters[key]
                parameters[key] = value - self._inner_lrs[key] * grad
        logits = self._forward(images, parameters)
        score = util_score(logits, labels)
        accuracies.append(score)

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        return parameters, accuracies

    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)
            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
            # TODO: finish implementing this method.
            # For a given task, use the _inner_loop method to adapt for
            # _num_inner_steps steps, then compute the MAML loss and other
            # metrics.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate outer_loss_batch, accuracies_support_batch,
            # and accuracy_query_batch.
            parameters, accuracies = self._inner_loop(images_support, labels_support, train)
            logits_query = self._forward(images_query, parameters)
            loss_test = F.cross_entropy(logits_query, labels_query)
            accuracy_query = util_score(logits_query, labels_query)

            outer_loss_batch.append(loss_test)
            accuracies_support_batch.append(accuracies)
            accuracy_query_batch.append(accuracy_query)

            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(
            accuracies_support_batch,
            axis=0
        )
        accuracy_query = np.mean(accuracy_query_batch)

        return outer_loss, accuracies_support, accuracy_query

    def train(self, writer):
        """Train the MAML.

        Consumes dataloader_train to optimize MAML meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """

        task_batch = get_data()
        self._optimizer.zero_grad()
        outer_loss, accuracies_support, accuracy_query = (
            self._outer_step(task_batch, train=True)
        )
        outer_loss.backward()
        self._optimizer.step()

        i_step = self._start_train_step
        self._start_train_step += 1

        # print(self._start_train_step)

        if i_step % LOG_INTERVAL == 0:
            print(
                f'Iteration {i_step}: '
                f'loss: {outer_loss.item():.3f}, '
                f'pre-adaptation support accuracy: '
                f'{accuracies_support[0]:.3f}, '
                f'post-adaptation support accuracy: '
                f'{accuracies_support[-1]:.3f}, '
                f'post-adaptation query accuracy: '
                f'{accuracy_query:.3f}'
            )
            writer.add_scalar('loss/train', outer_loss.item(), i_step)
            writer.add_scalar(
                'train_accuracy/pre_adapt_support',
                accuracies_support[0],
                i_step
            )
            writer.add_scalar(
                'train_accuracy/post_adapt_support',
                accuracies_support[-1],
                i_step
            )
            writer.add_scalar(
                'train_accuracy/post_adapt_query',
                accuracy_query,
                i_step
            )

        if i_step % VAL_INTERVAL == 0:
            losses = []
            accuracies_pre_adapt_support = []
            accuracies_post_adapt_support = []
            accuracies_post_adapt_query = []
            val_task_batch = get_data(mode="valid")
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(val_task_batch, train=False))
            losses.append(outer_loss.item())
            accuracies_pre_adapt_support.append(accuracies_support[0])
            accuracies_post_adapt_support.append(accuracies_support[-1])
            accuracies_post_adapt_query.append(accuracy_query)
            loss = np.mean(losses)
            accuracy_pre_adapt_support = np.mean(
                accuracies_pre_adapt_support
            )
            accuracy_post_adapt_support = np.mean(
                accuracies_post_adapt_support
            )
            accuracy_post_adapt_query = np.mean(
                accuracies_post_adapt_query
            )
            print(
                f'Validation: '
                f'loss: {loss:.3f}, '
                f'pre-adaptation support accuracy: '
                f'{accuracy_pre_adapt_support:.3f}, '
                f'post-adaptation support accuracy: '
                f'{accuracy_post_adapt_support:.3f}, '
                f'post-adaptation query accuracy: '
                f'{accuracy_post_adapt_query:.3f}'
            )
            writer.add_scalar('loss/val', loss, i_step)
            writer.add_scalar(
                'val_accuracy/pre_adapt_support',
                accuracy_pre_adapt_support,
                i_step
            )
            writer.add_scalar(
                'val_accuracy/post_adapt_support',
                accuracy_post_adapt_support,
                i_step
            )
            writer.add_scalar(
                'val_accuracy/post_adapt_query',
                accuracy_post_adapt_query,
                i_step
            )

        if i_step % SAVE_INTERVAL == 0:
            self._save(i_step)

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')

    def _load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def load_specific_mapped(self, file_path, mapping):
        # Load the entire state_dict of the saved model
        saved_state_dict = torch.load(file_path)

        # Create a new_state_dict and populate it with parameters from the saved_state_dict
        new_state_dict = self.state_dict()  # starts with the current model's state_dict
        for k_new, k_saved in mapping.items():
            if k_saved in saved_state_dict:
                new_state_dict[k_new] = saved_state_dict[k_saved]

        # Load the new_state_dict into the current model
        self.load_state_dict(new_state_dict)


class ColorClassifier:
    def __init__(self, load_checkpoint=False, checkpoint_path="state200.pt"):
        num_way = 3
        num_inner_steps = 1
        inner_lr = 0.4
        outer_lr = 0.001
        learn_inner_lrs = True

        self.maml = MAML(
            num_way,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            checkpoint_path
        )
        parameters_map = {'meta_parameters': '_meta_parameters',
                          'inner_lrs': '_inner_lrs',
                          }
        #if load_checkpoint:
        #    self.load_checkpoint_file()
        print(f"loading from {checkpoint_path}")
        self.maml.load_specific_mapped(checkpoint_path, parameters_map)
        #task_batch = get_data(mode='inference')
        self.parameters = {
            k: torch.clone(v)
            for k, v in self.maml._meta_parameters.items()
        }
        self.maml.to(DEVICE)
        self.parameters.to(DEVICE)
    def load_image(self, file_path):
        # Open the image file with PIL
        image = Image.open(file_path)

        # Convert the PIL image to a PyTorch tensor
        tensor = ToTensor()(image)

        return tensor


    def load_images(self, file_paths):
        # Create a list to store the image tensors
        tensors = []

        # Loop over the file paths
        for file_path in file_paths:
            # Open the image file
            image = Image.open(file_path)

            # Convert the image to a tensor and resize it
            # tensor = transform(image)
            tensor = ToTensor()(image)
            # Add the tensor to the list
            tensors.append(tensor)

        # Stack the tensors along a new dimension
        tensors = torch.stack(tensors)

        return tensors

    def load_checkpoint_file(self):
        file_url = "https://drive.google.com/uc?export=download&id=1-EIWKrO7kiuNuf4Ku2oXqSHfR3-hbJcH"
        save_path = "checkpoint200.pt"  # Replace with the desired file name

        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Unable to download the file.")
    def save_images(self, tensor, file_paths):
        # Convert the tensor to a PIL Image and save it
        for i in range(tensor.shape[0]):
            img = ToPILImage()(tensor[i])
            img.save(file_paths[i])


    def colors_classify(self, file_path):
        # array_img = []
        # for m in images:
        image = self.load_image(file_path)
        image = image.unsqueeze(0)
        # array_img.append(image)
        # array_img = torch.stack(array_img)
        image = image.to(DEVICE)
        result = self.maml._forward(image, self.parameters)
        result = class_idx = torch.argmax(F.softmax(result))
        return f"the color class is {int(result.item())}"