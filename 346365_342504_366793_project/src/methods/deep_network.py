import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    # Added filters as argument to make it easier to change the number of filters
    # Added padding as argument to make it easier to change the padding
    # Added kernel_size : the size of the convolution kernel : maybe change to a list to have different kernel sizes
    # Added stride : the stride of the convolution : maybe change to a list to have different strides
    # Added max_pooling_kernel : the size of the max_pooling kernel : maybe change to a list to have different kernel sizes
    # Added max_pooling_number : the number of max_pooling layers
    # Added mlp_size : the size of the linear layers

    # Only need to fix the number of filters and the layers of the MLP

    def __init__(self, input_channels, n_classes,
                 filters=(16, 32, 64), kernel_size=3, hLayersNodes=[120, 60, 20], actF='relu'):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##
        self.input_channels = input_channels
        self.n_classes = n_classes
        hLayersNodes[-1] = n_classes

        self.filters = filters
        self.kernel_size = kernel_size
        self.nbConv = len(filters)

        self.conv2d = [nn.Conv2d] * self.nbConv
        self.conv2d[0] = nn.Conv2d(input_channels, filters[0], kernel_size)
        for convLayer in range(self.nbConv-1):
            self.conv2d[convLayer+1] = nn.Conv2d(
                filters[convLayer], filters[convLayer+1], kernel_size)

        match actF:
            case 'sigmoid':
                self.actF = F.sigmoid
            case 'tanh':
                self.actF = F.tanh
            case _:
                self.actF = F.relu

        self.fc1 = nn.Linear(filters[self.nbConv-1]*4*4, hLayersNodes[0])
        self.fc2 = nn.Linear(hLayersNodes[0], hLayersNodes[1])
        self.fc3 = nn.Linear(hLayersNodes[1], hLayersNodes[2])

        self.params = nn.ParameterList(self.conv2d)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##

        for convIndex in range(self.nbConv):
            x = F.pad(x, (1, 1, 1, 1, 0, 0, 0, 0), "constant", 0)
            conv = self.conv2d[convIndex](x)
            pool = F.max_pool2d(conv, 2)
            x = F.relu(pool)

        x = x.flatten(-3)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        preds = x
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        # Creates a state-less Stochastic Gradient Descent. Which one could be the best ?
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=lr)  # WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

            # WRITE YOUR CODE HERE if you want to do add else at each epoch

    def train_one_epoch(self, dataloader):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##

        self.model.train()  # set model to training mode
        for it, batch in enumerate(dataloader):
            # Fet the inputs and labels
            inputs, labels = batch

            # Run the forward pass
            logits = self.model.forward(inputs)

            # Compute the loss
            loss = self.criterion(logits, labels)

            # Compute the gradients
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Reset the gradients
            self.optimizer.zero_grad()

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        # WRITE YOUR CODE HERE!
        ###
        ##
        self.model.eval()  # set model to evaluation mode
        pred_list = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                inputs, _ = batch
                outputs = self.model(inputs)
                pred_list.append(outputs)

        pred_labels = torch.cat(pred_list)
        pred_labels = torch.argmax(pred_labels, dim=1)

        return pred_labels

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.numpy()
