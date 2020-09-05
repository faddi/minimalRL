import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

parser = argparse.ArgumentParser(description="PyTorch local error training")
parser.add_argument(
    "--model",
    default="vgg8b",
    help="model, mlp, vgg13, vgg16, vgg19, vgg8b, vgg11b, resnet18, resnet34, wresnet28-10 and more (default: vgg8b)",
)
parser.add_argument(
    "--dataset",
    default="CIFAR10",
    help="dataset, MNIST, KuzushijiMNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, STL10 or ImageNet (default: CIFAR10)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--num-layers",
    type=int,
    default=1,
    help="number of hidden fully-connected layers for mlp and vgg models (default: 1",
)
parser.add_argument(
    "--num-hidden",
    type=int,
    default=1024,
    help="number of hidden units for mpl model (default: 1024)",
)
parser.add_argument(
    "--dim-in-decoder",
    type=int,
    default=4096,
    help="input dimension of decoder_y used in pred and predsim loss (default: 4096)",
)
parser.add_argument(
    "--feat-mult",
    type=float,
    default=1,
    help="multiply number of CNN features with this number (default: 1)",
)
parser.add_argument(
    "--epochs", type=int, default=400, help="number of epochs to train (default: 400)"
)
parser.add_argument(
    "--classes-per-batch",
    type=int,
    default=0,
    help="aim for this number of different classes per batch during training (default: 0, random batches)",
)
parser.add_argument(
    "--classes-per-batch-until-epoch",
    type=int,
    default=0,
    help="limit number of classes per batch until this epoch (default: 0, until end of training)",
)
parser.add_argument(
    "--lr", type=float, default=5e-4, help="initial learning rate (default: 5e-4)"
)
parser.add_argument(
    "--lr-decay-milestones",
    nargs="+",
    type=int,
    default=[200, 300, 350, 375],
    help="decay learning rate at these milestone epochs (default: [200,300,350,375])",
)
parser.add_argument(
    "--lr-decay-fact",
    type=float,
    default=0.25,
    help="learning rate decay factor to use at milestone epochs (default: 0.25)",
)
parser.add_argument(
    "--optim", default="adam", help="optimizer, adam, amsgrad or sgd (default: adam)"
)
parser.add_argument(
    "--momentum", type=float, default=0.0, help="SGD momentum (default: 0.0)"
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0, help="weight decay (default: 0.0)"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.0,
    help="unsupervised fraction in similarity matching loss (default: 0.0)",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.99,
    help="fraction of similarity matching loss in predsim loss (default: 0.99)",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="dropout after each nonlinearity (default: 0.0)",
)
parser.add_argument(
    "--loss-sup",
    default="predsim",
    help="supervised local loss, sim or pred (default: predsim)",
)
parser.add_argument(
    "--loss-unsup",
    default="none",
    help="unsupervised local loss, none, sim or recon (default: none)",
)
parser.add_argument(
    "--nonlin", default="relu", help="nonlinearity, relu or leakyrelu (default: relu)"
)
parser.add_argument(
    "--no-similarity-std",
    action="store_true",
    default=False,
    help="disable use of standard deviation in similarity matrix for feature maps",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disable CUDA training"
)
parser.add_argument(
    "--backprop", action="store_true", default=False, help="disable local loss training"
)
parser.add_argument(
    "--no-batch-norm",
    action="store_true",
    default=False,
    help="disable batch norm before non-linearities",
)
parser.add_argument(
    "--no-detach",
    action="store_true",
    default=False,
    help="do not detach computational graph",
)
parser.add_argument(
    "--pre-act", action="store_true", default=False, help="use pre-activation in ResNet"
)
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument(
    "--save-dir",
    default="/hdd/results/local-error",
    type=str,
    help="the directory used to save the trained models",
)
parser.add_argument(
    "--resume", default="", type=str, help="checkpoint to resume training from"
)
parser.add_argument(
    "--progress-bar",
    action="store_true",
    default=False,
    help="show progress bar during training",
)
parser.add_argument(
    "--no-print-stats",
    action="store_true",
    default=False,
    help="do not print layerwise statistics during training with local loss",
)
parser.add_argument(
    "--bio",
    action="store_true",
    default=False,
    help="use more biologically plausible versions of pred and sim loss (default: False)",
)
parser.add_argument(
    "--target-proj-size",
    type=int,
    default=128,
    help="size of target projection back to hidden layers for biologically plausible loss (default: 128",
)
parser.add_argument(
    "--cutout", action="store_true", default=False, help="apply cutout regularization"
)
parser.add_argument(
    "--n_holes", type=int, default=1, help="number of holes to cut out from image"
)
parser.add_argument(
    "--length", type=int, default=16, help="length of the cutout holes in pixels"
)

args = parser.parse_args()


class LinearFAFunction(torch.autograd.Function):
    """Autograd function for linear feedback alignment module.
    """

    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class LinearFA(nn.Module):
    """Linear feedback alignment module.

    Args:
        input_features (int): Number of input features to linear layer.
        output_features (int): Number of output features from linear layer.
        bias (bool): True if to use trainable bias.
    """

    def __init__(self, input_features, output_features, bias=True):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight_fa = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        if args.cuda:
            self.weight.data = self.weight.data.cuda()
            self.weight_fa.data = self.weight_fa.data.cuda()
            if bias:
                self.bias.data = self.bias.data.cuda()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_fa.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.input_features)
            + ", out_features="
            + str(self.output_features)
            + ", bias="
            + str(self.bias is not None)
            + ")"
        )


def similarity_matrix(x):
    """ Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). """
    if x.dim() == 4:
        if not args.no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R


class LocalLossBlockLinear(nn.Module):
    """A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
       
    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        batchnorm (bool): True if to use batchnorm, if None, read from args.no_batch_norm.
    """

    def __init__(
        self,
        num_in,
        num_out,
        num_classes,
        first_layer=False,
        dropout=None,
        batchnorm=None,
    ):
        super(LocalLossBlockLinear, self).__init__()

        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        self.batchnorm = not args.no_batch_norm if batchnorm is None else batchnorm
        self.encoder = nn.Linear(num_in, num_out, bias=True)

        if not args.backprop and args.loss_unsup == "recon":
            self.decoder_x = nn.Linear(num_out, num_in, bias=True)
        if not args.backprop and (
            args.loss_sup == "pred" or args.loss_sup == "predsim"
        ):
            if args.bio:
                self.decoder_y = LinearFA(num_out, args.target_proj_size)
            else:
                self.decoder_y = nn.Linear(num_out, num_classes)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        if (
            not args.backprop
            and not args.bio
            and (
                args.loss_unsup == "sim"
                or args.loss_sup == "sim"
                or args.loss_sup == "predsim"
            )
        ):
            self.linear_loss = nn.Linear(num_out, num_out, bias=False)
        if self.batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        if args.nonlin == "relu":
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == "leakyrelu":
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
        if args.optim == "sgd":
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=0,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )
        elif args.optim == "adam" or args.optim == "amsgrad":
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=0,
                weight_decay=args.weight_decay,
                amsgrad=args.optim == "amsgrad",
            )

        self.clear_stats()

    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = "{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n".format(
                self.encoder,
                self.loss_sim / self.examples,
                self.loss_pred / self.examples,
                100.0 * float(self.examples - self.correct) / self.examples,
                self.examples,
            )
            return stats
        else:
            return ""

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def optim_step(self):
        self.optimizer.step()

    def forward(self, x, y, y_onehot):
        # The linear transformation
        h = self.encoder(x)

        # Add batchnorm and nonlinearity
        if self.batchnorm:
            h = self.bn(h)
        h = self.nonlin(h)

        # Save return value and add dropout
        h_return = h
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)

        # Calculate local loss and update weights
        if (self.training or not args.no_print_stats) and not args.backprop:
            # Calculate hidden layer similarity matrix
            if (
                args.loss_unsup == "sim"
                or args.loss_sup == "sim"
                or args.loss_sup == "predsim"
            ):
                if args.bio:
                    h_loss = h
                else:
                    h_loss = self.linear_loss(h)
                Rh = similarity_matrix(h_loss)

            # Calculate unsupervised loss
            if args.loss_unsup == "sim":
                Rx = similarity_matrix(x).detach()
                loss_unsup = F.mse_loss(Rh, Rx)
            elif args.loss_unsup == "recon" and not self.first_layer:
                x_hat = self.nonlin(self.decoder_x(h))
                loss_unsup = F.mse_loss(x_hat, x.detach())
            else:
                if args.cuda:
                    loss_unsup = torch.cuda.FloatTensor([0])
                else:
                    loss_unsup = torch.FloatTensor([0])

            # Calculate supervised loss
            if args.loss_sup == "sim":
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                loss_sup = F.mse_loss(Rh, Ry)
                if not args.no_print_stats:
                    self.loss_sim += loss_sup.item() * h.size(0)
                    self.examples += h.size(0)
            elif args.loss_sup == "pred":
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    float_type = (
                        torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    )
                    y_onehot_pred = (
                        self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    )
                    loss_sup = F.binary_cross_entropy_with_logits(
                        y_hat_local, y_onehot_pred
                    )
                else:
                    loss_sup = F.cross_entropy(y_hat_local, y.detach())
                if not args.no_print_stats:
                    self.loss_pred += loss_sup.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            elif args.loss_sup == "predsim":
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                    float_type = (
                        torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    )
                    y_onehot_pred = (
                        self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    )
                    loss_pred = (1 - args.beta) * F.binary_cross_entropy_with_logits(
                        y_hat_local, y_onehot_pred
                    )
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                    loss_pred = (1 - args.beta) * F.cross_entropy(
                        y_hat_local, y.detach()
                    )
                loss_sim = args.beta * F.mse_loss(Rh, Ry)
                loss_sup = loss_pred + loss_sim
                if not args.no_print_stats:
                    self.loss_pred += loss_pred.item() * h.size(0)
                    self.loss_sim += loss_sim.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)

            # Combine unsupervised and supervised loss
            loss = args.alpha * loss_unsup + (1 - args.alpha) * loss_sup

            # Single-step back-propagation
            if self.training:
                loss.backward(retain_graph=args.no_detach)

            # Update weights in this layer and detatch computational graph
            if self.training and not args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()

            loss = loss.item()
        else:
            loss = 0.0

        return h_return, loss


class Net(nn.Module):
    """
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    """

    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes):
        super(Net, self).__init__()

        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList(
            [
                LocalLossBlockLinear(
                    input_dim * input_dim * input_ch,
                    num_hidden,
                    num_classes,
                    first_layer=True,
                )
            ]
        )
        self.layers.extend(
            [
                LocalLossBlockLinear(
                    int(num_hidden // (reduce_factor ** (i - 1))),
                    int(num_hidden // (reduce_factor ** i)),
                    num_classes,
                )
                for i in range(1, num_layers)
            ]
        )
        self.layer_out = nn.Linear(
            int(num_hidden // (reduce_factor ** (num_layers - 1))), num_classes
        )
        if not args.backprop:
            self.layer_out.weight.data.zero_()

    def parameters(self):
        if not args.backprop:
            return self.layer_out.parameters()
        else:
            return super(Net, self).parameters()

    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()

    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()

    def forward(self, x, y, y_onehot):
        x = x.view(x.size(0), -1)
        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, loss = layer(x, y, y_onehot)
            total_loss += loss
        x = self.layer_out(x)

        return x, total_loss


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(
            0.01, 0.08 - 0.01 * (n_epi / 200)
        )  # Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            score = 0.0
    env.close()


if __name__ == "__main__":
    main()
