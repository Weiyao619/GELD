import argparse
import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')

parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.)
parser.add_argument('--percentage', type = float, help = 'percentage of sault and pepper noise', default = 0.)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,clean,saltpepper]', default='clean')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--cifar_10or100', default='10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet34)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--K', default=5, type=int, help='iteration times')
parser.add_argument('--N', default=6, type=int, help='Number of Subsets')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
best_acc1 = 0