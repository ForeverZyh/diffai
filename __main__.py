import future
import builtins
import past
import six
import copy

from timeit import default_timer as timer
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import Dataset
import decimal
import torch.onnx
import numpy as np

import inspect
from inspect import getargspec
import os
import helpers as h
from helpers import Timer
import copy
import random

from components import *
import models

import goals
import scheduling

from goals import *
from scheduling import *

import math

import warnings
from torch.serialization import SourceChangeWarning

POINT_DOMAINS = [m for m in h.getMethods(goals) if issubclass(m, goals.Point)]
SYMETRIC_DOMAINS = [goals.Box] + POINT_DOMAINS

datasets.Imagenet12 = None


class Top(nn.Module):
    def __init__(self, args, net, ty=Point):
        super(Top, self).__init__()
        self.net = net
        self.ty = ty
        self.w = args.width
        self.global_num = 0
        self.getSpec = getattr(self, args.spec)
        self.sub_batch_size = args.sub_batch_size
        self.curve_width = args.curve_width
        self.regularize = args.regularize

        self.speedCount = 0
        self.speed = 0.0

    def addSpeed(self, s):
        self.speed = (s + self.speed * self.speedCount) / (self.speedCount + 1)
        self.speedCount += 1

    def forward(self, x):
        return self.net(x)

    def clip_norm(self):
        self.net.clip_norm()

    def boxSpec(self, x, target, **kargs):
        return [(self.ty.box(x, w=self.w, model=self, target=target, untargeted=True, **kargs).to_dtype(), target)]

    def curveSpec(self, x, target, **kargs):
        if self.ty.__class__ in SYMETRIC_DOMAINS:
            return self.boxSpec(x, target, **kargs)

        batch_size = x.size()[0]

        newTargs = [None for i in range(batch_size)]
        newSpecs = [None for i in range(batch_size)]
        bestSpecs = [None for i in range(batch_size)]

        for i in range(batch_size):
            newTarg = target[i]
            newTargs[i] = newTarg
            newSpec = x[i]

            best_x = newSpec
            best_dist = float("inf")
            for j in range(batch_size):
                potTarg = target[j]
                potSpec = x[j]
                if (not newTarg.data.equal(potTarg.data)) or i == j:
                    continue
                curr_dist = (newSpec - potSpec).norm(1).item()  # must experiment with the type of norm here
                if curr_dist <= best_dist:
                    best_x = potSpec

            newSpecs[i] = newSpec
            bestSpecs[i] = best_x

        new_batch_size = self.sub_batch_size
        batchedTargs = h.chunks(newTargs, new_batch_size)
        batchedSpecs = h.chunks(newSpecs, new_batch_size)
        batchedBest = h.chunks(bestSpecs, new_batch_size)

        def batch(t, s, b):
            t = h.lten(t)
            s = torch.stack(s)
            b = torch.stack(b)

            if h.use_cuda:
                t.cuda()
                s.cuda()
                b.cuda()

            m = self.ty.line(s, b, w=self.curve_width, **kargs)
            return (m, t)

        return [batch(t, s, b) for t, s, b in zip(batchedTargs, batchedSpecs, batchedBest)]

    def regLoss(self):
        if self.regularize is None or self.regularize <= 0.0:
            return 0
        reg_loss = 0
        r = self.net.regularize(2)
        return self.regularize * r

    def aiLoss(self, dom, target, **args):
        r = self(dom)
        return self.regLoss() + r.loss(target=target, **args)

    def printNet(self, f):
        self.net.printNet(f)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch DiffAI Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='input batch size for training')
parser.add_argument('--test-first', type=h.str2bool, nargs='?', const=True, default=True, help='test first')
parser.add_argument('--test-freq', type=int, default=1, metavar='N', help='number of epochs to skip before testing')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N', help='input batch size for testing')
parser.add_argument('--sub-batch-size', type=int, default=3, metavar='N', help='input batch size for curve specs')

parser.add_argument('--custom-schedule', type=str, default="", metavar='net',
                    help='Learning rate scheduling for lr-multistep.  Defaults to [200,250,300] for CIFAR10 and [15,25] for everything else.')

parser.add_argument('--test', type=str, default=None, metavar='net',
                    help='Saved net to use, in addition to any other nets you specify with -n')
parser.add_argument('--update-test-net', type=h.str2bool, nargs='?', const=True, default=False,
                    help="should update test net")

parser.add_argument('--sgd', type=h.str2bool, nargs='?', const=True, default=False, help="use sgd instead of adam")
parser.add_argument('--onyx', type=h.str2bool, nargs='?', const=True, default=False, help="should output onyx")
parser.add_argument('--save-dot-net', type=h.str2bool, nargs='?', const=True, default=False,
                    help="should output in .net")
parser.add_argument('--update-test-net-name', type=str, choices=h.getMethodNames(models), default=None,
                    help="update test net name")

parser.add_argument('--normalize-layer', type=h.str2bool, nargs='?', const=True, default=True,
                    help="should include a training set specific normalization layer")
parser.add_argument('--clip-norm', type=h.str2bool, nargs='?', const=True, default=False,
                    help="should clip the normal and use normal decomposition for weights")

parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--log-freq', type=int, default=10, metavar='N',
                    help='The frequency with which log statistics are printed')
parser.add_argument('--save-freq', type=int, default=1, metavar='N',
                    help='The frequency with which nets and images are saved, in terms of number of test passes')
parser.add_argument('--number-save-images', type=int, default=0, metavar='N',
                    help='The number of images to save. Should be smaller than test-size.')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--lr-multistep', type=h.str2bool, nargs='?', const=True, default=False,
                    help='learning rate multistep scheduling')

parser.add_argument('--threshold', type=float, default=-0.01, metavar='TH', help='threshold for lr schedule')
parser.add_argument('--patience', type=int, default=0, metavar='PT', help='patience for lr schedule')
parser.add_argument('--factor', type=float, default=0.5, metavar='R', help='reduction multiplier for lr schedule')
parser.add_argument('--max-norm', type=float, default=10000, metavar='MN',
                    help='the maximum norm allowed in weight distribution')

parser.add_argument('--curve-width', type=float, default=None, metavar='CW', help='the width of the curve spec')

parser.add_argument('--width', type=float, default=0.01, metavar='CW', help='the width of either the line or box')
parser.add_argument('--spec',
                    choices=[x for x in dir(Top) if x[-4:] == "Spec" and len(getargspec(getattr(Top, x)).args) == 3]
                    , default="boxSpec", help='picks which spec builder function to use for training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument("--use-schedule", type=h.str2bool, nargs='?',
                    const=True, default=False,
                    help="activate learning rate schedule")

parser.add_argument('-d', '--domain', sub_choices=None, action=h.SubAct
                    , default=[], help='picks which abstract goals to use for training', required=True)

parser.add_argument('-t', '--test-domain', sub_choices=None, action=h.SubAct
                    , default=[], help='picks which abstract goals to use for testing.  Examples include ' + str(goals),
                    required=True)

parser.add_argument('-n', '--net', choices=h.getMethodNames(models), action='append'
                    , default=[], help='picks which net to use for training')  # one net for now

parser.add_argument('-D', '--dataset'
                    # , choices = [n for (n,k) in inspect.getmembers(datasets, inspect.isclass) if issubclass(k, Dataset)]
                    , default="MNIST", help='picks which dataset to use.')

parser.add_argument('-o', '--out', default="out", help='picks which net to use for training')
parser.add_argument('--dont-write', type=h.str2bool, nargs='?', const=True, default=False,
                    help='dont write anywhere if this flag is on')
parser.add_argument('--write-first', type=h.str2bool, nargs='?', const=True, default=False,
                    help='write the initial net.  Useful for comparing algorithms, a pain for testing.')
parser.add_argument('--test-size', type=int, default=2000, help='number of examples to test with')
parser.add_argument('--test-swap-delta', type=int, default=0, help='number of swaps in each sentence')

parser.add_argument('-r', '--regularize', type=float, default=None, help='use regularization')
parser.add_argument("--gpu_id", type=str, default=None, help="specify gpu id, None for all")
parser.add_argument("--decay-fir", type=bool, default=False, help="decay the first Mix domain")

args = parser.parse_args()
if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

largest_domain = max([len(h.catStrs(d)) for d in (args.domain)])
largest_test_domain = max([len(h.catStrs(d)) for d in (args.test_domain)])

args.log_interval = int(50000 / (args.batch_size * args.log_freq))

h.max_c_for_norm = args.max_norm

if h.use_cuda:
    torch.cuda.manual_seed(1 + args.seed)
else:
    torch.manual_seed(args.seed)

train_loader = h.loadDataset(args.dataset, args.batch_size, True, False)
val_loader = h.loadDataset(args.dataset, args.batch_size, True, False, True)
test_loader = h.loadDataset(args.dataset, args.test_batch_size, False, False)

input_dims = train_loader.dataset[0][0].size()
num_classes = (int(max(getattr(train_loader.dataset,
                               'train_labels' if args.dataset != "SVHN" else 'labels'))) + 1) if args.dataset != "AG" else 4

print("input_dims: ", input_dims)
print("Num classes: ", num_classes)

vargs = vars(args)

total_batches_seen = 0


def train(epoch, models, decay=True):
    global total_batches_seen

    for model in models:
        model.train()
        #if args.decay_fir:
        #    if epoch > 1 and isinstance(model.ty, goals.DList) and len(model.ty.al) == 2 and decay:
        #        for (i, a) in enumerate(model.ty.al):
        #            if i == 1:
        #                model.ty.al[i] = (a[0], Const(min(a[1].getVal() + 0.0025, 0.75)))
        #            else:
        #                model.ty.al[i] = (a[0], Const(max(a[1].getVal() - 0.0025, 0.25)))

    for batch_idx, (data, target) in enumerate(train_loader):
        if total_batches_seen * args.batch_size % 4000 == 0:
            for model in models:
                if args.decay_fir:
                    if isinstance(model.ty, goals.DList) and len(model.ty.al) == 2 and decay:
                        for (i, a) in enumerate(model.ty.al):
                            if i == 1:
                                model.ty.al[i] = (a[0], Const(min(a[1].getVal() + 0.0025, 3)))
                            # else:
                            #    model.ty.al[i] = (a[0], Const(max(a[1].getVal() - 0.00075, 0.25)))

        total_batches_seen += 1
        time = float(total_batches_seen) / len(train_loader)
        if h.use_cuda:
            data, target = data.cuda(), target.cuda()

        for model in models:
            model.global_num += data.size()[0]

            timer = Timer("train a sample from " + model.name + " with " + model.ty.name, data.size()[0], False)
            lossy = 0
            with timer:
                for s in model.getSpec(data.to_dtype(), target, time=time):
                    model.optimizer.zero_grad()
                    loss = model.aiLoss(*s, time=time, **vargs).mean(dim=0)
                    lossy += loss.detach().item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    for p in model.parameters():
                        if p is not None and torch.isnan(p).any():
                            print("Such nan in vals")
                        if p is not None and p.grad is not None and torch.isnan(p.grad).any():
                            print("Such nan in postmagic")
                            stdv = 1 / math.sqrt(h.product(p.data.shape))
                            p.grad = torch.where(torch.isnan(p.grad),
                                                 torch.normal(mean=h.zeros(p.grad.shape), std=stdv), p.grad)

                    model.optimizer.step()

                    for p in model.parameters():
                        if p is not None and torch.isnan(p).any():
                            print("Such nan in vals after grad")
                            stdv = 1 / math.sqrt(h.product(p.data.shape))
                            p.data = torch.where(torch.isnan(p.data),
                                                 torch.normal(mean=h.zeros(p.data.shape), std=stdv), p.data)

                    if args.clip_norm:
                        model.clip_norm()
                    for p in model.parameters():
                        if p is not None and torch.isnan(p).any():
                            raise Exception("Such nan in vals after clip")

            model.addSpeed(timer.getUnitTime())

            if batch_idx % args.log_interval == 0:
                print(('Train Epoch {:12} {:' + str(
                    largest_domain) + '}: {:3} [{:7}/{} ({:.0f}%)] \tAvg sec/ex {:1.8f}\tLoss: {:.6f}').format(
                    model.name, model.ty.name,
                    epoch,
                    batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    model.speed,
                    lossy))

    val = 0
    val_origin = 0
    batch_cnt = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        batch_cnt += 1
        if h.use_cuda:
            data, target = data.cuda(), target.cuda()

        for model in models:
            for s in model.getSpec(data.to_dtype(), target):
                loss = model.aiLoss(*s, **vargs).mean(dim=0)
                val += loss.detach().item()

            loss = model.aiLoss(data, target, **vargs).mean(dim=0)
            val_origin += loss.detach().item()

    return val_origin / batch_cnt, val / batch_cnt


num_tests = 0


def test(models, epoch, f=None):
    global num_tests
    num_tests += 1

    class MStat:
        def __init__(self, model):
            model.eval()
            self.model = model
            self.correct = 0

            class Stat:
                def __init__(self, d, dnm):
                    self.domain = d
                    self.name = dnm
                    self.width = 0
                    self.max_eps = None
                    self.safe = 0
                    self.proved = 0
                    self.time = 0

            self.domains = [Stat(h.parseValues(d, goals), h.catStrs(d)) for d in args.test_domain]

    model_stats = [MStat(m) for m in models]
    dict_map = dict(np.load("./dataset/AG/dict_map.npy").item())
    lines = open("./dataset/en.key1").readlines()
    adjacent_keys = [[] for i in range(len(dict_map))]
    for line in lines:
        tmp = line.strip().split()
        ret = set(tmp[1:]).intersection(dict_map.keys())
        ids = []
        for x in ret:
            ids.append(dict_map[x])
        adjacent_keys[dict_map[tmp[0]]].extend(ids)

    num_its = 0
    saved_data_target = []
    for data, target in test_loader:
        if num_its >= args.test_size:
            break

        if num_tests == 1:
            saved_data_target += list(zip(list(data), list(target)))

        num_its += data.size()[0]
        if num_its % 100 == 0:
            print(num_its, model_stats[0].domains[0].safe * 100.0 / num_its)
        if args.test_swap_delta > 0:
            length = data.size()[1]
            data = data.repeat(1, length)
            for i in data:
                for j in range(length - 1):
                    for _ in range(args.test_swap_delta):
                        t = np.random.randint(0, length)
                        while len(adjacent_keys[int(i[t])]) == 0:
                            t = np.random.randint(0, length)
                        cid = int(i[t])
                        i[j * length + t] = adjacent_keys[cid][0]
            target = (target.view(-1, 1).repeat(1, length)).view(-1)
            data = data.view(-1, length)

        if h.use_cuda:
            data, target = data.cuda().to_dtype(), target.cuda()

        for m in model_stats:

            with torch.no_grad():
                pred = m.model(data).vanillaTensorPart().max(1, keepdim=True)[
                    1]  # get the index of the max log-probability
                m.correct += pred.eq(target.data.view_as(pred)).sum()

            for stat in m.domains:
                timer = Timer(shouldPrint=False)
                with timer:
                    def calcData(data, target):
                        box = stat.domain.box(data, w=m.model.w, model=m.model, untargeted=True,
                                              target=target).to_dtype()
                        with torch.no_grad():
                            bs = m.model(box)
                            org = m.model(data).vanillaTensorPart().max(1, keepdim=True)[1]
                            stat.width += bs.diameter().sum().item()  # sum up batch loss
                            stat.proved += bs.isSafe(org).sum().item()
                            stat.safe += bs.isSafe(target).sum().item()
                            # stat.max_eps += 0 # TODO: calculate max_eps

                    if m.model.net.neuronCount() < 5000 or stat.domain in SYMETRIC_DOMAINS:
                        calcData(data, target)
                    else:
                        if args.test_swap_delta > 0:
                            length = data.size()[1]
                            pre_stat = copy.deepcopy(stat)
                            for i, (d, t) in enumerate(zip(data, target)):
                                calcData(d.unsqueeze(0), t.unsqueeze(0))
                                if (i + 1) % length == 0:
                                    d_proved = (stat.proved - pre_stat.proved) // length
                                    d_safe = (stat.safe - pre_stat.safe) // length
                                    d_width = (stat.width - pre_stat.width) / length
                                    stat.proved = pre_stat.proved + d_proved
                                    stat.safe = pre_stat.safe + d_safe
                                    stat.width = pre_stat.width + d_width
                                    pre_stat = copy.deepcopy(stat)
                        else:
                            for d, t in zip(data, target):
                                calcData(d.unsqueeze(0), t.unsqueeze(0))
                stat.time += timer.getUnitTime()

    l = num_its  # len(test_loader.dataset)
    for m in model_stats:
        if args.lr_multistep:
            m.model.lrschedule.step()

        pr_corr = float(m.correct) / float(l)
        if args.use_schedule:
            m.model.lrschedule.step(1 - pr_corr)

        h.printBoth(('Test: {:12} trained with {:' + str(
            largest_domain) + '} - Avg sec/ex {:1.12f}, Accuracy: {}/{} ({:3.1f}%)').format(
            m.model.name, m.model.ty.name,
            m.model.speed,
            m.correct, l, 100. * pr_corr), f=f)

        model_stat_rec = ""
        for stat in m.domains:
            pr_safe = stat.safe / l
            pr_proved = stat.proved / l
            pr_corr_given_proved = pr_safe / pr_proved if pr_proved > 0 else 0.0
            h.printBoth(("\t{:" + str(
                largest_test_domain) + "} - Width: {:<36.16f} Pr[Proved]={:<1.3f}  Pr[Corr and Proved]={:<1.3f}  Pr[Corr|Proved]={:<1.3f} {}Time = {:<7.5f}").format(
                stat.name,
                stat.width / l,
                pr_proved,
                pr_safe, pr_corr_given_proved,
                "AvgMaxEps: {:1.10f} ".format(stat.max_eps / l) if stat.max_eps is not None else "",
                stat.time), f=f)
            model_stat_rec += "{}_{:1.3f}_{:1.3f}_{:1.3f}__".format(stat.name, pr_proved, pr_safe, pr_corr_given_proved)
        prepedname = m.model.ty.name.replace(" ", "_").replace(",", "").replace("(", "_").replace(")", "_").replace("=",
                                                                                                                    "_")
        net_file = os.path.join(out_dir,
                                m.model.name + "__" + prepedname + "_checkpoint_" + str(epoch) + "_with_{:1.3f}".format(
                                    pr_corr))

        h.printBoth("\tSaving netfile: {}\n".format(net_file + ".pynet"), f=f)

        if (num_tests % args.save_freq == 1 or args.save_freq == 1) and not args.dont_write and (
                num_tests > 1 or args.write_first):
            print("Actually Saving")
            torch.save(m.model.net, net_file + ".pynet")
            if args.save_dot_net:
                with h.mopen(args.dont_write, net_file + ".net", "w") as f2:
                    m.model.net.printNet(f2)
                    f2.close()
            if args.onyx:
                nn = copy.deepcopy(m.model.net)
                nn.remove_norm()
                torch.onnx.export(nn, h.zeros([1] + list(input_dims)), net_file + ".onyx",
                                  verbose=False, input_names=["actual_input"] + ["param" + str(i) for i in
                                                                                 range(len(list(nn.parameters())))],
                                  output_names=["output"])

    if num_tests == 1 and not args.dont_write:
        img_dir = os.path.join(out_dir, "images")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for img_num, (img, target) in zip(range(args.number_save_images), saved_data_target[:args.number_save_images]):
            sz = ""
            for s in img.size():
                sz += str(s) + "x"
            sz = sz[:-1]

            img_file = os.path.join(img_dir, args.dataset + "_" + sz + "_" + str(img_num))
            if img_num == 0:
                print("Saving image to: ", img_file + ".img")
            with open(img_file + ".img", "w") as imgfile:
                flatimg = img.view(h.product(img.size()))
                for t in flatimg.cpu():
                    print(decimal.Decimal(float(t)).__format__("f"), file=imgfile)
            with open(img_file + ".class", "w") as imgfile:
                print(int(target.item()), file=imgfile)


def createModel(net, domain, domain_name):
    net_weights, net_create = net
    domain.name = domain_name

    net = net_create()
    m = {}
    for (k, v) in net_weights.state_dict().items():
        m[k] = v.to_dtype()
    net.load_state_dict(m)

    model = Top(args, net, domain)
    if args.clip_norm:
        model.clip_norm()
    if h.use_cuda:
        model.cuda()
    if args.sgd:
        model.optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        model.optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_multistep:
        model.lrschedule = optim.lr_scheduler.MultiStepLR(
            model.optimizer,
            gamma=0.1,
            milestones=eval(args.custom_schedule) if args.custom_schedule != "" else (
                [200, 250, 300] if args.dataset == "CIFAR10" else [15, 25]))
    else:
        model.lrschedule = optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer,
            'min',
            patience=args.patience,
            threshold=args.threshold,
            min_lr=0.000001,
            factor=args.factor,
            verbose=True)

    net.name = net_create.__name__
    model.name = net_create.__name__

    return model


out_dir = os.path.join(args.out, args.dataset, str(args.net)[1:-1].replace(", ", "_").replace("'", ""),
                       args.spec, "width_" + str(args.width), h.file_timestamp())

print("Saving to:", out_dir)

if not os.path.exists(out_dir) and not args.dont_write:
    os.makedirs(out_dir)

print("Starting Training with:")
with h.mopen(args.dont_write, os.path.join(out_dir, "config.txt"), "w") as f:
    for k in sorted(vars(args)):
        h.printBoth("\t" + k + ": " + str(getattr(args, k)), f=f)
print("")


def buildNet(n):
    n = n(num_classes)
    if args.normalize_layer:
        if args.dataset in ["MNIST"]:
            n = Seq(Normalize([0.1307], [0.3081]), n)
        elif args.dataset in ["CIFAR10", "CIFAR100"]:
            n = Seq(Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]), n)
        elif args.dataset in ["SVHN"]:
            n = Seq(Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]), n)
        elif args.dataset in ["Imagenet12"]:
            n = Seq(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), n)
    n = n.infer(input_dims)
    if args.clip_norm:
        n.clip_norm()
    return n


if not args.test is None:

    test_name = None


    def loadedNet():
        if test_name is not None:
            n = getattr(models, test_name)
            n = buildNet(n)
            if args.clip_norm:
                n.clip_norm()
            return n
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SourceChangeWarning)
                return torch.load(args.test)


    net = loadedNet().double() if h.dtype == torch.float64 else loadedNet().float()

    if args.update_test_net_name is not None:
        test_name = args.update_test_net_name
    elif args.update_test_net and '__name__' in dir(net):
        test_name = net.__name__

    if test_name is not None:
        loadedNet.__name__ = test_name

    nets = [(net, loadedNet)]

elif args.net == []:
    raise Exception("Need to specify at least one net with either -n or --test")
else:
    nets = []

for n in args.net:
    m = getattr(models, n)
    net_create = (lambda m: lambda: buildNet(m))(
        m)  # why doesn't python do scoping right?  This is a thunk.  It is bad.
    net_create.__name__ = n
    net = buildNet(m)
    net.__name__ = n
    nets += [(net, net_create)]

    print("Name: ", net_create.__name__)
    print("Number of Neurons (relus): ", net.neuronCount())
    print("Number of Parameters: ", sum([h.product(s.size()) for s in net.parameters()]))
    print("Depth (relu layers): ", net.depth())
    print()
    net.showNet()
    print()

if args.domain == []:
    models = [createModel(net, goals.Box(args.width), "Box") for net in nets]
else:
    models = h.flat(
        [[createModel(net, h.parseValues(d, goals, scheduling), h.catStrs(d)) for net in nets] for d in args.domain])

patience = 30
last_best_origin = 0
best_origin = 1e10
last_best = 0
best = 1e10
decay = True
with h.mopen(args.dont_write, os.path.join(out_dir, "log.txt"), "w") as f:
    startTime = timer()
    for epoch in range(1, args.epochs + 1):
        if f is not None:
            f.flush()
        if (epoch - 1) % args.test_freq == 0 and (epoch > 1 or args.test_first):
            with Timer("test all models before epoch " + str(epoch), 1):
                test(models, epoch, f)
                if f is not None:
                    f.flush()
        h.printBoth("Elapsed-Time: {:.2f}s\n".format(timer() - startTime), f=f)
        if args.epochs <= args.test_freq:
            break
        with Timer("train all models in epoch", 1, f=f):
            val_origin, val = train(epoch, models, decay)
            h.printBoth("Original val loss: %.2f\t Val loss: %.2f\n" % (val_origin, val), f=f)
            if decay:
                if val_origin < best_origin:
                    best_origin = val_origin
                    last_best_origin = epoch
                elif epoch - last_best_origin > patience:
                    h.printBoth("Early stopping decay at epoch %d\n" % epoch, f=f)
                    decay = False
            if not decay:
                if val < best:
                    best = val
                    last_best = epoch
                elif epoch - last_best > patience:
                    h.printBoth("Early stopping at epoch %d\n" % epoch, f=f)
                    break
