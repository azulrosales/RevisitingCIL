import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet, SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# tune the model (with forward BN) at first session, and then conduct simple shot.

num_workers = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # self._network = IncrementalNet(args, True)
        if 'resnet' in args['convnet_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self.batch_size = 128

        else:
            self._network = SimpleVitNet(args, True)
            self.batch_size = args["batch_size"]

        self.init_lr = args.get("init_lr", 0.01)
        self.init_weight_decay = args.get("init_weight_decay", 0.0005)
        self.init_epoch = args.get("init_epoch", 40)
        self.epochs = args.get("epochs", 80)

        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        model = model.eval()

        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                # data, label = [_.to(device) for _ in batch]
                (_, data, label) = batch
                data = data.to(device)
                label = label.to(device)
                # model.module.mode = 'encoder'
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            print('Replacing...', class_index)
            # print(class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            # new_fc.append(proto)
            self._network.fc.weight.data[class_index] = proto
        return model

    def update_fc(self, dataloader, class_list, session):
        print('APER BN: Update FC layer')
        for batch in dataloader:
            # Load data and labels to the device
            data, label = [_.to(device) for _ in batch]
            # Compute embeddings from the input data
            data = self.encode(data).detach() 
        # Update the FC layer using the class prototypes
        new_fc = self.update_fc_avg(data, label, class_list)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        # Loop through each class in the class list
        for class_index in class_list:
            # Find the indices of data points belonging to the current class
            data_index = (label == class_index).nonzero().squeeze(-1)
            # Extract embeddings for the class
            embedding = data[data_index]
            # Compute the class prototype
            proto = embedding.mean(0)
            # Append the prototype to the list of new FC weights
            new_fc.append(proto)
            # Update the FC layer weights for the current clas
            self.fc.weight.data[class_index] = proto
        # Stack all prototypes into a tensor and return
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def incremental_train(self, data_manager):
        print('APER BN: INCREMENTAL TRAIN')
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        print('APER BN: TRAIN')
        self._network.to(self._device)

        # if self._cur_task == 0:
        #     self.tsne(Normalize=True)
        # finetune the model with current dataset.
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.init_weight_decay,
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'])
            self._init_train(train_loader, test_loader, optimizer, scheduler)

            self.construct_dual_branch_network()
        else:
            pass

        self.replace_fc(train_loader_for_protonet, self._network, None)

    def construct_dual_branch_network(self):
        print('APER BN: Constructing MultiBranchCosineIncrementalNet')
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network = network.to(self._device)

    def record_running_mean(self):
        # record the index of running mean and variance
        model_dict = self._network.state_dict()
        running_dict = {}
        for e in model_dict:
            if 'running' in e:
                key_name = '.'.join(e.split('.')[1:-1])
                if key_name in running_dict:
                    continue
                else:
                    running_dict[key_name] = {}
                # find the position of BN modules
                component = self._network.convnet
                for att in key_name.split('.'):
                    if att.isdigit():
                        component = component[int(att)]
                    else:
                        component = getattr(component, att)
                running_dict[key_name]['mean'] = component.running_mean
                running_dict[key_name]['var'] = component.running_var
                running_dict[key_name]['nbt'] = component.num_batches_tracked
        # print(running_dict[key_name]['mean'], running_dict[key_name]['var'], running_dict[key_name]['nbt'])

    def clear_running_mean(self):
        print('APER BN: Cleaning Running Mean')
        # record the index of running mean and variance
        model_dict = self._network.state_dict()
        running_dict = {}
        for e in model_dict:
            if 'running' in e:
                key_name = '.'.join(e.split('.')[1:-1])
                if key_name in running_dict:
                    continue
                else:
                    # print('running name',key_name)
                    running_dict[key_name] = {}
                # find the position of BN modules
                component = self._network.convnet
                for att in key_name.split('.'):
                    # print(att)
                    if att.isdigit():
                        component = component[int(att)]
                    else:
                        component = getattr(component, att)

                running_dict[key_name]['mean'] = component.running_mean
                running_dict[key_name]['var'] = component.running_var
                running_dict[key_name]['nbt'] = component.num_batches_tracked

                component.running_mean = component.running_mean * 0
                component.running_var = component.running_var * 0
                component.num_batches_tracked = component.num_batches_tracked * 0

        # print(running_dict[key_name]['mean'],running_dict[key_name]['var'],running_dict[key_name]['nbt'])
        # print(component.running_mean, component.running_var, component.num_batches_tracked)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        print('APER BN: Initial Training')
        
        # Print the bn statistics of the current model
        # self.record_running_mean()

        if 'resnet' in self.args['convnet_type']:
            # Reset the running statistics of the BN layers
            self.clear_running_mean()

        prog_bar = tqdm(range(self.args['tuned_epoch']))
        # Adapt to the current data via forward passing
        with torch.no_grad():
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                losses = 0.0
                correct, total = 0, 0
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = self._network(inputs)["logits"]
                    del logits

            losses = 0.0
            train_acc = 0.0
            test_acc = 0.0
            # test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.init_epoch,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:], fake_targets
                )

                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)




