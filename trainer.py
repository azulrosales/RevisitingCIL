import sys
import logging
import copy
import torch
import os
import time
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    args["seed"] = copy.deepcopy(args["seed"])
    args["device"] = copy.deepcopy(args["device"])
    _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    for session in range(data_manager.incremental_sessions):
        if session > 0:
            try:
                logging.info("Loading saved model for session {}".format(session))
                model._network.load_state_dict(torch.load('checkpoint.pth'))
            except FileNotFoundError:
                logging.warning("No checkpoint found!")

        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        start_time = time.time()
        model.incremental_train(data_manager) 
        logging.info("Training Time for Session {}: {:.2f} seconds".format(session, time.time() - start_time))

        logging.info("Saving the model after session {}".format(session))
        torch.save(model._network.state_dict(), 'checkpoint.pth')

        accuracies = model.eval_task(data_manager)
        model.after_task()
        
        logging.info("Accuracy: {}".format(accuracies["per_class"]))

    
def _set_device(args):
    device_type = args["device"]
    devices = []

    if device_type == "-1":
        # If device_type is "-1", use the CPU
        devices.append(torch.device("cpu"))
    else:
        # Otherwise, treat device_type as a list of device indices
        for device in device_type:
            device = torch.device("cuda:{}".format(device))
            devices.append(device)

    args["device"] = devices


def _set_random():
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
