import numpy as np
import gzip
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join, isdir
from os import mkdir
import inspect
import sys
import time
import ml_metrics as metrics
import warnings
from multiprocessing import Process, Queue
import cv2
from torch.jit import trace
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import torchvision
from torchvision import utils

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_path = "../Data"
exp_path = "../Experiments"
run_id = None
log_file = None


label_map = np.load(join(data_path, "label_map.npy"))

run_desc = "Generating images"

class DrawingLoader:
    def __init__(self, feats, labels, ids, batch_size, shuffle=True, image_size=224, line_width=4):
        self.feats = feats
        self.labels = labels
        self.ids = ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = (len(self.feats) - 1) // self.batch_size + 1
        self.image_size = image_size
        self.line_width = line_width
    
    def __len__(self):
        return  self.n_batches
    
    def to_image(self, x, im_size=224, lw=4):
        x_copy = x.copy()
        x[:, :2] *= 27
        x[:, :2] += 2
        x_copy[:, :2] *= im_size - 1
        x = x_copy.astype(np.uint8)
        ones = np.nonzero(x[:,2])[0]
        img = np.zeros((im_size, im_size), dtype=np.uint8)
        ones = np.concatenate([ones, [len(x)]]).astype(np.uint8)
        polys = [x[ones[i] : ones[i + 1], 0:2].astype(np.int32) for i in range(len(ones) - 1)]
        cv2.polylines(img, polys, False, 255, lw)
        return img
    
    def process_instance(self, x):
        x = x.astype(np.float64)
        x_min, y_min, _ = np.min(x, axis=0)
        x_max, y_max, _ = np.max(x, axis=0)
        dx = x_max - x_min
        dy = y_max - y_min
        x[:, 0] -= x_min
        x[:, 1] -= y_min
        if dx > dy:
            x[:, :2] /= dx
            x[:, 1] += 0.5 - dy / (2 * dx)
        else:
            x[:, :2] /= dy
            x[:, 0] += 0.5 - dx / (2 * dy)
        return self.to_image(x, self.image_size, self.line_width).astype(np.float32) / 255.0, x.astype(np.float32)
    
    def process_batch(self, x):
        imgs = []
        vals = []
        for instance in x:
            img, val = self.process_instance(instance)
            imgs.append(img)
            vals.append(val)
        vals = np.array(vals, dtype="O")
        imgs = np.stack(imgs, axis=0)
        return imgs, vals
    
    def background_process(self, q, indices):
        for i in range(self.n_batches):
            batch = self.feats[indices[i * self.batch_size : (i + 1) * self.batch_size]]
            q.put(self.process_batch(batch), block=True, timeout=None)
        
    def __iter__(self):
        q = Queue(10)
        indices = np.arange(len(self.feats))
        if self.shuffle:
            np.random.shuffle(indices)
        p = Process(target=self.background_process, args=(q, indices))
        p.start()
        for i in range(self.n_batches):
            imgs, vals = q.get(block=True, timeout=None)
            if self.labels is not None:
                targets = self.labels[indices[i * self.batch_size : (i + 1) * self.batch_size]]
            else:
                targets = None
            ids = self.ids[indices[i * self.batch_size : (i + 1) * self.batch_size]]
            yield imgs, vals, targets, ids
        q.close()
        p.join()

class Preprocessor(nn.Module):

    def __init__(self):
        super(Preprocessor, self).__init__()
        self.elu = nn.ELU()
        self.conv1 = nn.Conv1d(3, 48, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(48)
        self.conv2 = nn.Conv1d(48, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 96, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(96)
    
    def forward(self, x):
        imgs, padded_sequence, lengths = x
        padded_sequence = padded_sequence.permute(1, 2, 0)
        op = self.elu(self.bn1(self.conv1(padded_sequence)))
        op = self.elu(self.bn2(self.conv2(op)))
        op = self.elu(self.bn3(self.conv3(op)))
        return imgs, op.permute(2, 0, 1), lengths

class FeatExtractor(nn.Module):
    
    def __init__(self, feat_size=96):
        super(FeatExtractor, self).__init__()
        self.feat_size = feat_size
    
    def forward(self, x):
        imgs, points, lengths = x
        batch_size = imgs.shape[0]
        return torch.zeros((1, batch_size, self.feat_size)).to(device)

class BiLSTM(nn.Module):
    def __init__(self, preprocessor, feat_extractor, input_size=96, feat_size=512, in_feat_size=96, hidden_size=512, num_layers=3):
        super(BiLSTM, self).__init__()
        self.preprocessor = preprocessor
        self.feat_extractor = feat_extractor
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.feat_size = feat_size
        self.in_feat_size = in_feat_size
        self.rnn = nn.LSTM(input_size=input_size + in_feat_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True)
        self.score = nn.Linear(2 * self.hidden_size, 340)
        self.h0 = nn.Parameter(torch.randn((2 * self.num_layers, 1, self.hidden_size)))
        self.c0 = nn.Parameter(torch.randn((2 * self.num_layers, 1, self.hidden_size)))
    
    def forward(self, x):
        imgs, padded_sequence, lengths = self.preprocessor(x)
        img_feats = self.feat_extractor(x)
        inp = torch.cat([padded_sequence, img_feats.expand(padded_sequence.shape[0], -1, -1)], dim=2)
        h0 = self.h0.expand(-1, padded_sequence.shape[1], -1).contiguous()
        c0 = self.c0.expand(-1, padded_sequence.shape[1], -1).contiguous()
        packed = rnn.pack_padded_sequence(inp, lengths)
        output, _ = self.rnn(packed, (h0, c0))
        padded, lengths = rnn.pad_packed_sequence(output)
        out_means = padded.sum(dim=0) / lengths.float().to(device).unsqueeze(1)
        return self.score(out_means)
    
    def predict(self, x, top_k=3):
        decoded = []
        with torch.no_grad():
            logits = self.forward(x).cpu().numpy()
            labels = np.argsort(logits, axis=1)[:, ::-1][:, :top_k]
        return labels


class Resnet18(nn.Module):

    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)

        def squeeze_weights(m):
            m.weight.data = m.weight.data.sum(dim=1)[:,None]
            m.in_channels = 1
        
        self.resnet.conv1.apply(squeeze_weights)
        self.resnet.fc = nn.Linear(in_features=512, out_features=340, bias=True)

    def forward(self, x):
        imgs, _, _ = x
        return self.resnet(imgs)
    
    def predict(self, x, top_k=3):
        decoded = []
        with torch.no_grad():
            logits = self.forward(x).cpu().numpy()
            labels = np.argsort(logits, axis=1)[:, ::-1][:, :top_k]
        return labels 

class Resnet18Feats(nn.Module):

    def __init__(self, resnet18, trainable=False):
        super(Resnet18Feats, self).__init__()
        self.feat_extractor = nn.Sequential(*(list(resnet18.resnet.children())[:-1]))
        self.trainable = trainable
    
    def forward(self, x):
        imgs, _, _ = x
        if self.trainable:
            return self.feat_extractor(imgs)
        else:
            with torch.no_grad():
                return self.feat_extractor(imgs)
    
    def get_params(self):
        if self.trainable:
            return list(self.parameters())
        else:
            return []

class DWConv(nn.Module):
    
    def __init__(self, inp_channels, op_channels, stride):
        super(DWConv, self).__init__()
        self.inp_channels = inp_channels
        self.op_channels = op_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(inp_channels, inp_channels, 3, stride, 1, groups=inp_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(inp_channels)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(inp_channels, op_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(op_channels)
    
    def forward(self, x):
        x = self.elu(self.bn1(self.conv1(x)))
        return self.elu(self.bn2(self.conv2(x)))

class ConvNet2D(nn.Module):

    def __init__(self):
        super(ConvNet2D, self).__init__()
        self.conv1 = DWConv(1, 32, 2)
        self.conv2 = DWConv(32, 64, 2)
        self.conv3 = DWConv(64, 128, 2)
        self.conv4 = DWConv(128, 128, 1)
        self.conv5 = DWConv(128, 256, 2)
        self.conv6 = DWConv(256, 256, 1)
        self.conv7 = DWConv(256, 512, 2)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 340)
        self.elu = nn.ELU()
    
    def forward(self, x):
        imgs, _, _ = x
        x = self.conv3(self.conv2(self.conv1(imgs)))
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x = x.view(imgs.shape[0], 512, -1).mean(dim=2)
        return self.l2(self.elu(self.l1(x)))
    
    def predict(self, x, top_k=3):
        decoded = []
        with torch.no_grad():
            logits = self.forward(x).cpu().numpy()
            labels = np.argsort(logits, axis=1)[:, ::-1][:, :top_k]
        return labels


class ConvNet1D(nn.Module):

    def __init__(self):
        super(ConvNet1D, self).__init__()
        self.elu = nn.ELU()
        self.conv1 = nn.Conv1d(3, 48, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(48)
        self.conv2 = nn.Conv1d(48, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 5, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, 5, padding=3)
        self.bn5 = nn.BatchNorm1d(256)
        self.l1 = nn.Linear(256, 512)
        self.l2 = nn.Linear(512, 340)
    
    def forward(self, x):
        imgs, padded_sequence, lengths = x
        padded_sequence = padded_sequence.permute(1, 2, 0)
        op = self.elu(self.bn1(self.conv1(padded_sequence)))
        op = self.elu(self.bn2(self.conv2(op)))
        op = self.elu(self.bn3(self.conv3(op)))
        op2 = self.elu(self.bn4(self.conv4(op)))
        op2 = op2 + op
        op2 = self.elu(self.bn5(self.conv5(op2)))
        op2 = op2.sum(dim=2) / lengths.float().to(device).unsqueeze(1)
        op2 = self.elu(self.l1(op2))
        return self.elu(self.l2(op2))
    
    def predict(self, x, top_k=3):
        decoded = []
        with torch.no_grad():
            logits = self.forward(x).cpu().numpy()
            labels = np.argsort(logits, axis=1)[:, ::-1][:, :top_k]
        return labels

class Ensemble(nn.Module):

    def __init__(self, fixed_modules, trainable_modules):
        super(Ensemble, self).__init__()
        self.fixed = nn.ModuleList(fixed_modules)
        self.trainable = nn.ModuleList(trainable_modules)
        self.weights = nn.Parameter(torch.ones((len(fixed_modules) + len(trainable_modules),)))
        for module in self.fixed:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        logits = 0
        index = 0
        for module in self.fixed:
            logits = logits + self.weights[index] * module(x)
            index += 1
        for module in self.trainable:
            logits = logits + self.weights[index] * module(x)
            index += 1
        return logits
    
    def predict(self, x, top_k=3):
        decoded = []
        with torch.no_grad():
            logits = self.forward(x).cpu().numpy()
            labels = np.argsort(logits, axis=1)[:, ::-1][:, :top_k]
        return labels

    def get_params(self):
        params = []
        for module in self.trainable:
            params = params + list(module.parameters())
        params = params + [self.weights]
        return params

def trace_model(model, sample_input):
    """Traces the model
    
    Args:
        model (nn.Module): The model to be traced
        sample_input (torch.Tensor): The sample input to the model
    
    Returns:
        The traced model
    """

    if torch.__version__[0] != '1':
        return model
    else:
        return trace(model, sample_input)

def make_train_chunk(ids):
    """Loads the training data
    
    Args:
        ids (list): Indices of the chunks to be loaded
    
    Returns:
        tuple: Tuple of loaded features, labels and IDs
    """

    log("Loading data....")
    data = []
    labels = []
    drawing_ids = []
    for id in tqdm(ids):
        data.append(np.load(join(data_path, "train_feats.{}.npy".format(id))))
        labels.append(np.load(join(data_path, "train_labels.{}.npy".format(id))))
        drawing_ids.append(np.load(join(data_path, "train_ids.{}.npy".format(id))))
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels)
    drawing_ids = np.concatenate(drawing_ids)
    return data, labels, drawing_ids

def load_data(data_chunks, val_size, train_batch_size, eval_batch_size):
    """Loads the data into dataloaders
    
    Args:
        data_chunks (list): IDs for training data chunks
        val_size (int): Size of the validation set
        train_batch_size (int): Size of the training batch
        eval_batch_size (int): Size of the val and test batch
    
    Returns:
        tuple: The loaded datasets
    """
    if data_chunks:
        feats, labels, ids = make_train_chunk(data_chunks)
        train_feats = feats[: -val_size]
        train_labels = labels[: -val_size]
        train_ids = ids[: -val_size]
        val_feats = feats[-val_size :]
        val_labels = labels[-val_size :]
        val_ids = ids[-val_size :]
        traindl = DrawingLoader(train_feats, train_labels, train_ids, train_batch_size, True)
        valdl = DrawingLoader(val_feats, val_labels, val_ids, eval_batch_size, False)
    else:
        traindl = None
        valdl = None
    test_feats = np.load(join(data_path, "test_feats.npy"))
    test_ids = np.load(join(data_path, "test_ids.npy"))
    testdl = DrawingLoader(test_feats, None, test_ids, eval_batch_size, False)
    return traindl, valdl, testdl

def log(x):
    """Logs a string to the log file and stdout
    """
    tqdm.write(x)
    log_file.write(str(x) + "\n")
    log_file.flush()

def init_run():
    """Initialies a run
    """

    global log_file, run_id
    if not isdir(exp_path):
        mkdir(exp_path)
    run_id = str(int(time.time()))
    mkdir(join(exp_path, run_id))
    mkdir(join(exp_path, run_id, "models"))
    mkdir(join(exp_path, run_id, "submissions"))
    log_file = open(join(exp_path, run_id, "log.txt"), "w")
    descriptions_file = open(join(exp_path, "descriptions.txt"), "a")
    descriptions_file.write("{}: {}\n".format(run_id, run_desc))
    descriptions_file.close()
    log("RUN ID: {}".format(run_id))
    log("RUN DESCRIPTION: {}".format(run_desc))
    source_file = open(join(exp_path, run_id, "source.py"), "w")
    source_file.write(inspect.getsource(sys.modules[__name__]))
    source_file.close()
    log("--------------------------------------------------------")

def make_batch(batch):
    imgs, vals, targets, ids = batch
    lengths = np.array([len(x) for x in vals])
    order = np.argsort(lengths)[::-1]
    imgs = imgs[order]
    vals = vals[order]
    ids = ids[order]
    lengths = lengths[order]
    inv_order = np.argsort(order)
    max_len = max(lengths)
    feat_size = vals[0].shape[1]
    batch_size = len(lengths)
    padded = np.zeros((max_len, batch_size, feat_size), dtype=np.float32)
    for i in range(len(vals)):
        padded[:lengths[i], i, :] = vals[i]
    lengths = torch.LongTensor(lengths)
    imgs = torch.FloatTensor(imgs).unsqueeze(1).to(device)
    vals = torch.FloatTensor(padded).to(device)
    if targets is not None:
        targets = torch.LongTensor(targets[order]).to(device)
    return (imgs, vals, lengths), targets, ids, inv_order

    

def generate_submission(tags, model, dataset, make_batch_fn=make_batch):
    """Writes the prediction for the model
    
    Args:
        tags (tuple): Tags for the prediction file
        model (nn.Module): The model to be used
        dataset (DrawingDataloader): The test dataset
        make_batch_fn (function, optional): Defaults to make_batch. The function for making batches
    """

    model.eval()
    predictions = []
    keys = []
    log("Generating submissions....")
    name = ".".join([str(i) for i in tags]) + ".csv"
    path = join(exp_path, run_id, "submissions", name)
    for batch in tqdm(dataset, desc="Testing", file=sys.stdout):
        inputs, _, ids, inv_order = make_batch_fn(batch)
        prediction = model.predict(inputs)
        keys.append(ids[inv_order])
        predictions.append(prediction[inv_order])
    predictions = np.concatenate(predictions, axis=0)
    keys = np.concatenate(keys)
    f = open(path, "w")
    f.write("key_id,word\n")
    for key, pred in zip(keys, predictions):
        f.write("{},{} {} {}\n".format(key, label_map[pred[0]], label_map[pred[1]], label_map[pred[2]]))
    f.close()
    log("Written to file {}".format(path))
    log("--------------------------------------------------------")

def save_state(tags, model=None, optimizer=None):
    """Saves the model and the optimizer state

    Args:
        tags (tuple): tags for the saved model
        model (nn.Module, optional): Defaults to None. The model to be saved
        optimizer (optim.Optimizer, optional): Defaults to None. The optimizer to be saved
    """

    log("Saving model...")
    name = ".".join([str(i) for i in tags]) + ".pt"
    state = {}
    if model:
        state["model_state"] = model.state_dict()
    if optimizer:
        state["optim_state"] = optimizer.state_dict()
    path = join(exp_path, run_id, "models", name)
    torch.save(state, path)
    log("Saved to {}".format(path))
    log("--------------------------------------------------------")


def load_state(run_id, tags, model=None, optimizer=None, strict=False):
    """Loads the model from a particular run ID

    Args:
        run_id (int): The ID of the run to load the model from
        tags (tuple): The tags identifying the model
        model (nn.Module, optional): Defaults to None. The target model for loading the state
        optimizer (optim.Optimizer, optional): Defaults to None. The target optimizer for loading the state
        strict (bool, optional): Defaults to False. Loading type
    """

    name = ".".join([str(i) for i in tags]) + ".pt"
    log("Loading model {} from run_id {}...".format(name, run_id))
    run_id = str(run_id)
    path = join(exp_path, run_id, "models", name)
    state = torch.load(path)
    if model is not None:
        model.load_state_dict(state["model_state"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(state["optim_state"])
    log("Loaded.")
    log("--------------------------------------------------------")

def validate(model, dataset, make_batch_fn=make_batch):
    """Validates the model
    
    Args:
        model (nn.Module): The model to be validated
        dataset (DrawingLoader): The validation dataset
        make_batch_fn (function, optional): Defaults to make_batch. The function for making batches.
    
    Returns:
        float: The MAP@3 validation score
    """

    model.eval()
    actual = []
    predicted = []
    log("Calculating MAP@3 for the validation dataset...")
    for batch in tqdm(dataset, desc="Validating", file=sys.stdout):
        inputs, targets, _, _ = make_batch_fn(batch)
        predicted.append(model.predict(inputs))
        actual.append(targets.data.cpu().numpy())
    actual = np.concatenate(actual)    
    actual = [[x] for x in actual]
    predicted = np.concatenate(predicted, axis=0)
    map_3 = metrics.mapk(actual, predicted, 3)
    log("Validation MAP@3: {}".format(map_3))
    log("--------------------------------------------------------")
    return map_3

def train_batch(model, optimizer, batch, criterion, make_batch_fn=make_batch):
    """Traines the model on a batch
    
    Args:
        model (nn.Module): The model to be trained
        optimizer (optim.Optimizer): The optimizer to use
        batch (tuple): batch yielded by the dataset
        criterion (nn.Module): The crietrion to use
        make_batch_fn (function, optional): Defaults to make_batch. The function for making batches.
    
    Returns:
        float: The loss for the batch
    """

    model.train()
    inputs, targets, _, _ = make_batch_fn(batch)
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def get_correct_incorrect_batch(model, batch, make_batch_fn=make_batch):
    model.eval()
    correct = []
    incorrect = []
    inputs, targets, ids, _ = make_batch_fn(batch)
    imgs, _, _ = inputs
    prediction = model.predict(inputs)
    for label, pred, img, id in zip(targets, prediction, imgs, ids):
        if label.item() not in pred:
            incorrect.append((id, img.squeeze().data.cpu().numpy(), pred, label.item()))
        else:
            correct.append((id, img.squeeze().data.cpu().numpy(), pred, label.item()))
    return correct, incorrect

def get_n_correct_incorrect(model, dataset, n=10, make_batch_fn=make_batch):
    correct = []
    incorrect = []
    for batch in dataset:
        c, i = get_correct_incorrect_batch(model, batch, make_batch_fn)
        correct = correct + c
        incorrect = incorrect + i
        if len(correct) >= n and len(incorrect) >= n:
            break
    return correct[:n], incorrect[:n]

def generate_correct_incorrect(model, dataset, n=10, make_batch_fn=make_batch):
    correct, incorrect = get_n_correct_incorrect(model, dataset, n, make_batch_fn)
    for c, i in zip(correct, incorrect):
        plt.clf()
        plt.imshow(c[1])
        name = "C_" + c[0] + "_" + "_".join([label_map[j] for j in c[2]]) + "_" + label_map[c[3]] + ".png"
        plt.savefig(join(exp_path, run_id, name))
        plt.clf()
        plt.imshow(i[1])
        name = "I_" + i[0] + "_" + "_".join([label_map[j] for j in i[2]]) + "_" + label_map[i[3]] + ".png"
        plt.savefig(join(exp_path, run_id, name))


def train(model, train_dataset, val_dataset, test_dataset, criterion, optimizer, iters, make_batch_fn=make_batch, save_every=-1000, validate_every=-1000, test_every=1,
          name="checkpoint"):
    """Trains the model
          
    Args:
        model (nn.Module): The model to be trained
        train_dataset (DrawingLoader): The dataset for training
        val_dataset (DrawingLoader): The dataset for validation
        test (DrawingLoader): The dataset for testing
        criterion (nn.Module): The loss function
        optimizer (optim.Optimizer): The optimizer to be used
        iters (int): The number of iterations
        make_batch_fn (function, optional): Defaults to make_batch. The function for making batches.
        save_every (int, optional): Defaults to 20. Saves the model every save_every iters or every save_every batches if negative
        validate_every (int, optional): Defaults to 1. Validates the model every validate_every iters or every validate_every batches if negative
        test_every (int, optional): Defaults to 1. Validates the model every test_every iters or every test_every batches if negative
        name (str, optional): Defaults to "checkpoint". Name for the saved checkpoints
    """

    for i in range(1, iters + 1):
        losses = []
        log("Iteration {}/{}:".format(i, iters))
        bar = tqdm(train_dataset, desc="Current training loss: NaN", file=sys.stdout)
        for j, batch in enumerate(bar, 1):
            loss = train_batch(model, optimizer, batch, criterion, make_batch_fn)
            losses.append(loss)
            bar.set_description("Current training loss: {}".format(loss))
            if save_every < 0 and not j % -save_every:
                save_state((name, i, j), model, optimizer)
            if validate_every < 0 and not j % -validate_every:
                validate(model, val_dataset, make_batch_fn)
                log("Mean training loss: {}".format(np.mean(losses)))
                losses = []
            if test_every < 0 and not j % -test_every:
                generate_submission(("submission", i, j), model, train_dataset, make_batch_fn)
        log("Mean loss for the iteration: {}".format(np.mean(losses)))
        log("--------------------------------------------------------")
        if save_every > 0 and not i % save_every:
            save_state((name, i), model, optimizer)
        if validate_every > 0 and not i % validate_every:
            validate(model, val_dataset, make_batch_fn)
        if test_every > 0 and not i % test_every:
            generate_submission(("submission", i), model, test_dataset, make_batch_fn)
        

def main():
    init_run()    
    train_data, val_data, test_data = load_data([2], 10000, 128, 256)
    preprocessor = Preprocessor()
    res = Resnet18().to(device)
    f = FeatExtractor()
    bilstm = BiLSTM(preprocessor, f)
    bilstm = bilstm.to(device)
    ensemble = Ensemble([], [bilstm, res]).to(device)
    load_state(1543722611, ("checkpoint", 1, 38000), ensemble)
    validate(ensemble, val_data)
    generate_correct_incorrect(ensemble, val_data)
    
if __name__ == "__main__":
    main()