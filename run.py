#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import math
import time
import torch

from tqdm import tqdm
from parser_model import ParserModel
from utils.parser_utils import mini_batches, load_and_preprocess_data, AverageMeter


def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    best_dev_UAS = 0

    optimizer = torch.optim.Adam(parser.model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    parser.model.train()
    n_mini_batches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=n_mini_batches) as prog:
        for i, (train_x, train_y) in enumerate(mini_batches(train_data, batch_size)):
            optimizer.zero_grad()
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            logits = model(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()

            prog.update(1)
            loss_meter.update(loss.item())

    print("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set", )
    parser.model.eval()  # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS


debug = True
# debug = False
parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)
output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
output_path = output_dir + "model.weights"

print(80 * "=")
print("INITIALIZING")
print(80 * "=")

start = time.time()
model = ParserModel(embeddings)
parser.model = model
print("took {:.2f} seconds\n".format(time.time() - start))

if __name__ == "__main__":
    # assert (torch.__version__ == "1.0.0"), "Please install torch version 1.0.0"

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set")
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")
