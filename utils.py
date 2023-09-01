import pandas as pd
import sys
import time
import os
import numpy as np


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def getTools(args):
    labels = []
    data_info = open(os.path.join("data_prepare", args.dataset, "train.txt"))
    for line in data_info:
        line_split = line.split(" ")
        labels.append(np.array(line_split[1:]).astype(float).tolist())

    db_str = []
    # convert the labels list to labels string
    for l in labels:
        db_str.append("")
        for b in l:
            db_str[-1] += str(int(b))
    # the number of every label, {label: number}
    str2count = {}
    # {label string: label list}
    str2lab = {}
    # a list of label string
    strlist = []
    str2index = {}
    str2anchor = {}

    db_index = 0
    for s in db_str:
        if str2count.get(s) == None:
            str2count[s] = 1
        else:
            str2count[s] += 1
        if str2lab.get(s) == None:
            str2lab[s] = [int(i) for i in s]
            strlist.append(s)
        if str2index.get(s) == None:
            str2index[s] = []
            str2index[s].append(db_index)
        else:
            str2index[s].append(db_index)
        db_index += 1

    file = np.loadtxt(os.path.join(args.path, "train_hash.txt"))
    if args.dataset == 'imagenet':
        _, train_hash = file[:, :100], file[:, 100:]
    elif args.dataset == 'coco':
        _, train_hash = file[:, :80], file[:, 80:]
    elif args.dataset == 'places365':
        _, train_hash = file[:, :36], file[:, 36:]
    for k in strlist:
        choose_indexes = str2index[k]
        anchor = get_anchor_code(np.array(train_hash)[np.array(choose_indexes)]).tolist()
        str2anchor[k] = anchor
    return str2count, str2lab, strlist, str2index, str2anchor


# calculate anchor code using Component-voting Scheme
def get_anchor_code(codes):
    return np.sign(np.sum(codes, axis=0))
