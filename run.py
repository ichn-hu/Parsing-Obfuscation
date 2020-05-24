#! $ROOT/Software/anaconda3/envs/p36/bin/python
import subprocess
import os
import time
import sys
from multiprocessing import Process, Queue
from os import makedirs, listdir
from os.path import join, dirname, basename, isdir, isfile, exists
import random
import argparse
import importlib
import logging
import shutil


class MachineSpecificInfo:
    PROJ_DIR = "$ROOT/Project"
    SRC_DIR = "$ROOT/Project/rsync_HO"

    DATA_DIR = "$ROOT/Project/data"
    TRAIN_PATH = "$ROOT/Project/data/ptb/train.conll"
    TEST_PATH = "$ROOT/Project/data/ptb/test.conll"
    DEV_PATH = "$ROOT/Project/data/ptb/dev.conll"
    PRETRAINED_EMBEDDING_PATH = "$ROOT/Project/data/sskip.eng.100.gz"

    LOG_DIR = "$ROOT/Project/log"
    WORK_DIR = "$ROOT/Project/work"
    PYTHON_PATH = "$ROOT/Project/rsync_HO:" +\
                  "$ROOT/Project/rsync_HO/model/parser:" +\
                  "{$PYTHONPATH}"
    CACHE_DIR = "/dev/shm/zfhu/ho_cache"
    TRASH_DIR = "/tmp/zfhu/trash"
    EXECUTOR = "$ROOT/Software/anaconda3/envs/p36/bin/python"

class MachineSpecificInfoOnLevi:
    def __init__(self, src_dir="$ROOT/Project/homo"):
        self.PROJ_DIR = "$ROOT/Project"
        self.SRC_DIR = src_dir

        self.DATA_DIR = "$ROOT/Project/data"
        self.TRAIN_PATH = "$ROOT/Project/data/ptb/train_ptb3.txt"
        self.TEST_PATH = "$ROOT/Project/data/ptb/test_ptb3.txt"
        self.DEV_PATH = "$ROOT/Project/data/ptb/dev_ptb3.txt"
        self.PRETRAINED_EMBEDDING_PATH = "$ROOT/Project/data/sskip.eng.100.gz"
        self.LOG_DIR = "$ROOT/Project/log2"
        self.WORK_DIR = "$ROOT/v1zhu2/work"
        self.PYTHON_PATH = src_dir + ":" +\
                src_dir + "/model/parser:" +\
                "{$PYTHONPATH}"
        self.CACHE_DIR = "/dev/shm/zfhu/ho_cache"
        self.TRASH_DIR = "/tmp/zfhu/trash"
        self.EXECUTOR = "$ROOT/v1zhu2/p36env/bin/python"

class MachineSpecificInfoOnLocal:
    def __init__(self, src_dir="/home/ichn/Projects/Homomorphic-Obfuscation/src"):
        self.PROJ_DIR = "/home/ichn/Projects/Homomorphic-Obfuscation"
        self.SRC_DIR = src_dir

        self.DATA_DIR = "/home/ichn/Projects/Homomorphic-Obfuscation/data"
        self.TRAIN_PATH = "/home/ichn/Projects/Homomorphic-Obfuscation/data/dev_data/train_ptb3.txt"
        self.TEST_PATH = "/home/ichn/Projects/Homomorphic-Obfuscation/data/dev_data/test_ptb3.txt"
        self.DEV_PATH = "/home/ichn/Projects/Homomorphic-Obfuscation/data/dev_data/dev_ptb3.txt"
        self.PRETRAINED_EMBEDDING_PATH = "/home/ichn/Projects/Homomorphic-Obfuscation/data/sskip.eng.100.gz"
        self.LOG_DIR = "/home/ichn/Projects/Homomorphic-Obfuscation/log"
        self.WORK_DIR = "/home/ichn/Projects/Homomorphic-Obfuscation/work"
        self.PYTHON_PATH = src_dir + ":" +\
                src_dir + "/model/parser:" +\
                "{$PYTHONPATH}"
        self.CACHE_DIR = "/dev/shm/zfhu/ho_cache"
        self.TRASH_DIR = "/tmp/zfhu/trash"
        self.EXECUTOR = "/home/ichn/anaconda3/envs/torch/bin/python"


class MachineSpecificInfoOnTest:
    def __init__(self, src_dir="/home/ichn/Projects/Homomorphic-Obfuscation/src"):
        self.PROJ_DIR = "$ROOT/Project"
        self.SRC_DIR = src_dir

        self.DATA_DIR = "$ROOT/Project/data"
        self.TRAIN_PATH = "$ROOT/test_atk_data/train_ptb3.txt"
        self.TEST_PATH = "$ROOT/Project/data/ptb/test_ptb3.txt"
        self.DEV_PATH = "$ROOT/Project/data/ptb/dev_ptb3.txt"
        self.PRETRAINED_EMBEDDING_PATH = "$ROOT/Project/data/sskip.eng.100.gz"
        self.LOG_DIR = "$ROOT/Project/log2"
        self.WORK_DIR = "$ROOT/v1zhu2/work"
        self.PYTHON_PATH = src_dir + ":" +\
                src_dir + "/model/parser:" +\
                "{$PYTHONPATH}"
        self.CACHE_DIR = "/dev/shm/zfhu/ho_cache"
        self.TRASH_DIR = "/tmp/zfhu/trash"
        self.EXECUTOR = "$ROOT/v1zhu2/p36env/bin/python"

def get_exp_identifier(args):
    return args.name + '-' + args.time

def get_log_to(args, info):
    return join(info.LOG_DIR, get_exp_identifier(args) + '.log')

def get_envs(args, info):
    save_dir = join(info.WORK_DIR, get_exp_identifier(args))
    makedirs(save_dir, exist_ok=True)
    envs = {
        "exp_time": args.time,
        "exp_name": args.name,
        "config": args.config,
        "data_dir": info.DATA_DIR,
        "train_path": info.TRAIN_PATH,
        "test_path": info.TEST_PATH,
        "dev_path": info.DEV_PATH,
        "cache_dir": info.CACHE_DIR,
        "save_dir": save_dir,
        "pretrained_embedding_path": info.PRETRAINED_EMBEDDING_PATH
    }
    additional_kvpairs = [kvpair for kvpair in args.env.split(' ') if '=' in kvpair]
    for kvpair in additional_kvpairs:
        k, v = kvpair.split('=')
        envs[k] = v
    return envs

def get_envs_str(envs):
    env = {key: val for key, val in envs.items() if val is not None}
    os.environ.update(env)
    return " ".join(['{}="{}"'.format(key, val) for key, val in env.items()])

def get_command(envs, args, info):
    envs_str = get_envs_str(envs)
    executor = info.EXECUTOR
    entry = join(info.SRC_DIR, args.main)
    log_to = get_log_to(args, info)
    cmd = " ".join([
        envs_str,
        "PYTHONPATH={}".format(info.PYTHON_PATH),
        "CUDA_VISIBLE_DEVICES={}".format(args.gpu),
        executor,
        entry,
        "| tee -a {}".format(log_to)
    ])
    return cmd

def archieving_result(envs, args, info, dst_dir):
    makedirs(dst_dir, exist_ok=True)
    save_dir = envs['save_dir']
    shutil.move(save_dir, dst_dir)
    print("Archieving %s to %s" % (save_dir, dst_dir))
    log_to = get_log_to(args, info)
    shutil.move(log_to, dst_dir)
    print("Archieving %s to %s" % (log_to, join(dst_dir, get_exp_identifier(args))))

def make_title(args, envs):
    commit_id = subprocess.check_output("cd {} && git rev-parse HEAD".format(args.src_dir), shell=True).decode('utf-8').strip()
    # import ipdb
    # ipdb.set_trace()
    command = "python " + " ".join([argv if " " not in argv else '"' + argv + '"' for argv in sys.argv])
    exp_name = args.name
    exp_time = args.time
    server = subprocess.check_output("hostname", shell=True).decode('utf-8').strip()
    save_dir = envs["save_dir"]
    result = " "
    conclusion = " "
    title = "|".join([
        commit_id,
        command,
        exp_name,
        exp_time,
        server,
        save_dir,
        result,
        conclusion
    ])
    print(title)
    return title

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-n", "--name", required=True)
    arg_parser.add_argument("-c", "--config", default="basic")
    arg_parser.add_argument("-m", "--main", default="gateway.py")
    arg_parser.add_argument("-t", "--time", default=time.strftime("%m.%d_%H:%M", time.gmtime()))
    arg_parser.add_argument("-g", "--gpu", default=0)
    arg_parser.add_argument("-e", "--env", default="")
    arg_parser.add_argument("-src", "--src_dir", default="$ROOT/Project/homo")
    args = arg_parser.parse_args()
    info = MachineSpecificInfoOnLevi(args.src_dir)
    # info = MachineSpecificInfoOnLocal(args.src_dir)
    # info = MachineSpecificInfoOnTest(args.src_dir)
    envs = get_envs(args, info)

    cmd = get_command(envs, args, info)
    exper_title = make_title(args, envs)

    print(cmd)
    try:
        start_time = time.time()
        rc = subprocess.call(cmd, shell=True)
    except:
        print("Experiment %s exited abnormally" % (get_exp_identifier(args)))
        # TODO: email the last several lines of log to me!
        rc = -1

    end_time = time.time()
    used = end_time - start_time
    print("Experiment %s ended in %.2fs" % (get_exp_identifier(args), used))
    if rc == -1:
        print("Exit abnormally, trashing result ...")
        archieving_result(envs, args, info, info.TRASH_DIR)
    else:
        import requests
        import json

        url = "http://115.159.159.232:22339/"
        token = "send2it"
        title = "[homo] An experiment has finished"
        log_to = get_log_to(args, info)
        content = exper_title + "\n\n log is saved to \n\n" + log_to + "\n\n"
        tailed_log = subprocess.check_output("tail {}".format(log_to), shell=True).decode('utf-8')
        content += tailed_log
        requests.post(url, json=json.dumps({"token": token, "title": title, "content": content}))

class ExceptionNotify(Exception):
    def __init__(self, title, content):
        requests.post(url, json=json.dumps({"token": token, "title": title, "content": content}))
