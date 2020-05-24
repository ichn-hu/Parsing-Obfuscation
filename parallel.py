import time
import os
from multiprocessing import Process, Queue
import random
import subprocess

vacant_gpu = {}
all_gpu = []
occupied_gpu = {}
inprocess_gpu = {}

def check_vacancy(gpu_info):
    cmd = 'ssh {} "/disk/ostrom/v1zhu2/p36env/bin/gpustat"'.format(gpu_info.name)
    info = os.popen(cmd).read()
    if len(info) == 0:
        print(gpu_info.name, " access failed")
        return False
    for status in info.split("\n"):
        if len(status.split("|")) == 1:
            continue
        if int(status[1]) == gpu_info.gnum:
            return len(status.split("|")[-1]) == 0
    return False

class GPUInfo(object):
    def __init__(self, name, str_status):
        self.name = name
        self.str_status = str_status
        status = str_status.split('|')
        self.gnum = int(status[0][1]) # [1]
        self.ginfo = status[0]
        self.usage = status[1]
        self.mem_status = status[2]
        self.vacant = len(status[-1]) == 0
        self.gid = self.name + '.' + str(self.gnum)

    def __repr__(self):
        return "{0.name}.{0.gum} vacant: {0.vacant} {0.str_status}".format(self)

class ServerInfo(object):
    def __init__(self, name, storage_root="/disk/ostrom/v1zhu2"):
        self.name = name
        self.storage_root = storage_root

    def update_gpu_info(self):
        cmd = 'ssh {} "/disk/ostrom/v1zhu2/p36env/bin/gpustat"'.format(self.name)
        info = os.popen(cmd).read()
        if len(info) == 0:
            print(self.name, " update failed")
        print(info)
        for status in info.split("\n"):
            if len(status.split("|")) == 1:
                continue
            gpu_info = GPUInfo(self.name, status)
            all_gpu.append(gpu_info)

            if gpu_info.vacant:
                vacant_gpu[gpu_info.gid] = gpu_info
            else:
                if gpu_info.gid in vacant_gpu:
                    del vacant_gpu[gpu_info.gid]
    
    def clean_cache(self):
        cmd = 'ssh {} rm /dev/shm/zfhu -rf'.format(self.name)
        try:
            print("Removing cache from {} ...".format(self.name))
            info = os.popen(cmd).read()
        except:
            print(info)
        print("Done!")

class Task(object):
    def __init__(self, cmd, src_dir=None):
        self.cmd = cmd
        self.src_dir = src_dir

    def get_name(self):
        cmd = self.cmd.split(' ')
        return time.strftime("%m.%d_%H:%M", time.gmtime()) + '-' + cmd[-1]

    def execute(self, gpu, gpu_queue):
        cmd = self.cmd.format(self.src_dir, gpu.gnum)
        cmd = "ssh -ntt {} \"{}\"".format(gpu.name, cmd + " -src " + self.src_dir)
        print(cmd)
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, encoding='utf-8')
        # result = os.popen(cmd, shell=True)
        log_path = "$ROOT/Project/parallel_log/" + self.get_name() + ".log"
        open(log_path, "w").write(result)
        print("Log is written to ", log_path)
        gpu_queue.put(gpu)

def run_cmds(cmds, num_gpu=20):
    src_dir = "$ROOT/Project/homo"
    import string
    import shutil
    tmp_src_dir = "/disk/ostrom/v1zhu2/tmp/src_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    shutil.copytree(src_dir, tmp_src_dir)
    print("Copied src from {} to {}".format(src_dir, tmp_src_dir))

    num_gpu = min(num_gpu, len(vacant_gpu))
    gpu_queue = Queue()
    gpu_keys = list(vacant_gpu.keys())
    random.shuffle(gpu_keys)
    for key in gpu_keys[:num_gpu]:
        gpu_queue.put(vacant_gpu.pop(key))

    jobs = []
    for cmd in cmds:
        task = Task(cmd, tmp_src_dir)
        tell_no_vacancy = False
        wait_start = None
        while gpu_queue.empty():
            if tell_no_vacancy:
                continue
            else:
                print("No GPU vacant, waiting...")
                tell_no_vacancy = True
                wait_start = time.time()
        if wait_start is not None:
            print("Watied for {:.2f}s".format(time.time() - wait_start))

        gpu = gpu_queue.get()
        pr = Process(target=task.execute, args=(gpu, gpu_queue))
        pr.start()
        jobs.append(pr)

    for job in jobs:
        job.join()

gpu_servers = [
    # removed for publishing
]

def update_server_info():
    for server in gpu_servers:
        server.update_gpu_info()
        server.clean_cache()
    print("there are {} gpus vacant among all {} valid gpus".format(len(vacant_gpu), len(all_gpu)))


update_server_info()
from utils.cmds_abandon import cmds
run_cmds(cmds, 14)


