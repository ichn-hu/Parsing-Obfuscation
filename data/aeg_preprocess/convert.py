import time
import os
from multiprocessing import Process, Queue
import random
import subprocess


AEG_DATA_ROOT = "$ROOT/Project/data/annotated_english_gigaword"
AGIGA_ROOT = "$ROOT/Software/agiga"
AGIGA_CMD = 'mvn exec:java -Dexec.mainClass="edu.jhu.agiga.StreamingSentenceReader" -Dexec.args="{path_to_gz} {path_to_output}"'
PROCESSED_OUT_DIR = "/disk/scratch1/zfhu/aeg_processed"

def get_cmd(inpfile, oupfile):
    return "cd {}\n{}".format(AGIGA_ROOT, AGIGA_CMD.format(path_to_gz=inpfile, path_to_output=oupfile))

class Task(object):
    def __init__(self, cmd):
        self.cmd = cmd

    def execute(self, gpu, gpu_queue):
        cmd = self.cmd
        print(cmd)
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        print(result)
        gpu_queue.put(gpu)

def process(cmds, num_worker=10):
    task_queue = Queue()
    for i in range(num_worker):
        task_queue.put(i)

    jobs = []
    from tqdm import tqdm
    for cmd in tqdm(cmds):
        task = Task(cmd)
        tell_no_vacancy = False
        wait_start = None
        while task_queue.empty():
            if tell_no_vacancy:
                continue
            else:
                print("No worker vacant, waiting...")
                tell_no_vacancy = True
                wait_start = time.time()
        if wait_start is not None:
            print("Watied for {:.2f}s".format(time.time() - wait_start))

        worker = task_queue.get()
        pr = Process(target=task.execute, args=(worker, task_queue))
        pr.start()
        jobs.append(pr)

    for job in jobs:
        job.join()


if __name__ == "__main__":
    xml_path = os.path.join(AEG_DATA_ROOT, "data", "xml")
    inpfiles = filter(lambda fn: fn.startswith("nyt"), os.listdir(xml_path))
    cmds = []
    for fn in inpfiles:
        cmd = get_cmd(os.path.join(xml_path, fn), os.path.join(PROCESSED_OUT_DIR, fn.split('.')[0] + ".txt"))
        cmds.append(cmd)
    process(cmds)
        


