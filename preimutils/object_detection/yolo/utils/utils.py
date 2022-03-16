import json
import os

def jsonfile2dict(json_dir: str) -> dict:
    f = open(json_dir, 'r')
    d = json.load(f)
    f.close()
    return d

def mkdir_p(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)