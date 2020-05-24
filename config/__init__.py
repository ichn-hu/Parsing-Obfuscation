import os
import sys
import importlib

config_file = os.environ.get('config', 'basic')
if not config_file.startswith("config."):
    config_file = "config." + config_file

config = importlib.import_module(config_file)

this = sys.modules[__name__]
for attr in dir(config):
    cont = getattr(config, attr)
    setattr(this, attr, cont)

