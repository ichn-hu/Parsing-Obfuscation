import os
from collections import OrderedDict
import torch

def get_work_path():
    project_root = "$ROOT/Project/"
    possible_work_dirs = ["/disk/ostrom/s1884529", "/disk/scratch1", "/disk/scratch"]
    for dirname in possible_work_dirs:
        if os.path.exists(dirname):
            work_path = os.path.join(dirname, "zfhu", "work")
            os.makedirs(work_path, exist_ok=True)
            return work_path
    return "KeyError"


class AttrDict(dict):
    def valid_val(self, val):
        if self.__dict__["_blob"]:
            return True
        return isinstance(val, int) or isinstance(val, float) \
            or isinstance(val, list) or isinstance(val, str) \
            or isinstance(val, dict) or isinstance(val, tuple) \
            or isinstance(val, type(None)) or isinstance(val, AttrDict)

    @staticmethod
    def valid_key(key):
        if isinstance(key, str) and len(key) > 0:
            return key[0].isalpha()
        return False

    def __init__(self, cfgdict=None, fixed=True, blob=False):
        self.__dict__["_initialized"] = False
        super().__init__()
        self.__dict__["_fixed"] = fixed
        self.__dict__["_blob"] = blob
        if cfgdict is None:
            cfgdict = dict()

        for key, val in cfgdict.items():
            if isinstance(val, dict):
                val = MrDict(val)
            self.__setattr__(key, val)
        self.__dict__["_initialized"] = True

    def __setattr__(self, key, val):
        if self._initialized and not self.valid_key(key):
            raise KeyError("Key must be a valid python identifier that starts with letters")

        if self._initialized and not self.valid_val(val):
            raise KeyError("Only primitive types allowed, get " + str(val))

        self.__dict__.update({key: val})

    def __getattr__(self, item):
        if item not in self.__dict__ and not self.__dict__["_fixed"]:
            self.__dict__.update({item: AttrDict()})

        if item in self.__dict__:
            return self.__dict__[item]

        return None

    def __getitem__(self, item):
        return self.__getattr__(item)

    def dict(self):
        ret = {}
        for key, val in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(val, AttrDict):
                val = val.dict()
            ret[key] = val
        return ret

    def update(self, dic):
        self.__dict__.update(dic)
        return self

    def has(self, key):
        if key in self.__dict__:
            return True
        return False
    
    def fix(self):
        self.__dict__['_fixed'] = True

class MrDict(torch.nn.Module):
    def valid_val(self, val):
        if self.__dict__["_blob"]:
            return True
        return isinstance(val, int) or isinstance(val, float) \
            or isinstance(val, list) or isinstance(val, str) \
            or isinstance(val, dict) or isinstance(val, tuple) \
            or isinstance(val, type(None)) or isinstance(val, MrDict)

    @staticmethod
    def valid_key(key):
        if isinstance(key, str) and len(key) > 0:
            return key[0].isalpha()
        return False

    def __init__(self, cfgdict=None, fixed=True, blob=False):
        self.__dict__["_initialized"] = False
        super().__init__()
        self.__dict__["_fixed"] = fixed
        self.__dict__["_blob"] = blob
        if cfgdict is None:
            cfgdict = dict()

        for key in self.__dir__():
            if not key.startswith('_') and key not in {'valid_key', 'valid_val', 'dict', 'update', 'has', 'fix', 'add_module', 'apply', 'children', 'cpu', 'cuda', 'double', 'dump_patches', 'eval', 'extra_repr', 'float', 'forward', 'half', 'load_state_dict', 'modules', 'named_children', 'named_modules', 'named_parameters', 'parameters', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_parameter', 'share_memory', 'state_dict', 'to', 'train', 'training', 'type', 'zero_grad'}:
                cfgdict.update({key: getattr(self, key)})

        for key, val in cfgdict.items():
            if isinstance(val, dict):
                val = MrDict(val)
            self.__setattr__(key, val)
        self.__dict__["_initialized"] = True

    def __setattr__(self, key, val):
        if self._initialized and not self.valid_key(key):
            raise KeyError("Key must be a valid python identifier that starts with letters")

        if self._initialized and not self.valid_val(val):
            raise KeyError("Only primitive types allowed, get " + str(val))

        self.__dict__.update({key: val})

    def __getattr__(self, item):
        if item not in self.__dict__ and not self.__dict__["_fixed"]:
            self.__dict__.update({item: MrDict()})
        if item in self.__dict__:
            return self.__dict__[item]
        return None

    def __getitem__(self, item):
        return self.__getattr__(item)

    def dict(self):
        ret = {}
        for key, val in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(val, MrDict):
                val = val.dict()
            ret[key] = val
        return ret

    def update(self, dic):
        self.__dict__.update(dic)
        return self

    def has(self, key):
        if key in self.__dict__:
            return True
        return False
    
    def fix(self):
        self.__dict__['_fixed'] = True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)
        for key, val in self.__dict__.items():
            if val is not None:
                destination[prefix + key] = val
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name, param in state_dict.items():
            local_state = {}
            if name.startswith(prefix):
                key = name[len(prefix):]
                val = param
                local_state[key] = val
            self.__dict__.update(local_state)

    def forward(self):
        raise NotImplementedError