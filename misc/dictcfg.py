"""
mimic the builtin dict
"""

class MrDict():
    @staticmethod
    def valid_val(val):
        return isinstance(val, int) or isinstance(val, float) \
            or isinstance(val, list) or isinstance(val, str) \
            or isinstance(val, dict) or isinstance(val, tuple) \
            or isinstance(val, type(None)) or isinstance(val, MrDict)

    @staticmethod
    def valid_key(key):
        if isinstance(key, str) and len(key) > 0:
            return key[0].isalpha()
        return False

    def __init__(self, cfgdict=None, fixed=True):
        self.__dict__["_fixed"] = fixed
        if cfgdict is None:
            cfgdict = dict()

        for key in self.__dir__():
            if not key.startswith('_') and key not in {'valid_key', 'valid_val', 'dict', 'has'}:
                cfgdict.update({key: getattr(self, key)})

        for key, val in cfgdict.items():
            if isinstance(val, dict):
                val = MrDict(val)
            self.__setattr__(key, val)

    def __setattr__(self, key, val):
        if not self.valid_key(key):
            raise KeyError("Key must be a valid python identifier that starts with letters")

        if not self.valid_val(val):
            raise KeyError("Only primitive types allowed, get " + str(val))

        self.__dict__.update({key: val})

    def __getattr__(self, item):
        if item not in self.__dict__ and not self._fixed:
            self.__dict__.update({item: MrDict()})
        if item in self.__dict__:
            return self.__dict__[item]
        return None

    def __getitem__(self, item):
        return self.__dict__[item]

    def dict(self):
        ret = {}
        for key, val in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(val, MrDict):
                val = val.dict()
            ret[key] = val
        return ret

    def has(self, key):
        if key in self.__dict__:
            return True
        return False


class MrDictCfg(MrDict):
    param0 = 1
    param1 = [1, 2, 3]
    param2 = "ok"


def test_init():
    print("test Init")
    dct = MrDict({'a': 10, 'b': {'c': "ok"}, 'd': [1, 2, 3]})
    print(dct.a == 10)
    print(dct.b.c == "ok")
    print(dct.d[2] == 3)

def test_assign():
    print("test Assign")
    dct = MrDict()
    dct.a = 10
    print(dct.a == 10)
    dct.b.d = 12
    dct.b.x = [1, 2, 3]
    print(dct.b.d == 12)
    print(dct.b.x[0] == 1)
    dct.b.c.d = 1
    print(dct.t)

def test_cfg():
    cfg = MrDictCfg()
    print(cfg.has("param0"))
    print(cfg.param1)
    print(cfg.dict())

test_cfg()
