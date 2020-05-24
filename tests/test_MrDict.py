import sys
sys.path.append("/mnt/e/Projects/Homomorphic-Obfuscation/src")
from config.utils import AttrDict

a_dict = AttrDict({
    "age": 12,
    "name": "Sergii",
    "family": {
        "father": "Apoplectic",
        "mother": "antic",
        "siblings": {
            "sister": {
                "name": "Wasserstein"
            }
        }
    }
})

assert a_dict.age == 12
assert a_dict.family.father == "Apoplectic"
assert a_dict.sdf is None
assert a_dict.family.siblings.sister.name == "Wasserstein"