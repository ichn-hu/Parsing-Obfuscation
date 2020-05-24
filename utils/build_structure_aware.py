import torch
import torch.nn as nn
from net.parser import BiaffineAttnParser
from net.structure_aware import StructureAwareGenerator
from model.obfuscator import Obfuscator


def structure_aware():
