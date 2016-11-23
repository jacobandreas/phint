from misc import util

from collections import defaultdict, namedtuple

Material = namedtuple("Material", ["name", "type", "color", "variant"])

MATERIALS = {
    "stone": Material("stone", "stone", None, "smooth_andesite"),
    "wood": Material("wood", "planks", None, "birch"),
    "roof": Material("roof", "wooden_slab", None, "dark_oak"),
    "clay": Material("clay", "stained_hardened_clay", "RED", None),
    "glass": Material("glass", "glass", None, None),
    "ice": Material("ice", "ice", None, None),
    "greenglass": Material("greenglass", "stained_glass", "GREEN", None),
    "air": Material("air", "air", None, None),
    "floor": Material("floor", "clay", None, None),
    "bookshelf": Material("bookshelf", "bookshelf", None, None),
    "anvil": Material("anvil", "anvil", None, None),
    "chest": Material("chest", "chest", None, None),
    "cauldron": Material("cauldron", "cauldron", None, None)
}
MATERIAL_INDEX = util.Index()
for material in MATERIALS.keys():
    MATERIAL_INDEX.index(material)
N_MATERIALS = len(MATERIAL_INDEX)
TYPE_TO_NAME = defaultdict(lambda: "XXX") # TODO
TYPE_TO_NAME.update({m.type: m.name for m in MATERIALS.values()})

MATERIALS_FOR = {
    "wall": ["stone", "wood", "clay"],
    "window": ["glass", "ice", "greenglass"],
    "door": ["air"],
    "prop": ["bookshelf", "anvil", "chest", "cauldron"]
}

KINDS = ["prop", "window", "door"]
