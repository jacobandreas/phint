from misc import util

from collections import namedtuple
import itertools
import numpy as np

GROUND = 4
#HEIGHTS = [3, 4]
HEIGHTS = [3]
WIDTH = 4
MEAN_NEIGHBORS = 1
ROOF_PROB = 0.667
MAX_CUBES = 4
NN_VOIDS = [2,3,4,5]
WALL_MATERIALS = [
    #("stained_hardened_clay", "RED", None),
    #("stained_hardened_clay", "GREEN", None),
    #("stained_hardened_clay", "BLUE", None)
    ("stone", None, "smooth_andesite"),
    ("planks", None, "birch"),
    ("stained_hardened_clay", "RED", None)
]
ROOF_MATERIAL = ("wooden_slab", None, "dark_oak")
WINDOW_MATERIALS = [
    #("stained_glass", "RED", None),
    #("stained_glass", "GREEN", None),
    #("stained_glass", "BLUE", None)
    ("stained_glass", None, None),
    ("ice", None, None),
    ("diamond_block", None, None)
]
AIR = ("air", None, None)
STONE = ("stone", None, None)
PROPS = ["bookshelf", "anvil", "chest", "cauldron"]

Building = namedtuple("Building", ["hl", "ll", "wall_materials", "window_materials"])
Cube = namedtuple("Cube", ["kind", "height", "material", "cells"])
Roof = namedtuple("Roof", ["kind", "material", "cells"])
Void = namedtuple("Void", ["kind", "material", "cells"])
Prop = namedtuple("Prop", ["kind", "material", "cells"])

MCuboid = namedtuple("MCuboid", ["x1", "y1", "z1", "x2", "y2", "z2", "material"])
MItem = namedtuple("MItem", ["x", "y", "z", "type"])

random = np.random.RandomState(0)

KINDS = [w[0] for w in WALL_MATERIALS] + [w[0] for w in WINDOW_MATERIALS] \
        + PROPS + [STONE, AIR]
index = util.Index()
for k in KINDS: index.index(k)

def below(state, loc):
    x, y, z = loc
    return [(l, c) for l, c in state.hl.items() 
            if isinstance(c, Cube) and l[0] == x and l[1] < y and l[2] == z]

def free_neighbors(state, loc):
    x, y, z = loc
    neighbors = [(x+1, y, z), (x, y+1, z), (x, y, z+1)]
    out = []
    for nloc in neighbors:
        if nloc in state.hl:
            continue
        if not(len(below(state, nloc)) == nloc[1]):
            continue
        out.append(nloc)
    return out

def walls(state):
    interior = []
    exterior = []
    for loc in state.hl:
        if not isinstance(state.hl[loc], Cube):
            continue
        x, y, z = loc
        neighbors = [(x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)]
        for neighbor in neighbors:
            nx, ny, nz = neighbor
            if neighbor in state.hl and isinstance(state.hl[neighbor], Cube):
                interior.append((loc, (nx-x, nz-z)))
            else:
                exterior.append((loc, (nx-x, nz-z)))
    return interior, exterior

def enclosed_cells(cuboid):
    # TODO do we actually want to count the empty parts?
    return list(itertools.product(
            range(cuboid[0], cuboid[3]+1),
            range(cuboid[1], cuboid[4]+1),
            range(cuboid[2], cuboid[5]+1)))

def build_cube(state, loc):
    assert loc not in state.hl
    x, y, z = loc

    bottom = sum(c.height for l, c in below(state, loc))
    height = random.choice(HEIGHTS)
    material = state.wall_materials.pop()

    ll = [
        MCuboid(x*WIDTH, GROUND+bottom, z*WIDTH, 
                (x+1)*WIDTH-1, GROUND+bottom+height-1, (z+1)*WIDTH-1, 
                material),
        MCuboid(x*WIDTH+1, GROUND+bottom, z*WIDTH+1, 
                (x+1)*WIDTH-2, GROUND+bottom+height-2, (z+1)*WIDTH-2, 
                AIR)
    ]
    if y == 0:
        ll.append(MCuboid(x*WIDTH, GROUND-1, z*WIDTH, 
                (x+1)*WIDTH-1, GROUND-1, (z+1)*WIDTH-1, 
                STONE))

    cells = frozenset(sum([enclosed_cells(cuboid) for cuboid in ll], []))
    state.hl[loc] = Cube("room", height, material, cells)
    state.ll.extend(ll)

    if random.rand() < ROOF_PROB:
        build_roof(state, (x, y+1, z))

    neighbor_locs = free_neighbors(state, loc)
    random.shuffle(neighbor_locs)
    neighbor_locs = neighbor_locs[:random.poisson(MEAN_NEIGHBORS)]
    for nloc in neighbor_locs:
        if len([c for c in state.hl.values() if isinstance(c, Cube)]) >= MAX_CUBES:
            break
        else:
            build_cube(state, nloc)

def build_roof(state, loc):
    assert loc not in state.hl
    x, y, z = loc

    bottom = sum(c.height for l, c in below(state, loc))
    material = ROOF_MATERIAL
    state.hl[loc] = Roof("roof", material, frozenset())
    ll_parts = []
    up = 0
    #for inset in range(-1, WIDTH/2):
    for inset in range(0, WIDTH/2):
        if int(up) == up:
            material_here = material
        else:
            material_here = ("double_" + material[0],) + material[1:]
        ll_parts.append(MCuboid(
            x*WIDTH+inset, GROUND+bottom+int(up), z*WIDTH+inset,
            (x+1)*WIDTH-inset-1, GROUND+bottom+int(up), (z+1)*WIDTH-inset-1,
            material_here))
        up += 0.5
    state.ll.extend(ll_parts)

def add_voids(state):
    interior, exterior = walls(state)
    random.shuffle(exterior)
    n_voids = random.choice(NN_VOIDS)
    curr_voids = 0
    made_door = False
    for loc, dir in exterior:
        if curr_voids > n_voids:
            break
        x, y, z = loc
        if y > 0:
            continue
        dx, dz = dir
        bottom = sum(c.height for l, c in below(state, loc))
        height = state.hl[loc].height
        if not made_door:
            dy = 0
        elif y > 0:
            dy = 1
        else:
            dy = random.randint(2)
        if dy == 1:
            kind = "window"
            material = state.window_materials.pop()
        else:
            kind = "door"
            material = AIR
            made_door = True

        if dx == 0:
            if dz == 1:
                dzl = 2
                dzh = 1
            elif dz == -1:
                dzl = -1
                dzh = -2
            ll = MCuboid(
                    x*WIDTH+1, GROUND+bottom+dy, z*WIDTH+1+dzl,
                    (x+1)*WIDTH-2, GROUND+bottom+height-2, (z+1)*WIDTH-2+dzh,
                    material)
        else:
            if dx == 1:
                dxl = 2
                dxh = 1
            elif dx == -1:
                dxl = -1
                dxh = -2
            ll = MCuboid(
                    x*WIDTH+1+dxl, GROUND+bottom+dy, z*WIDTH+1,
                    (x+1)*WIDTH-2+dxh, GROUND+bottom+height-2, (z+1)*WIDTH-2,
                    material)
        state.ll.append(ll)
        state.hl[x, y, z, dx, dz] = Void(kind, material, frozenset(enclosed_cells(ll)))
        curr_voids += 1

    for loc, dir in interior:
        x, y, z = loc
        dx, dz = dir
        bottom = sum(c.height for l, c in below(state, loc))
        height = state.hl[loc].height
        ll = MCuboid(
                x*WIDTH+1+dx, GROUND+bottom, z*WIDTH+1+dz,
                (x+1)*WIDTH-2+dx, GROUND+bottom+height-2, (z+1)*WIDTH-2+dz,
                AIR)
        state.ll.append(ll)

def add_props(state):
    n_props = random.choice(len(PROPS))
    props = []
    for _ in range(n_props):
        props.append(PROPS[random.choice(len(PROPS))])
    cubes = [(k, v) for k, v in state.hl.items() if k[1] == 0 and isinstance(v, Cube)]
    random.shuffle(cubes)
    for k, cube in cubes:
        if len(props) == 0:
            break
        x, y, z = min(cube.cells)
        prop = props.pop()
        state.ll.append(MItem(x+1, y+1, z+1, prop))
        state.hl[k, "prop"] = Prop("prop", (prop, None, None), frozenset([(x+1, y+1, z+1)]))

def draw_mc(state):
    out = []
    for draw in state.ll:
        if isinstance(draw, MCuboid):
            s = '<DrawCuboid x1="%s" y1="%s" z1="%s" x2="%s" y2="%s" z2="%s"' % draw[:-1]
            s += ' type="%s"' % draw.material[0]
            if draw.material[1] is not None:
                s += ' colour="%s"' % draw.material[1]
            if draw.material[2] is not None:
                s += ' variant="%s"' % draw.material[2]
            s += '/>'
        elif isinstance(draw, MItem):
            s = '<DrawBlock x="%s" y="%s" z="%s" type="%s"/>' % draw
        out.append(s)
    return "\n".join(out)

def sample():
    def choose(lst):
        return lst[random.choice(len(lst))]

    primary_wall = choose(WALL_MATERIALS)
    secondary_wall = choose(WALL_MATERIALS)
    wall_materials = [secondary_wall] * MAX_CUBES + [primary_wall]
    primary_window = choose(WINDOW_MATERIALS)
    secondary_window = choose(WINDOW_MATERIALS)
    window_materials = [secondary_window] * max(NN_VOIDS) + [primary_window]

    state = Building({}, [], wall_materials, window_materials)
    build_cube(state, (0, 0, 0))
    add_voids(state)
    add_props(state)

    landmarks = state.hl.values()
    spec = draw_mc(state)
    return landmarks, spec
