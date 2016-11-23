import resources

from collections import namedtuple
import itertools
import numpy as np

RANDOM = np.random.RandomState(0)

ROOM_HEIGHT = 3
ROOM_WIDTH = 4
MEAN_NEIGHBORS = 1
ROOF_PROB = 0.7
MAX_ROOMS = 4
MIN_ROOMS = 2
MAX_VOIDS = 6
MAX_PROPS = 4
GROUND = 4 # TODO client manager knows this

Component = namedtuple("Component", ["kind", "material", "cells"])
Environment = namedtuple("Environment", ["components", "commands"])

DrawCuboid = namedtuple("DrawCuboid", ["x1", "y1", "z1", "x2", "y2", "z2", "material"])
DrawBlock = namedtuple("DrawBlock", ["x", "y", "z", "material"])

def sample(mission):
    kinds, materials, wall_materials = _get_requirements(mission)
    kinds.add("door")
    # TODO something smarter for conjunctive constraints
    while len(kinds) < 2:
        kinds.add(RANDOM.choice(resources.KINDS))
    while len(materials) < 2:
        materials.add(RANDOM.choice(resources.MATERIALS_FOR["window"] +
            resources.MATERIALS_FOR["prop"]))
    while len(wall_materials) < 2:
        wall_materials.add(RANDOM.choice(resources.MATERIALS_FOR["wall"]))

    kinds = list(kinds)
    materials = list(materials)
    wall_materials = list(wall_materials)
    wall_materials_ = list(wall_materials)
    default_wall_material = wall_materials[0]

    env = Environment({}, [])
    _add_room(env, (0, 0, 0), kinds, materials, wall_materials, default_wall_material)

    key_rooms = []
    for material in set(wall_materials_):
        mrooms = [c for c in env.components.values() if c.kind == "room" and
                c.material == material]
        key_rooms.append(mrooms[0])

    _add_voids(env, kinds, materials)
    _add_props(env, kinds, materials, key_rooms)

    return env

def _get_requirements(mission):
    kinds = []
    materials = []
    wall_materials = []
    for arg in mission.goal[1:]:
        assert arg[0] == "where"
        constraint = arg[1]
        cname, cval = constraint
        if cname == "kind":
            kinds.append(cval)
        elif cname == "material":
            materials.append(cval)
        elif cname == "wall_material":
            wall_materials.append(cval)

    return set(kinds), set(materials), set(wall_materials)

def _add_room(env, loc, kinds, materials, wall_materials, default_wall_material):
    assert loc not in env.components
    x, y, z = loc
    width, height = ROOM_WIDTH, ROOM_HEIGHT

    bottom = height * len(_below(loc, env))
    if len(wall_materials) > 0:
        material = wall_materials.pop()
    else:
        material = default_wall_material
    commands = [
        DrawCuboid(x*width, GROUND+bottom, z*width,
            (x+1)*width-1, GROUND+bottom+height-1, (z+1)*width-1,
            resources.MATERIALS[material]),
        DrawCuboid(x*width+1, GROUND+bottom, z*width+1,
            (x+1)*width-2, GROUND+bottom+height-2, (z+1)*width-2,
            resources.MATERIALS["air"])
    ]
    if y == 0:
        commands.append(DrawCuboid(x*width, GROUND-1, z*width,
            (x+1)*width-1, GROUND-1, (z+1)*width-1,
            resources.MATERIALS["floor"]))

    cells = frozenset(sum([_enclosed_cells(c) for c in commands], []))
    env.components[loc] = Component("room", material, cells)
    env.commands.extend(commands)

    if RANDOM.rand() < ROOF_PROB:
        _add_roof(env, (x, y+1, z))

    neighbor_locs = _free_neighbors(env, loc)
    RANDOM.shuffle(neighbor_locs)
    n_neighbors = max(RANDOM.poisson(MEAN_NEIGHBORS), 
            1 if len(env.components) < MIN_ROOMS else 0,
            1 if len(wall_materials) > 0 else 0)
    neighbor_locs = neighbor_locs[:n_neighbors]
    for nloc in neighbor_locs:
        if len([c for c in env.components.values() if c.kind == "room"]) >= MAX_ROOMS:
            break
        else:
            _add_room(env, nloc, kinds, materials, wall_materials, default_wall_material)

def _add_roof(env, loc):
    assert loc not in env.components
    x, y, z = loc

    bottom = ROOM_HEIGHT * len(_below(loc, env))
    material = resources.MATERIALS["roof"]
    env.components[loc] = Component("roof", material.name, frozenset())
    commands = []
    up = 0
    for inset in range(0, ROOM_WIDTH/2):
        material_here = material
        if int(up) != up:
            material_here = material._replace(type="double_"+material.type)
        commands.append(DrawCuboid(
            x*ROOM_WIDTH+inset, GROUND+bottom+int(up), z*ROOM_WIDTH+inset,
            (x+1)*ROOM_WIDTH-inset-1, GROUND+bottom+int(up), (z+1)*ROOM_WIDTH-inset-1,
            material_here))
        up += 0.5
    env.commands.extend(commands)

def _add_voids(env, kinds, materials):
    window_materials = [m for m in materials if m in resources.MATERIALS_FOR["window"]]
    n_doors = kinds.count("door")
    n_windows = max(kinds.count("window"), len(window_materials))
    voids = ["door"] * n_doors + ["window"] * n_windows
    for kind in ("door", "window"):
        while kind in kinds:
            kinds.remove(kind)
    n_voids = 1 + RANDOM.randint(MAX_VOIDS-1)
    while len(voids) < n_voids:
        voids.append(RANDOM.choice(("door", "window")))
    voids = list(reversed(voids))

    interior, exterior = _walls(env)
    RANDOM.shuffle(exterior)
    for loc, dir in exterior:
        if len(voids) == 0:
            break
        x, y, z = loc
        if y > 0:
            continue
        dx, dz = dir
        bottom = ROOM_HEIGHT * len(_below(loc, env))
        kind = voids.pop()
        if kind == "door":
            dy = 0
            material = resources.MATERIALS["air"]
        elif kind == "window":
            dy = 1
            if len(window_materials) > 0:
                mat_name = window_materials.pop()
                materials.remove(mat_name)
                material = resources.MATERIALS[mat_name]
            else:
                material = resources.MATERIALS[
                        RANDOM.choice(resources.MATERIALS_FOR["window"])]
        else:
            assert False

        if dx == 0:
            if dz == 1:
                dzl = 2
                dzh = 1
            elif dz == -1:
                dzl = -1
                dzh = -2
            command = DrawCuboid(
                    x*ROOM_WIDTH+1, GROUND+bottom+dy, z*ROOM_WIDTH+1+dzl,
                    (x+1)*ROOM_WIDTH-2, GROUND+bottom+ROOM_HEIGHT-2, (z+1)*ROOM_WIDTH-2+dzh,
                    material)
        else:
            if dx == 1:
                dxl = 2
                dxh = 1
            elif dx == -1:
                dxl = -1
                dxh = -2
            command = DrawCuboid(
                    x*ROOM_WIDTH+1+dxl, GROUND+bottom+dy, z*ROOM_WIDTH+1,
                    (x+1)*ROOM_WIDTH-2+dxh, GROUND+bottom+ROOM_HEIGHT-2, (z+1)*ROOM_WIDTH-2,
                    material)
        env.commands.append(command)
        env.components[x, y, z, dx, dz] = Component(kind, material.name,
                frozenset(_enclosed_cells(command)))

    assert len(window_materials) == 0

    for loc, dir in interior:
        x, y, z = loc
        dx, dz = dir
        bottom = len(_below(loc, env))
        command = DrawCuboid(
                x*ROOM_WIDTH+1+dx, GROUND+bottom, z*ROOM_WIDTH+1+dz,
                (x+1)*ROOM_WIDTH-2+dx, GROUND+bottom+ROOM_HEIGHT-2, (z+1)*ROOM_WIDTH-2+dz,
                resources.MATERIALS["air"])
        env.commands.append(command)

def _add_props(env, kinds, materials, key_rooms):
    prop_materials = [m for m in materials if m in resources.MATERIALS_FOR["prop"]]
    n_props = max(kinds.count("prop"), len(prop_materials), 1 + RANDOM.randint(MAX_PROPS-1))
    while "prop" in kinds:
        kinds.remove("prop")
    for material in prop_materials:
        materials.remove(material)

    rooms = [(pos, comp) for pos, comp in env.components.items() 
            if pos[1] == 0 and comp.kind == "room"]
    RANDOM.shuffle(rooms)
    rooms = sorted(rooms, key=lambda pc: 0 if pc[1] in key_rooms else 1)
    for pos, room in rooms[:n_props]:
        x, y, z = min(room.cells)
        if len(prop_materials) > 0:
            material = resources.MATERIALS[prop_materials.pop()]
        else:
            material = resources.MATERIALS[
                    RANDOM.choice(resources.MATERIALS_FOR["prop"])]
        pos = (x+1, y+1, z+1)
        env.commands.append(DrawBlock(x+1, y+1, z+1, material))
        env.components[pos] = Component("prop", material.name, frozenset([pos]))

def _below(loc, env):
    x, y, z = loc
    return [l for l, c in env.components.items() 
            if c.kind == "room" and l[0] == x and l[1] < y and l[2] == z]

def _enclosed_cells(cuboid):
    # TODO do we actually want to count the empty parts?
    return list(itertools.product(
            range(cuboid[0], cuboid[3]+1),
            range(cuboid[1], cuboid[4]+1),
            range(cuboid[2], cuboid[5]+1)))

def _free_neighbors(env, loc):
    x, y, z = loc
    neighbors = [(x+1, y, z), (x, y+1, z), (x, y, z+1)]
    out = []
    for nloc in neighbors:
        if nloc in env.components:
            continue
        if not(len(_below(nloc, env)) == nloc[1]):
            continue
        out.append(nloc)
    return out

def _walls(env):
    interior = []
    exterior = []
    for loc in env.components:
        if env.components[loc].kind != "room":
            continue
        x, y, z = loc
        neighbors = [(x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)]
        for npos in neighbors:
            nx, ny, nz = npos
            if npos in env.components and env.components[npos].kind == "room":
                interior.append((loc, (nx-x, nz-z)))
            else:
                exterior.append((loc, (nx-x, nz-z)))
    return interior, exterior
