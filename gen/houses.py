from collections import namedtuple
import numpy as np

GROUND = 4
HEIGHTS = [3, 4]
WIDTH = 4
MEAN_NEIGHBORS = 1
ROOF_PROB = 0.667
MAX_CUBES = 3
MAX_VOIDS = 5
CUBE_MATERIALS = ["brick_block", "stonebrick", "log"]
ROOF_MATERIALS = ["clay", "stained_hardened_clay", "hardened_clay"]

Building = namedtuple("Building", ["hl", "ll"])
Cube = namedtuple("Cube", ["height", "material"])
Roof = namedtuple("Roof", ["material"])

MCuboid = namedtuple("MCuboid", ["x1", "y1", "z1", "x2", "y2", "z2", "type"])

random = np.random.RandomState(0)

def below(state, loc):
    x, y, z = loc
    return [(l, c) for l, c in state.hl.items() if l[0] == x and l[1] < y and l[2] == z]

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

def free_walls(state):
    out = []
    for loc in state.hl:
        if not isinstance(state.hl[loc], Cube):
            continue
        x, y, z = loc
        neighbors = [(x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)]
        free_neighbors = [n for n in neighbors if n not in state.hl]
        for nx, ny, nz in free_neighbors:
            out.append((loc, (nx-x, nz-z)))
    return out

def build_cube(state, loc):
    assert loc not in state.hl
    x, y, z = loc

    bottom = sum(c.height for l, c in below(state, loc))
    height = random.choice(HEIGHTS)
    material = random.choice(CUBE_MATERIALS)
    state.hl[loc] = Cube(height, material)
    state.ll.extend([
        MCuboid(x*WIDTH, GROUND+bottom, z*WIDTH, 
                (x+1)*WIDTH-1, GROUND+bottom+height-1, (z+1)*WIDTH-1, 
                material),
        MCuboid(x*WIDTH+1, GROUND+bottom, z*WIDTH+1, 
                (x+1)*WIDTH-2, GROUND+bottom+height-2, (z+1)*WIDTH-2, 
                "air")
    ])

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
    material = random.choice(ROOF_MATERIALS)
    state.hl[loc] = Roof(material)
    ll_parts = []
    up = 0
    for inset in range(-1, WIDTH/2):
        ll_parts.append(MCuboid(
            x*WIDTH+inset, GROUND+bottom+up, z*WIDTH+inset,
            (x+1)*WIDTH-inset-1, GROUND+bottom+up, (z+1)*WIDTH-inset-1,
            material))
        up += 1
    state.ll.extend(ll_parts)

def add_voids(state):
    candidates = free_walls(state)
    random.shuffle(candidates)
    candidates = candidates[:random.randint(MAX_VOIDS)]
    for loc, dir in candidates:
        x, y, z = loc
        dx, dz = dir
        bottom = sum(c.height for l, c in below(state, loc))
        height = state.hl[loc].height
        if y > 0:
            dy = 1
        else:
            dy = random.randint(2)
        state.ll.append(MCuboid(
            x*WIDTH+1+dx, GROUND+bottom+dy, z*WIDTH+1+dz,
            (x+1)*WIDTH-2+dx, GROUND+bottom+height-2, (z+1)*WIDTH-2+dz,
            "air"))

def draw_mc(state):
    out = []
    for cuboid in state.ll:
        s = '<DrawCuboid x1="%s" y1="%s" z1="%s" x2="%s" y2="%s" z2="%s" type="%s"/>' % cuboid
        out.append(s)
    return "\n".join(out)

def main():
    for i in range(10):
        state = Building({}, [])
        build_cube(state, (0, 0, 0))
        add_voids(state)
        template = TEMPLATE % draw_mc(state)
        with open("spec_%d.xml" % i, "w") as spec_f:
            #print
            #for h in state.hl:
            #    print h, state.hl[h]
            print i, len(state.hl)
            print >>spec_f, template

if __name__ == "__main__":
    main()
