import resources

import numpy as np

RAND = np.random.RandomState(0)

class MalmoMission(object):
    def __init__(self, goal, hint):
        self.goal = goal
        self.hint = hint

    def prepare(self, env, client):
        command, args = self.goal[0], self.goal[1:]
        arg0 = args[0]
        if command == "remove":
            # TODO redundant check
            assert arg0[0] == "where"
            assert arg0[1][0] == "material"
            client.target_material_counts = client.material_counts.copy()
            client.target_material_counts[resources.MATERIAL_INDEX[arg0[1][1]]] -= 1

    def mark(self, env, client):
        command, args = self.goal[0], self.goal[1:]
        arg0 = args[0]
        if command == "go":
            assert len(args) == 1
            assert arg0[0] == "where"
            reward, done = _test_go(arg0[1], env, client)
        elif command == "remove":
            assert len(args) == 1
            assert arg0[0] == "where"
            reward, done = _test_remove(arg0[1], env, client)
        else:
            assert False

        # TODO should is_mission_running test be in the client?
        return reward, done or not client.state.is_mission_running

def _test_go(constraint, env, client):
    cname, cval = constraint
    if cname == "kind":
        match = [c for c in env.components.values() if c.kind == cval]
    elif cname == "material":
        match = [c for c in env.components.values() if c.material == cval]
    elif cname == "wall_material":
        rooms = [c for c in env.components.values() if c.kind == "room"
                and c.material == cval]
        match = []
        for c in env.components.values():
            if c.kind == "room":
                continue
            mrooms = [r for r in rooms if len(r.cells & c.cells) > 0]
            if len(mrooms) > 0:
                match.append(c)
    else:
        assert False

    agent_x, agent_z = client.coordinate.x, client.coordinate.z
    reward = 0
    done = False
    for comp in match:
        dists = [abs(agent_x - x) + abs(agent_z - z) for x, y, z in
                comp.cells]
        if len(dists) > 0 and min(dists) <= 1:
            reward = 1
            done = True
            break

    return reward, done

def _test_remove(constraint, env, client):
    cname, cval = constraint
    assert cname == "material"
    i_mat = resources.MATERIAL_INDEX[cval]
    reward, done = 0, False

    if client.material_counts[i_mat] == client.target_material_counts[i_mat]:
        reward = 1
        done = True
    return reward, done

def sample():
    goal_type = RAND.randint(4)
    if goal_type == 0: # find by kind
        kind = _sample_kind()
        goal = ("go", ("where", kind))
        hint = (("go", kind[1]),)
        return MalmoMission(goal, hint)
    elif goal_type == 1: # find by material
        material = _sample_material()
        goal = ("go", ("where", material))
        hint = (("go", material[1]),)
        return MalmoMission(goal, hint)
    elif goal_type == 2: # find by location
        wall_material = _sample_wall_material()
        goal = ("go", ("where", wall_material))
        hint = (("go", wall_material[1]),)
        return MalmoMission(goal, hint)
    elif goal_type == 3: # remove prop by material
        material = _sample_prop_material()
        goal = ("remove", ("where", material))
        hint = (("go", material[1]), ("remove", material[1]))
        return MalmoMission(goal, hint)
    else:
        assert False
    # remove prop in room by material
    # remove prop with neighbor by material
    # create prop by material
    # create prop in room with material
    # create prop with neighbor by material
    # remove window by material
    # remove window with neighbor
    # remove window in room
    # replace window from material with material
    # replace window in room with material
    # replace window with neighbor with material
    # create window with material
    # create window in room with material
    # create window with neighbor with material
    # fill door
    # fill door in location
    # fill door with neighbor
    # create door
    # create door in location
    # create door with neighbor

def _sample_kind():
    return ("kind", RAND.choice(resources.KINDS))

def _sample_material():
    candidates = resources.MATERIALS_FOR["prop"] + resources.MATERIALS_FOR["window"]
    return ("material", RAND.choice(candidates))

def _sample_prop_material():
    return ("material", RAND.choice(resources.MATERIALS_FOR["prop"]))

def _sample_wall_material():
    return ("wall_material", RAND.choice(resources.MATERIALS_FOR["wall"]))
