import resources

import numpy as np

RAND = np.random.RandomState(0)

class MalmoMission(object):
    def __init__(self, goal, hint):
        self.goal = goal
        self.hint = hint

    def prepare(self, env, client):
        pass

    def mark(self, env, client):
        reward = 0
        done = False
        command, args = self.goal[0], self.goal[1:]
        if command == "go":
            assert len(args) == 1
            arg = args[0]
            assert arg[0] == "where"
            constraint = arg[1]
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
            for comp in match:
                #print comp
                #print comp.cells
                dists = [abs(agent_x - x) + abs(agent_z - z) for x, y, z in
                        comp.cells]
                if len(dists) > 0 and min(dists) <= 1:
                    reward = 1
                    done = True
                    break

        else:
            assert False

        # TODO should is_mission_running test be in the client?
        return reward, done or not client.state.is_mission_running

def sample():
    goal_type = RAND.randint(3)
    #if goal_type == 0: # find by kind
    #    kind = _sample_kind()
    #    goal = ("go", ("where", kind))
    #    hint = (("go", kind),)
    #    return MalmoMission(goal, hint)
    #elif goal_type == 1: # find by material
    #    material = _sample_material()
    #    goal = ("go", ("where", material))
    #    hint = (("go", material),)
    #    return MalmoMission(goal, hint)
    #elif goal_type == 2: # find by location
    if True:
        wall_material = _sample_wall_material()
        goal = ("go", ("where", wall_material))
        hint = (("go", wall_material),)
        return MalmoMission(goal, hint)
    else:
        assert False
    # remove prop by material
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

def _sample_wall_material():
    return ("wall_material", RAND.choice(resources.MATERIALS_FOR["wall"]))
