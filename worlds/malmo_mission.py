import malmo_gen
import numpy as np

from collections import defaultdict

#def sample():
#    x = np.random.randint(-5, 6)
#    z = np.random.randint(-5, 6)
#    landmarks = [malmo_gen.Cube("target", 1, "stone", [(x, 4, z)])]
#    spec = '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="{}"/>'.format(
#            x, 4, z, x, 4, z, "stone")
#    goal = ("find", "target")
#    return MalmoMission(landmarks, spec, goal)

GOAL_TYPES = ["find", "find_constrained"]

def sample():
    goal_type = np.random.randint(3)
    while True:
        landmarks, spec = malmo_gen.sample()
        landmarks = parse_scene(landmarks)
        goal = sample_goal(landmarks, goal_type)
        if goal is None:
            continue
        return MalmoMission(landmarks, spec, goal)

def parse_scene(landmarks):
    t_landmarks = {}
    for landmark in landmarks:
        if isinstance(landmark, malmo_gen.Cube) \
                or isinstance(landmark, malmo_gen.Roof):
            continue
        attributes = [landmark.kind, landmark.material[0]]
        containing_room = [l for l in landmarks if isinstance(l,
            malmo_gen.Cube) and len(l.cells & landmark.cells) > 0]
        assert len(containing_room) == 1
        containing_room = containing_room[0]
        attributes.append(("in", containing_room.material[0]))
        neighbors = [l for l in landmarks if l != landmark and 
                l != containing_room and len(l.cells & containing_room.cells) > 0]
        for neighbor in neighbors:
            attributes.append(("with", neighbor.kind))
        t_landmarks[landmark] = attributes
    return t_landmarks

def sample_goal(landmarks, goal_type):
    if goal_type == 0: # find kind
        kinds = list(set(l.kind for l in landmarks))
        if len(kinds) == 0:
            return None
        goal = ("find", np.random.choice(kinds))

    elif goal_type == 1: # find any attribute
        candidates = landmarks.keys()
        if len(candidates) == 0:
            return None
        np.random.shuffle(candidates)
        attrs = landmarks[candidates[0]]
        np.random.shuffle(attrs)
        goal = ("find", attrs[0])

    elif goal_type == 2: # find with multiple
        candidates = [l for l in landmarks if len(landmarks[l]) > 1]
        if len(candidates) == 0:
            return None
        np.random.shuffle(candidates)
        attrs = landmarks[candidates[0]]
        np.random.shuffle(attrs)
        goal = ("find",) + tuple(attrs[:2])

    else:
        assert False

    return goal

class MalmoMission(object):
    def __init__(self, landmarks, spec, goal):
        self.landmarks = landmarks
        self.spec = spec
        self.goal = goal
        self.action = self.goal[0]
        self.attrs = self.goal[1:]

    def test(self, world):
        if self.action == "find":
            for landmark in self.landmarks:
                if not all(attr in self.landmarks[landmark] for attr in self.attrs):
                    continue

                dists = [abs(world.x - x) + abs(world.z - z) 
                        for x, y, z in landmark.cells]
                if min(dists) in (0, 1):
                    return True

            return False
        else:
            assert False
