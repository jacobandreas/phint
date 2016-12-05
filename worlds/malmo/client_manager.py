from env_generator import DrawCuboid, DrawBlock
import resources

from collections import namedtuple
import json
import MalmoPython
import numpy as np
from skimage.measure import block_reduce
import time

# COORDINATES

Coordinate = namedtuple("Coordinate", ["x", "y", "z", "pitch", "yaw"])


# PARAMS

DISCRETE = True
DURATION = 50000

HBOUNDS = (-5, 20)
VBOUNDS = (0, 20)

OBS_HBOUNDS = (-2, 2)
OBS_HBOUNDS_BIG = (-12, 12)
OBS_HBOUNDS_BIG_SCALE = 5
OBS_VBOUNDS = (-1, 4)
OBS_VBOUNDS_BIG = (0, 4)
OBS_VBOUNDS_BIG_SCALE = 5

OBS_HSIZE = OBS_HBOUNDS[1] - OBS_HBOUNDS[0] + 1
OBS_HSIZE_BIG = OBS_HBOUNDS_BIG[1] - OBS_HBOUNDS_BIG[0] + 1
OBS_VSIZE = OBS_VBOUNDS[1] - OBS_VBOUNDS[0] + 1
OBS_VSIZE_BIG = OBS_VBOUNDS_BIG[1] - OBS_VBOUNDS_BIG[0] + 1

TICK = 10
TICK_SCALE = 5


# SPECS

TEMPLATE_ARGS = {
    "nh": HBOUNDS[0], 
    "ph": HBOUNDS[1], 
    "nhl": HBOUNDS[0]-1, 
    "phl": HBOUNDS[1]+1,
    "nhc": HBOUNDS[0]-5,
    "phc": HBOUNDS[1]+5,
    "bottom": 0,
    "nv": 3,
    "nvc": 4,
    "pv": VBOUNDS[1]+5,
    "o_nh": OBS_HBOUNDS[0],
    "o_ph": OBS_HBOUNDS[1],
    "o_nv": OBS_VBOUNDS[0],
    "o_pv": OBS_VBOUNDS[1],
    "ob_nh": OBS_HBOUNDS_BIG[0],
    "ob_ph": OBS_HBOUNDS_BIG[1],
    "ob_nv": OBS_VBOUNDS_BIG[0],
    "ob_pv": OBS_VBOUNDS_BIG[1],
    "duration": DURATION,
    "tick": TICK
}

SPEC_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary/>
  </About>

  <ModSettings>
      <MsPerTick>{tick}</MsPerTick>
  </ModSettings>

  <ServerSection>
    <ServerInitialConditions>
      <Time>
        <StartTime>8000</StartTime>
        <AllowPassageOfTime>false</AllowPassageOfTime>
      </Time>
    </ServerInitialConditions>
    <ServerHandlers>
        <FlatWorldGenerator/>
        <DrawingDecorator>
          <DrawCuboid x1="{nhl}" y1="{nv}" z1="{nhl}" x2="{phl}" y2="{pv}" z2="{phl}" type="wool"/>
          <DrawCuboid x1="{nh}" y1="{bottom}" z1="{nh}" x2="{ph}" y2="{pv}" z2="{ph}" type="grass"/>
          <DrawCuboid x1="{nhc}" y1="{nvc}" z1="{nhc}" x2="{phc}" y2="{pv}" z2="{phc}" type="air"/>
          {{}}
        </DrawingDecorator>
        <ServerQuitFromTimeUp timeLimitMs="{duration}"/>
        <ServerQuitWhenAnyAgentFinishes/>
      </ServerHandlers>
    </ServerSection>
    
    <AgentSection mode="Creative">
      <Name>Agent</Name>
      <AgentStart>
          <Placement x="-1.5" y="4" z="0.5" pitch="0" yaw="0"/>
        <Inventory>
            <InventoryItem slot="0" type="stonebrick"/>
        </Inventory>
      </AgentStart>
      <AgentHandlers>
        <ObservationFromFullStats/>
        <ObservationFromGrid>
            <Grid name="agent_near">
                <min x="{o_nh}" y="{o_nv}" z="{o_nh}"/>
                <max x="{o_ph}" y="{o_pv}" z="{o_ph}"/>
            </Grid>
            <Grid name="agent_far">
                <min x="{ob_nh}" y="{ob_nv}" z="{ob_nh}"/>
                <max x="{ob_ph}" y="{ob_pv}" z="{ob_ph}"/>
            </Grid>
            <Grid name="sensor" absoluteCoords="true">
                <min x="{nh}" y="{nv}" z="{nh}"/>
                <max x="{ph}" y="{pv}" z="{ph}"/>
            </Grid>
        </ObservationFromGrid>
        <DiscreteMovementCommands autoJump="true" autoFall="true"/>
        <MissionQuitCommands/>
      </AgentHandlers>
    </AgentSection>
</Mission>""".format(**TEMPLATE_ARGS)

def create_spec(env, mission):
    commands = []
    for draw in env.commands:
        if isinstance(draw, DrawCuboid):
            s = ('<DrawCuboid x1="%s" y1="%s" z1="%s" x2="%s" y2="%s" z2="%s"' 
                    % draw[:-1])
            s += ' type="%s"' % draw.material.type
            if draw.material.color is not None:
                s += ' colour="%s"' % draw.material.color
            if draw.material.variant is not None:
                s += ' variant="%s"' % draw.material.variant
            s += '/>'
        elif isinstance(draw, DrawBlock):
            assert draw.material.color is None and draw.material.variant is None
            # TODO
            s = '<DrawBlock x="%s" y="%s" z="%s" type="%s"/>' % (draw.x, draw.y,
                    draw.z, draw.material.type)
        commands.append(s)
    draw_commands = "\n".join(commands)
    return SPEC_TEMPLATE.format(draw_commands)


# FEATURES

N_FEATURES = (
        # near observation
        OBS_HSIZE ** 2 * OBS_VSIZE * resources.N_MATERIALS
        # far observation
        + (OBS_HSIZE_BIG / OBS_HBOUNDS_BIG_SCALE) ** 2
            * (OBS_VSIZE_BIG / OBS_VBOUNDS_BIG_SCALE)
            * resources.N_MATERIALS
        # coordinates
        + 5)


# ACTIONS

def _make_discrete_action(base_action):
    command, arg = base_action.split()
    arg = int(arg)
    def _act(host, coord):
        if command == "look":
            np = coord.pitch + arg * 45
            if coord.pitch < -45:
                host.sendCommand("look 1")
            elif coord.pitch > 45:
                host.sendCommand("look -1")
            elif -45 <= np <= 45:
                host.sendCommand(base_action)

        elif command == "move" and arg != 0:
            nx, nz = coord.x, coord.z
            if coord.yaw == 0:
                nz += arg
            elif coord.yaw == 180:
                nz -= arg
            elif coord.yaw == 270:
                nx += arg
            elif coord.yaw == 90:
                nx -= arg

            if (HBOUNDS[0] < nx < HBOUNDS[1] 
                    and HBOUNDS[0] < nz < HBOUNDS[1]):
                host.sendCommand(base_action)

        else:
            host.sendCommand(base_action)

    return _act

assert DISCRETE
STOP = _make_discrete_action("move 0")
FORWARD = _make_discrete_action("move 1")
BACK = _make_discrete_action("move -1")
LEFT = _make_discrete_action("turn -1")
RIGHT = _make_discrete_action("turn 1")
UP = _make_discrete_action("look -1")
DOWN = _make_discrete_action("look 1")
JUMP = _make_discrete_action("jump 1")
ATTACK = _make_discrete_action("attack 1")
USE = _make_discrete_action("use 1")
#ACTIONS = [FORWARD, LEFT, RIGHT, UP, DOWN, ATTACK, USE]
ACTIONS = [FORWARD, LEFT, RIGHT]
N_ACTIONS = len(ACTIONS)


# CLIENTS

class Client(object):
    def __init__(self):
        self.host = MalmoPython.AgentHost()
        self.malmo_mission = None
        self.mission_data = None

    def reset(self, spec):
        if self.malmo_mission is not None:
            self.host.sendCommand("quit")
            while self.state.is_mission_running:
                self._update()

        # TODO
        time.sleep(0.5)

        self.malmo_mission = MalmoPython.MissionSpec(spec, True)
        self.malmo_record = MalmoPython.MissionRecordSpec()
        self.host.startMission(self.malmo_mission, self.malmo_record)
        state = self.host.peekWorldState()
        while not state.has_mission_begun:
            time.sleep(0.1)
            state = self.host.peekWorldState()
        self._update()

    def step(self, i_action):
        action = ACTIONS[i_action]
        action(self.host, self.coordinate)
        time.sleep(TICK/1000. * TICK_SCALE)
        self._update()

    def _update(self):
        self.state = self.host.peekWorldState()
        while (self.state.number_of_observations_since_last_state == 0 
                or all(e.text=='{}' for e in self.state.observations)):
            self.state = self.host.peekWorldState()
            if not self.state.is_mission_running:
                return

        obs = json.loads(self.state.observations[-1].text)

        x = int(np.floor(obs["XPos"]))
        y = int(np.floor(obs["YPos"]))
        z = int(np.floor(obs["ZPos"]))
        pitch = int(obs["Pitch"])
        yaw = int(obs["Yaw"])
        self.coordinate = Coordinate(x, y, z, pitch, yaw)

        n_materials = resources.N_MATERIALS

        # TODO what?
        if "agent_near" not in obs:
            return
        
        near = obs["agent_near"]
        feats_near = np.zeros((len(near), n_materials))
        for i, o in enumerate(near):
            idx = resources.MATERIAL_INDEX[resources.TYPE_TO_NAME[o]] or 0
            feats_near[i, idx] = 1
        feats_near = feats_near.reshape((OBS_VSIZE, OBS_HSIZE, OBS_HSIZE,
                n_materials))
        feats_near = feats_near.transpose((1, 2, 0, 3))

        far = obs["agent_far"] 
        feats_far = np.zeros((len(far), n_materials))
        for i, o in enumerate(far):
            idx = resources.MATERIAL_INDEX[resources.TYPE_TO_NAME[o]] or 0
            feats_far[i, idx] = 1
        feats_far = feats_far.reshape((OBS_VSIZE_BIG, OBS_HSIZE_BIG,
            OBS_HSIZE_BIG, n_materials))
        feats_far = block_reduce(feats_far, (OBS_VBOUNDS_BIG_SCALE,
            OBS_HBOUNDS_BIG_SCALE, OBS_HBOUNDS_BIG_SCALE, 1), func=np.mean)
        feats_far = feats_far.transpose((1, 2, 0, 3))

        assert yaw % 90 == 0
        rot = yaw / 90
        feats_near = np.rot90(feats_near, rot)
        feats_far = np.rot90(feats_far, rot)

        #print feats_near.sum(axis=2)
        #print feats_far.sum(axis=2)

        feats_near = feats_near.ravel()
        feats_far = feats_far.ravel()
        feats_coord = [x / 10., y / 10., z / 10., pitch / 360., yaw / 360.]

        self.features = np.concatenate((feats_near, feats_far, feats_coord))

        self.material_counts = np.zeros(n_materials)
        sensor = obs["sensor"]
        for o in sensor:
            idx = resources.MATERIAL_INDEX[resources.TYPE_TO_NAME[o]] or 0
            self.material_counts[idx] += 1
