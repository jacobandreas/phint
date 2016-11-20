import malmo_mission
import malmo_gen
from misc import array

from collections import namedtuple
import json
import MalmoPython
import numpy as np
import skimage
from skimage.measure import block_reduce
import time

DISCRETE = True
DURATION = 50000

HBOUNDS = (-10, 10)
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


TICK = 8
#TICK = 50

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


TARGET = np.asarray([[0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 1, 0, 1, 0],
                     [0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 0]]).ravel()

TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
            <Grid name="agent">
                <min x="{o_nh}" y="{o_nv}" z="{o_nh}"/>
                <max x="{o_ph}" y="{o_pv}" z="{o_ph}"/>
            </Grid>
            <Grid name="agent_big">
                <min x="{ob_nh}" y="{ob_nv}" z="{ob_nh}"/>
                <max x="{ob_ph}" y="{ob_pv}" z="{ob_ph}"/>
            </Grid>
            <Grid name="target" absoluteCoords="true">
                <min x="-2" y="4" z="-2"/>
                <max x="2" y="4" z="2"/>
            </Grid>
        </ObservationFromGrid>
        <DiscreteMovementCommands autoJump="true"/>
        <MissionQuitCommands/>
      </AgentHandlers>
    </AgentSection>
</Mission>""".format(**TEMPLATE_ARGS)

class MalmoTask(object):
    def __init__(self, config, mission):
        self.spec = TEMPLATE.format(mission.spec)
        self.hint = mission.goal
        self.mission = mission

class MalmoWorld(object):
    def __init__(self, config):
        self.config = config
        self.setup_actions()
        self.n_actions = len(self.actions)
        n_kinds = len(malmo_gen.index)
        self.n_features = \
                OBS_HSIZE ** 2 * OBS_VSIZE * n_kinds \
                + (OBS_HSIZE_BIG / OBS_HBOUNDS_BIG_SCALE) ** 2 \
                    * (OBS_VSIZE_BIG / OBS_VBOUNDS_BIG_SCALE) \
                    * n_kinds \
                + 5
        self.host = MalmoPython.AgentHost()
        self.mission = None

    def setup_actions(self):
        if DISCRETE:
            self.stop = self.make_discrete_action("move 0")
            self.forward = self.make_discrete_action("move 1")
            self.back = self.make_discrete_action("move -1")
            self.left = self.make_discrete_action("turn -1")
            self.right = self.make_discrete_action("turn 1")
            self.up = self.make_discrete_action("look -1")
            self.down = self.make_discrete_action("look 1")
            self.jump = self.make_discrete_action("jump 1")
            self.attack = self.make_discrete_action("attack 1")
            self.use = self.make_discrete_action("use 1")

            self.movenorth = self.make_discrete_action("movenorth 1")
            self.moveeast = self.make_discrete_action("moveeast 1")
            self.movesouth = self.make_discrete_action("movesouth 1")
            self.movewest = self.make_discrete_action("movewest 1")
        else:
            assert False
        #self.actions = [self.stop, self.forward, self.left, self.right,
        #        self.down, self.down, self.up, self.use, self.attack]
        self.actions = [self.forward, self.left, self.right]
        #self.actions = [self.movenorth, self.moveeast, self.movesouth, self.movewest]

    def sample_task(self):
        mission = malmo_mission.sample()
        return MalmoTask(self.config, mission)

    def reset(self, tasks):
        assert len(tasks) == 1
        task = tasks[0]
        if self.mission is not None:
            self.host.sendCommand("quit")
            while self.state.is_mission_running:
                self.update_obs(hard=True)

        # TODO wait for client to become free
        time.sleep(0.5)

        self.mission = MalmoPython.MissionSpec(task.spec, True)
        self.mission_record = MalmoPython.MissionRecordSpec()
        self.host.startMission(self.mission, self.mission_record)
        state = self.host.peekWorldState()
        while not state.has_mission_begun:
            time.sleep(0.1)
            state = self.host.peekWorldState()

        self.update_obs(hard=True)
        return self.features

    def update_obs(self, hard=False):
        state = self.host.peekWorldState()
        if hard:
            while state.number_of_observations_since_last_state == 0 or \
                    all(e.text=='{}' for e in state.observations):
                state = self.host.peekWorldState()
                if not state.is_mission_running:
                    self.state = state
                    return
            obs = json.loads(state.observations[-1].text)
        elif state.number_of_observations_since_last_state > 0:
                obs = json.loads(state.observations[-1].text)

        self.state = state
        self.obs = obs
        self.x = int(np.floor(obs["XPos"]))
        self.y = int(np.floor(obs["YPos"]))
        self.z = int(np.floor(obs["ZPos"]))
        self.yaw = int(obs["Yaw"])
        self.pitch = int(obs["Pitch"])

        n_kinds = len(malmo_gen.index)
        grid_feats = np.zeros((len(self.obs["agent"]), n_kinds))
        for i, o in enumerate(self.obs["agent"]):
            idx = malmo_gen.index[o] or 0
            grid_feats[i, idx] = 1
        grid_feats = grid_feats.reshape((OBS_VSIZE, OBS_HSIZE, OBS_HSIZE, n_kinds))
        grid_feats = grid_feats.transpose((1, 2, 0, 3))

        grid_feats_big = np.zeros((len(self.obs["agent_big"]), n_kinds))
        for i, o in enumerate(self.obs["agent_big"]):
            idx = malmo_gen.index[o] or 0
            grid_feats_big[i, idx] = 1
        grid_feats_big = grid_feats_big.reshape((OBS_VSIZE_BIG, OBS_HSIZE_BIG,
            OBS_HSIZE_BIG, n_kinds))
        grid_feats_big = block_reduce(grid_feats_big, (OBS_VBOUNDS_BIG_SCALE,
            OBS_HBOUNDS_BIG_SCALE, OBS_HBOUNDS_BIG_SCALE, 1), func=np.mean)
        grid_feats_big = grid_feats_big.transpose((1, 2, 0, 3))

        assert self.yaw % 90 == 0
        rot = self.yaw / 90
        grid_feats = np.rot90(grid_feats, rot)
        grid_feats_big = np.rot90(grid_feats_big, rot)

        #print grid_feats.sum(axis=2)
        #print grid_feats_big.sum(axis=2)

        grid_feats = grid_feats.ravel()
        grid_feats_big = grid_feats_big.ravel()

        target_feats = np.zeros(len(self.obs["target"]))
        for i, o in enumerate(self.obs["target"]):
            target_feats[i] = o != "air"

        self.features = np.concatenate(
                (grid_feats, 
                 grid_feats_big,
                 [self.x / 10., self.y / 10., self.z / 10., 
                     self.pitch / 360., self.yaw / 360.]))
        #self.features = np.asarray(
        #         [self.x / 10., self.y / 10., self.z / 10., 
        #             self.pitch / 360., self.yaw / 360.])
        #self.features = np.zeros(1)

    def step(self, actions, tasks):
        assert len(actions) == 1
        assert len(tasks) == 1
        action = actions[0]
        task = tasks[0]
        malmo_action = self.actions[action]
        malmo_action()
        #time.sleep(TICK/1000. * 1.5)
        time.sleep(TICK/1000. * 4)
        self.update_obs(hard=True)
        reward = 0
        stop = 0
        if task.mission.test(self):
            reward = 1
            stop = 1
        stop |= not self.state.is_mission_running
        return self.features, reward, stop

    def make_continuous_action(self, field, delta, start, stop):
        assert False

    def make_discrete_action(self, base_action):
        command, arg = base_action.split()
        arg = int(arg)
        def _act():
            if command == "look":
                np = self.pitch + arg * 45
                if self.pitch < -45:
                    self.host.sendCommand("look 1")
                elif self.pitch > 45:
                    self.host.sendCommand("look -1")
                elif -45 <= np <= 45:
                    self.host.sendCommand(base_action)

            elif command == "move" and arg != 0:
                nx, nz = self.x, self.z
                if self.yaw == 0:
                    nz += arg
                elif self.yaw == 180:
                    nz -= arg
                elif self.yaw == 270:
                    nx += arg
                elif self.yaw == 90:
                    nx -= arg

                if HBOUNDS[0] < nx < HBOUNDS[1] and \
                        HBOUNDS[0] < nz < HBOUNDS[1]:
                    #self.x, self.z = nx, nz
                    self.host.sendCommand(base_action)

            else:
                self.host.sendCommand(base_action)

        return _act
