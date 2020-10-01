from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from brain import *


class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.index = index
        self.control_spectrum = {'throttle': {0 : -1, 1 : -0.5, 2 : 0, 3 : 0.5, 4 : 1},
                            'pitch': {0 : -1, 1 : -0.5, 2 : 0, 3 : 0.5, 4 : 1},
                            'yaw': {0 : -1, 1 : -0.5, 2 : 0, 3 : 0.5, 4 : 1},
                            'roll': {0 : -1, 1 : 0, 2 : 1},
                            'jump' : {0 : False, 1 : True},
                            'handbrake' : {0 : False, 1 : True}}
        self.brain = None

    def initialize_agent(self):
        self.brain = ActorCritic()


    def translate_controls(self, controls):
        result = SimpleControllerState()
        for control, value in controls.items():
            result[control] = self.control_spectrum[control][value]
        return result 

    def state_to_tensor(self, packet):
        car = packet.game_cars[self.index]
        ball = packet.game_ball

        result = self.physics_to_list(car.physics)
        result.append(car.boost)
        result.extend(self.physics_to_list)

        return tf.Tensor(result)

    def physics_to_list(self, physics):
        result = list()
        for i in physics.values():
            for value in i.values():
                result.append(value)

        return result 

    def drive_around_field(self, packet):
        return 0 if packet.game_cars[self.index].physics.location.y > 0 else 1

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        state_tensor = self.state_to_tensor(packet)
        thoughts = self.brain(state_tensor, drive_around_field)

        return translate_controls(thoughts)

            



