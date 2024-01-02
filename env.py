from typing import Tuple, Optional

from config import NUMBER_INTERMEDIATE_STATES

State = int
Action = int


class Env:
    def __init__(
        self, number_intermediate_states: Optional[int] = NUMBER_INTERMEDIATE_STATES
    ):
        self.number_intermediate_states = number_intermediate_states

    def transition(self, state: State, action: Action) -> Tuple[State, int, bool]:
        next_state = state + action
        if self.state_is_last_before_terminal(state):
            return next_state, 1, True
        return next_state, 1, False

    @staticmethod
    def initial_state():
        return State(0)

    @staticmethod
    def state_is_last_before_terminal(state: State):
        return state == State(NUMBER_INTERMEDIATE_STATES)
