import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from config import (
    PROBABILITY_DIFFERENT_TRANSITION_AT_LAST_STATE,
    GAMMA,
    ALPHA,
    NUMBER_OF_EPISODES,
    N_STEP,
)
from env import Action, State, Env


@dataclass
class TimeStep:
    state: State
    next_state: State
    reward: int
    importance_ratio: float


class Agent:
    def __init__(self, env: Env, use_per_decision_importance_ratio: bool):
        self.env = env
        self.use_per_decision_importance_ratio = use_per_decision_importance_ratio
        self.v = np.zeros(
            env.number_intermediate_states + 3
        )  # Initial state & Terminal states & Intermediate states

    def n_step_with_importance_ratio(self):
        states_value = []
        for episode in range(NUMBER_OF_EPISODES):
            T = np.inf
            t = 0
            state = self.env.initial_state()
            time_steps = []
            while t < T + N_STEP:
                if t < T:
                    action, prob = self._behaviour_policy(state)
                    next_state, reward, terminal = self.env.transition(state, action)

                    importance_ratio = self._target_policy_action_prob(action) / prob
                    time_steps += [
                        TimeStep(state, next_state, reward, importance_ratio)
                    ]
                    if terminal:
                        T = t

                teta = t - N_STEP + 1
                if teta >= 0:
                    self._learning_increment(teta, T, time_steps)

                t += 1
                state = next_state
            states_value += [self.v.copy()]
        return states_value

    @staticmethod
    def _target_policy_action_prob(action_behaviour_policy: Action) -> float:
        if action_behaviour_policy == Action(1):
            return 1
        return 0

    def _behaviour_policy(self, state: State) -> Tuple[Action, float]:
        if self.env.state_is_last_before_terminal(state):
            if random.random() < PROBABILITY_DIFFERENT_TRANSITION_AT_LAST_STATE:
                return Action(2), PROBABILITY_DIFFERENT_TRANSITION_AT_LAST_STATE
            return Action(1), 1 - PROBABILITY_DIFFERENT_TRANSITION_AT_LAST_STATE
        return Action(1), 1

    def _learning_increment(self, teta: int, T: int, time_steps: List[TimeStep]):
        used_time_steps = time_steps[teta : min(T, teta + N_STEP) + 1]
        if self.use_per_decision_importance_ratio:
            importance_ratio_factors = self._compute_per_decision_importance_ratios(
                used_time_steps
            )
            self.v[teta] += ALPHA * sum(
                [
                    self._delta_t(
                        used_time_steps.state,
                        used_time_steps.next_state,
                        used_time_steps.reward,
                    )
                    * importance_ratio_factor
                    for used_time_steps, importance_ratio_factor in zip(
                        used_time_steps, importance_ratio_factors
                    )
                ]
            )
        else:
            importance_sampling = np.prod(
                [time_step.importance_ratio for time_step in used_time_steps]
            )
            self.v[teta] += (
                importance_sampling
                * ALPHA
                * sum(
                    [
                        self._delta_t(t_s.state, t_s.next_state, t_s.reward)
                        for t_s in used_time_steps
                    ]
                )
            )

    @staticmethod
    def _compute_per_decision_importance_ratios(time_steps: List[TimeStep]):
        importance_sampling_factors = []
        importance_ratio = 1
        for time_step in time_steps:
            importance_ratio *= time_step.importance_ratio
            importance_sampling_factors += [importance_ratio]
        return importance_sampling_factors

    def _delta_t(self, state: State, next_state: State, reward: int):
        return reward + GAMMA * self.v[next_state] - self.v[state]
