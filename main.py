from agent import Agent
from config import NUMBER_INTERMEDIATE_STATES
from env import Env
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == "__main__":
    env = Env()
    base_agent = Agent(env, False)
    per_decision_agent = Agent(env, True)
    base_v_over_time = base_agent.n_step_with_importance_ratio()
    per_decision_v_over_time = per_decision_agent.n_step_with_importance_ratio()

    for state, base_color, per_decision_color in zip(
        range(1 + NUMBER_INTERMEDIATE_STATES), mcolors.XKCD_COLORS, mcolors.BASE_COLORS
    ):
        plt.plot(
            [v[state] for v in base_v_over_time],
            label=f"base_state_{state}",
            color=base_color,
        )
        plt.plot(
            [v[state] for v in per_decision_v_over_time],
            label=f"per_decision_state_{state}",
            color=per_decision_color,
        )
    plt.legend(loc="upper right")
    plt.show()

    print()
