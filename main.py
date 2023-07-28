import numpy as np
import random


# first let's model the grid world and policy

class Environment:
    def __init__(self, cells_per_row=4):
        self.discount = 1
        self.actions = ['up', 'down', 'left', 'right']
        self.grid_size = cells_per_row ** 2
        self.states = range(0, self.grid_size)
        self._cells_per_row = cells_per_row

    def is_terminal_state(self, state):
        return state == 0 or state == self.grid_size - 1

    def tick(self, current_state, action):
        # get next state, considering boundaries
        if self.is_terminal_state(current_state):
            return current_state

        if action == 'up':
            next_state = current_state - self._cells_per_row
            if next_state < 0:
                return current_state
        elif action == 'down':
            next_state = current_state + self._cells_per_row
            if next_state >= self.grid_size:
                return current_state
        elif action == 'left':
            if current_state % self._cells_per_row == 0:
                return current_state
            next_state = current_state - 1
        elif action == 'right':
            if (current_state + 1) % self._cells_per_row == 0:
                return current_state
            next_state = current_state + 1

        return next_state

    def reward(self, current_state):
        if self.is_terminal_state(current_state):
            return 0
        else:
            return -1


class RandomPolicy:
    def __init__(self, environment):
        self.environment = environment

    def action(self, current_state):
        if self.environment.is_terminal_state(current_state):
            return {'stay': 1}
        return {e: 1 / len(self.environment.actions) for e in self.environment.actions}

# tests
e = Environment()
assert(e.tick(0, 'up') == 0)
assert(e.tick(0, 'left') == 0)
assert(e.tick(8, 'left') == 8)
assert(e.tick(11, 'right') == 11)
assert(e.tick(14, 'down') == 14)
assert(e.tick(5, 'up') == 1)
assert(e.tick(5, 'down') == 9)
assert(e.tick(5, 'left') == 4)
assert(e.tick(5, 'right') == 6)

p = RandomPolicy(e)
print(p.action(0))
print(p.action(1))

print("//---policy evaluation ---//")

# policy evaluation should converge
e = Environment()
p = RandomPolicy(e)


def evaluate_policy(policy, iterations):
    values = [0] * e.grid_size
    for i in range(iterations):
        new_values = [0] * e.grid_size
        for state in e.states:
            for (action, probability) in policy.action(state).items():
                next_state = e.tick(state, action)
                next_state_value = values[next_state]
                reward = e.reward(state)
                new_values[state] = new_values[state] + probability * (reward + next_state_value)
        values = new_values
    return values

print([round(e, 0) for e in evaluate_policy(p, 100)]) # this should match values from the slides

print("//---policy iteration using converged values---//")
# policy iteration using converged values (no early stopping)
class GreedyPolicy:
    def __init__(self, environment, random_policy_values):
        self.environment = environment
        self.random_policy_values = random_policy_values

    def action(self, state):
        if self.environment.is_terminal_state(state):
            return {'stay': 1}
        else:
            # take the action that yield most value increase
            choices = {}
            for action in self.environment.actions:
                next_state = self.environment.tick(state, action)
                incremental_value = self.random_policy_values[next_state] - self.random_policy_values[state]
                choices[action] = incremental_value
            choices_sorted = sorted(choices, key=choices.get, reverse=True)
            # cheating here, we know most actions the best policy can take is 2
            fst = choices_sorted[0]
            snd = choices_sorted[1]
            if round(choices[fst], 2) == round(choices[snd], 2):
                return {fst: 0.5, snd: 0.5}
            return {choices_sorted[0]: 1}

values = evaluate_policy(p,100)
gp = GreedyPolicy(e, values)
for s in e.states:
    print(gp.action(s))  # this should match with slides

print("//---stop after iteration 3 would give you optimised policy---//")

# now what if we stop early in random policy evaluation?
# in this example stop after iteration 3 would give you optimised policy

values = evaluate_policy(p, 3)
gp_earlystopping = GreedyPolicy(e, values)
for s in e.states:
    print(gp_earlystopping.action(s))  # this should be same policy with previous cell

print("//--- value for the final optimised policy---//")
print([round(e, 0) for e in evaluate_policy(gp, 100)])


print("//---iteration---//")
#value iteration should give same optimised policy, with less iterations

e = Environment()
values = [0] * e.grid_size
known_states = [0, e.grid_size - 1]

def iteration(values, known_states):
    new_values = values.copy()
    new_known_states = known_states.copy()
    for state in e.states:
        if e.is_terminal_state(state) or state in known_states:
            continue
        v = values[state]
        max_reward = float("-inf")
        updatable = False
        for action in e.actions:
            next_state = e.tick(state, action)
            if next_state in known_states:
                updatable = True
                reward = e.reward(state)
                new_values_states = values[next_state]
                total = reward + e.discount * new_values_states
                if total >= max_reward:
                    max_reward = total
        if updatable:
            new_values[state] = max_reward
            new_known_states.append(state)
    return new_values, new_known_states
i = 0
while len(known_states) < e.grid_size:
    values, known_states = iteration(values, known_states)
    print("=== iteration", i)
    i = i + 1
    print("values:", values)
    print("known states:", known_states)

print("=" * 10)
print("final values")
print(values)   # this should match previous cell's result

