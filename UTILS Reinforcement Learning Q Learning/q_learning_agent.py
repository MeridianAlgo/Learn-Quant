
import numpy as np


class FinancialMarketEnvironment:
    """
    Simulates a stochastic market environment with transaction costs.
    The state space is defined by three factors:
    1. Short term moving average relative to Long term (Trend Indicator)
    2. Current Volatility Level (Low or High)
    3. Current Position (Flat, Long, or Short)
    """
    def __init__(self, steps=1000):
        self.steps = steps
        self.current_step = 0
        self.prices = self._generate_price_path(steps)
        self.position = 0 # 0=Flat, 1=Long, 2=Short
        self.transaction_cost = 0.001

    def _generate_price_path(self, length):
        returns = np.random.normal(0.0005, 0.01, length)
        return np.exp(np.cumsum(returns)) * 100

    def reset(self):
        self.current_step = 50 # Start after enough data for moving averages
        self.position = 0
        return self.get_state()

    def get_state(self):
        self.prices[self.current_step]
        ma_short = np.mean(self.prices[self.current_step-10:self.current_step])
        ma_long = np.mean(self.prices[self.current_step-50:self.current_step])

        trend = "Bullish" if ma_short > ma_long else "Bearish"

        recent_returns = np.diff(self.prices[self.current_step-10:self.current_step]) / self.prices[self.current_step-10:self.current_step-1]
        volatility = "High" if np.std(recent_returns) > 0.01 else "Low"

        position_map = {0: "Flat", 1: "Long", 2: "Short"}
        return f"{trend}_{volatility}_{position_map[self.position]}"

    def step(self, action):
        """
        Actions: 0 = Sell/Short, 1 = Hold/Do Nothing, 2 = Buy/Long
        """
        current_price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]
        price_change = (next_price - current_price) / current_price

        reward = 0
        cost = 0

        # Action Logic
        if action == 2: # Buy
            if self.position != 1:
               cost = self.transaction_cost
            self.position = 1
        elif action == 0: # Sell
            if self.position != 2:
               cost = self.transaction_cost
            self.position = 2
        elif action == 1: # Hold
            pass

        # Reward Calculation
        if self.position == 1:
            reward = price_change - cost
        elif self.position == 2:
            reward = -price_change - cost

        self.current_step += 1
        done = self.current_step >= self.steps - 1

        return self.get_state(), reward, done


class AdvancedQLearningAgent:
    """
    Advanced Tabular Q Learning Implementation with decaying parameters
    and sophisticated tracking mechanisms.
    """
    def __init__(self, action_size=3, learning_rate=0.1, discount_factor=0.99, epsilon=1.0):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def choose_action(self, state, evaluate=False):
        if not evaluate and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.get_q_values(state))

    def learn(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q_values)

        q_values[action] = q_values[action] + self.lr * (target - q_values[action])

    def decay_exploration(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_and_evaluate_agent():
    print("Initializing Quantitative Reinforcement Learning Simulation...")
    agent = AdvancedQLearningAgent()

    episodes = 600

    print("\nTraining Phase Commencing...")
    for episode in range(episodes):
        env = FinancialMarketEnvironment(steps=200)
        state = env.reset()
        total_pnl = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_pnl += reward

        agent.decay_exploration()

        if episode % 100 == 0:
            print(f"Training Episode {episode:4} | PNL: {total_pnl:8.4f} | Epsilon: {agent.epsilon:.4f}")

    print("\nTraining Complete. Agent Knowledge Base (Q Table):")
    for st, q_vals in sorted(agent.q_table.items()):
        print(f"State [{st:<25}] Q Values [Sell, Hold, Buy]: {q_vals.round(4)}")

if __name__ == '__main__':
    train_and_evaluate_agent()
