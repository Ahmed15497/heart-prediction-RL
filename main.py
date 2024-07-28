import pandas as pd
import numpy as np
from scipy import stats
import statistics
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.models import load_model
import gym
import random
from gym import Env, spaces
import math
import time


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')



class Lab(Env):
  def __init__(self, X, y):
    super(Lab, self).__init__()
    # Define a 2-D observation space
    self.X = X.copy()
    self.y = y.copy()

    self.observation_shape = (1, self.X.shape[1])
    # age, trestbps, chol
    self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                    high = np.ones(self.observation_shape),
                                    dtype = np.float16)
    # inc dec hold

    self.action_space = spaces.Discrete(3 * self.X.shape[1])


    self.col_steps = []
    for col in range(X.shape[1]):
      sorted_arr = list(set(np.sort(X[:, col])))
      sorted_arr.sort()
      step = sorted_arr[1] - sorted_arr[0]
      self.col_steps.append(step)

    self.curr_obs = None




  def get_state_dim(self):
    return self.X.shape[1]


  def get_action_dim(self):
    return 3 * self.X.shape[1]



  def _next_observation(self):
    frame = self.X[self.current_step].copy()
    return frame


  def _get_observation(self, step):
    frame = self.X[step].copy()
    return frame

  def reset(self, initial_state=None):
    if initial_state is None:
      # Reset the state of the environment to an initial state
      self.current_step = random.randint(0, self.X.shape[0])
      print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
      print(f"Initial index: {self.current_step} with label {self.y[self.current_step]}")
      self.curr_obs = self._next_observation()
    else:
      self.curr_obs = initial_state


    self.reward = 0
    self.patient = 0
    self.similarity_score = 0
    return self.curr_obs


  def _minkowski_distance(self, vector, array, p):
      """
      Calculate the Minkowski distance between a vector and a NumPy array.

      Parameters:
      vector: numpy array, shape (n,)
          The vector.
      array: numpy array, shape (m, n)
          The NumPy array.
      p: int
          The parameter for the Minkowski distance.

      Returns:
      distance: numpy array, shape (m,)
          The Minkowski distance between the vector and each row of the array.
      """
      # Calculate absolute difference
      abs_diff = np.abs(array - vector)

      # Calculate sum of absolute differences raised to the power of p
      sum_abs_diff_p = np.sum(abs_diff**p, axis=1)

      # Calculate Minkowski distance
      distance = sum_abs_diff_p**(1/p)

      return distance


  def _cosine_distance(self, vector, array):
      """
      Compute the cosine distance between a vector and a 2D NumPy array.

      Parameters:
      vector: numpy array
          The vector.
      array: numpy array, shape (m, n)
          The 2D NumPy array.

      Returns:
      distances: numpy array, shape (m,)
          The cosine distances between the vector and each row of the array.
      """
      # Compute dot product between vector and each row of array
      dot_products = np.dot(array, vector)

      # Compute norms of vector and each row of array
      vector_norm = np.linalg.norm(vector)
      array_norms = np.linalg.norm(array, axis=1)

      # Compute cosine similarities
      similarities = dot_products / (vector_norm * array_norms)

      # Compute cosine distances
      distances = 1 - similarities

      return distances


  def _take_action(self, action):
    feature = math.floor(action/3)
    action = action % 3

    if action == 0:
      # increase
      self.curr_obs[feature] = self.curr_obs[feature]+self.col_steps[feature]
      print(f"Increase Feature: {feature} by value: {self.col_steps[feature]}")

    elif action == 1:
      # decrease
      self.curr_obs[feature] = self.curr_obs[feature]-self.col_steps[feature]
      print(f"Decrease Feature: {feature} by value: {self.col_steps[feature]}")

    else:
      # hold
      print(f"Hold Feature: {feature}")
      pass



    # Get the index of the minimum element
    #distance_vector = self._minkowski_distance(self.curr_obs, self.X, 2)
    ##########################################################################################
    distance_vector = self._cosine_distance(self.curr_obs, self.X)
    self.similarity_score = 1 - distance_vector[np.argmin(distance_vector)]
    ##########################################################################################
    index = np.argmin(distance_vector)
    print(f"Similar Sample: {index}")
    self.patient = self.y[index]
    if self.patient != 1:
      self.reward -= 1
    else:
      self.curr_obs =  self._get_observation(index)


    #print(self.curr_obs)



  def step(self, action):
    # Execute one time step within the environment

    self._take_action(action)


    reward = self.reward
    done = self.patient == 1

    return self.curr_obs, reward, done

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.01, model=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        if model is None:
          self.model = self.build_model()
        else:
          self.model = model

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_model(self):
        return self.model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        random_numbers = random.sample(range(0, len(self.memory)), batch_size)

        for i in random_numbers:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Load the model from file
loaded_model = load_model('model.keras')
env = Lab(X_train, y_train)
state_dim = env.get_state_dim()
action_dim = env.get_action_dim()
agent = DQNAgent(state_dim, action_dim, model=loaded_model)



def RL_predict(sample):
  state = env.reset(sample)
  state = np.reshape(state, [1, state_dim])
  done = False
  trial_index = 0
  reward_curr_eps = []
  while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    reward_curr_eps.append(reward)
    next_state = np.reshape(next_state, [1, state_dim])
    state = next_state
    trial_index = trial_index + 1
    if trial_index > 1:
      break
  return done * env.similarity_score



y_train_pred = []
pred_times = []
for i in range(X_train.shape[0]):
  start = time.time()
  score = RL_predict(X_train[i])
  pred_times.append(time.time() - start)
  print('#############################################')
  print(y_train[i], score)
  print('#############################################')
  y_train_pred.append(score>=0.5)
  #break



mean = statistics.mean(pred_times)
print(f"Mean inference time is {mean} seconds") 