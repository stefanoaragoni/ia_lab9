import gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# Inicializar la Q-table
Q_table = np.random.rand(env.observation_space.n, env.action_space.n) * 0.01

# Definir los parámetros de entrenamiento
gamma = 0.95 # factor de descuento
alpha = 0.8 # tasa de aprendizaje
epsilon = 0.99 # probabilidad de exploración

# Definir el número de episodios y pasos máximos por episodio
num_episodes = 10000
max_steps_per_episode = 100

steps = []

# Entrenar el agente
for episode in range(num_episodes):
    state, probability, *_ = env.reset() # genera un nuevo tablero y devuelve el estado inicial

    for step in range(max_steps_per_episode):

        # Seleccionar una acción con base en la Q-table actual
        if np.random.uniform(0.5, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state, :])

        # Tomar la acción y observar el nuevo estado y la recompensa
        new_state, reward, done, info, *_ = env.step(action)
        
        # Actualizar la Q-table utilizando la fórmula de Q-learning
        value = Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[new_state, :]) - Q_table[state, action])
        Q_table[state, action] = value

        # Actualizar el estado
        state = new_state

        # Si se alcanza el estado final, terminar el episodio
        if done:
            steps.append(step + 1)
            break
    
    # Disminuir la probabilidad de exploración
    epsilon = epsilon * 0.99

    if epsilon < 0.01:
        epsilon = 0.01

print("\nENTRENAMIENTO: Promedio de pasos por episodio: {}".format(sum(steps) / len(steps)))

# Utilizar la Q-table entrenada para jugar el juego
state, probability, *_ = env.reset() # genera un nuevo tablero y devuelve el estado inicial

for step in range(max_steps_per_episode):
    # Seleccionar la acción óptima utilizando la Q-table entrenada
    action = np.argmax(Q_table[state, :])

    # Tomar la acción y observar el nuevo estado y la recompensa
    new_state, reward, done, info, *_ = env.step(action)
    
    # Actualizar el estado
    state = new_state

    # Imprimir el estado actual del juego
    env.render()

    # Si se alcanza el estado final, terminar el episodio
    if done:
        print("\n\nDONE: finalizado después de {} pasos \n".format(step + 1))
        break

env.close()

print(Q_table)
