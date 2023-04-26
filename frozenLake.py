import time
import gym
import numpy as np

# ------------------------------ ENTRENAMIENTO DE AGENTE INTELIGENTE ------------------------------

# Inicializar la Q-table
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
Q_table = np.random.rand(env.observation_space.n, env.action_space.n) * 0.01

# Definir los parámetros de entrenamiento
gamma = 0.95 # factor de descuento
alpha = 0.8 # tasa de aprendizaje
epsilon = 0.99 # probabilidad de exploración

# Definir el número de episodios y pasos máximos por episodio
num_episodes = 10000
steps = []

# Entrenar el agente
for episode in range(num_episodes):
    state, probability, *_ = env.reset() # genera un nuevo tablero y devuelve el estado inicial

    contador = 0
    while True:
        contador += 1

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
        if done and reward == 0:
            state, probability, *_ = env.reset()
            
        elif done and reward == 1:
            steps.append(contador)
            break
    
    # Disminuir la probabilidad de exploración
    epsilon = epsilon * 0.99

    if epsilon < 0.01:
        epsilon = 0.01

print("\nENTRENAMIENTO: Promedio de pasos por episodio: {}".format(sum(steps) / len(steps)))


# ------------------------------ JUGAR EL JUEGO CON AGENTE INTELIGENTE ------------------------------

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')

state, probability, *_ = env.reset() # genera un nuevo tablero y devuelve el estado inicial
contador = 0

while True:
    contador += 1

    # Seleccionar la acción óptima utilizando la Q-table entrenada
    action = np.argmax(Q_table[state, :])

    # Tomar la acción y observar el nuevo estado y la recompensa
    new_state, reward, done, info, *_ = env.step(action)
    
    # Actualizar el estado
    state = new_state

    # Si se alcanza el estado final, terminar el episodio
    if done and reward == 0:
        state, probability, *_ = env.reset()
        
    elif done and reward == 1:
        print("\n\nDONE: finalizado después de {} pasos \n".format(contador))
        break

env.close()

print(Q_table)
