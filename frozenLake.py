import time
import gym
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------ ENTRENAMIENTO DE AGENTE INTELIGENTE ------------------------------

# Inicializar la Q-table
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
Q_table = np.random.rand(env.observation_space.n, env.action_space.n) * 0.01

# Definir los parámetros de entrenamiento
gamma = 0.95 # factor de descuento
alpha = 0.8 # tasa de aprendizaje
epsilon = 0.99 # probabilidad de exploración

# Definir el número de episodios y pasos máximos por episodio
num_episodes = 1000
steps = []

# Entrenar el agente
i = 0
while i <= num_episodes:
    state, probability, *_ = env.reset() # genera un nuevo tablero y devuelve el estado inicial

    falls = 0

    while True:
        # Seleccionar una acción con base en la Q-table actual
        if np.random.uniform(0.6, 1) < epsilon:
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
            falls += 1
            
        elif done and reward == 1:
            break

    steps.append(falls)
    
    # Disminuir la probabilidad de exploración
    epsilon = epsilon * 0.999

    if epsilon < 0.01:
        epsilon = 0.01

    if i == num_episodes:
        if steps[-1] > steps[0]*0.3:
            num_episodes += 500

    i += 1

# Graficar el número de caídas por episodio
steps = np.log10(steps) # para que se vea mejor la gráfica se aplica logaritmo
plt.bar(np.arange(len(steps)), steps, color='blue', alpha=0.4)
plt.title("Número de caídas por episodio")
plt.xlabel("Episodio")
plt.ylabel("Número de caídas")
plt.show()


# ------------------------------ JUGAR EL JUEGO CON AGENTE INTELIGENTE ------------------------------

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')

state, probability, *_ = env.reset() # genera un nuevo tablero y devuelve el estado inicial
falls = 0

while True:
    # Seleccionar la acción óptima utilizando la Q-table entrenada
    action = np.argmax(Q_table[state, :])

    # Tomar la acción y observar el nuevo estado y la recompensa
    new_state, reward, done, info, *_ = env.step(action)
    
    # Actualizar el estado
    state = new_state

    # Si se alcanza el estado final, terminar el episodio
    if done and reward == 0:
        state, probability, *_ = env.reset()
        falls += 1
        
    elif done and reward == 1:
        print("\n\nDONE: finalizado después {} caidas\n".format(falls))
        break

env.close()

print(Q_table)
