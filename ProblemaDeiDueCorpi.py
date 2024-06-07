import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Costanti fisiche
G = 6.67430e-11  # Costante gravitazionale, m^3 kg^-1 s^-2
M1 = 5.972e24    # Massa della Terra, kg
M2 = 7.348e22    # Massa della Luna, kg
r = 3.844e8      # Distanza media tra Terra e Luna, m

# Condizioni iniziali
r1_0 = np.array([0, 0])  # Terra inizialmente all'origine
r2_0 = np.array([r, 0])  # Luna inizialmente a distanza r dalla Terra
v1_0 = np.array([0, 0])  # Terra inizialmente ferma
v2_0 = np.array([0, 1022])  # Velocità iniziale della Luna, m/s

# Funzione per calcolare le accelerazioni
def accelerations(r1, r2):
    r12 = r2 - r1
    dist = np.linalg.norm(r12)
    a1 = G * M2 * r12 / dist**3
    a2 = -G * M1 * r12 / dist**3
    return a1, a2

# Parametri di simulazione
dt = 1800  # Passo temporale, 30 minuti
t_max = 2332800  # Durata della simulazione, in secondi -> 27 giorni
num_steps = int(t_max / dt)

# Inizializzazione degli array per memorizzare le posizioni
r1 = np.zeros((num_steps, 2))
r2 = np.zeros((num_steps, 2))
r1[0, :] = r1_0
r2[0, :] = r2_0

# Inizializzazione delle velocità
v1 = v1_0
v2 = v2_0

# Simulazione usando il metodo di Verlet
for i in range(1, num_steps):
    a1, a2 = accelerations(r1[i-1], r2[i-1])
    v1_half = v1 + 0.5 * a1 * dt
    v2_half = v2 + 0.5 * a2 * dt
    r1[i] = r1[i-1] + v1_half * dt
    r2[i] = r2[i-1] + v2_half * dt
    a1_new, a2_new = accelerations(r1[i], r2[i])
    v1 = v1_half + 0.5 * a1_new * dt
    v2 = v2_half + 0.5 * a2_new * dt

# Creazione dell'animazione
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5 * r, 1.5 * r)
ax.set_ylim(-1.5 * r, 1.5 * r)

earth, = ax.plot([], [], 'bo', markersize=10)
moon, = ax.plot([], [], 'o', color='gray', markersize=5)
earth_orbit, = ax.plot([], [], 'b-', lw=0.5)
moon_orbit, = ax.plot([], [], 'gray', lw=0.5)
Tempo_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    earth.set_data([], [])
    moon.set_data([], [])
    earth_orbit.set_data([], [])
    moon_orbit.set_data([], [])
    Tempo_text.set_text('')
    return earth, moon, earth_orbit, moon_orbit, Tempo_text

def update(frame):
    # Calcolo del frame attuale in un ciclo continuo
    idx = frame % num_steps
    earth.set_data(r1[idx, 0], r1[idx, 1])
    moon.set_data(r2[idx, 0], r2[idx, 1])
    earth_orbit.set_data(r1[:idx, 0], r1[:idx, 1])
    moon_orbit.set_data(r2[:idx, 0], r2[:idx, 1])
    Tempo_text.set_text(f'Tempo: {idx * dt / 3600:.1f} ore')
    return earth, moon, earth_orbit, moon_orbit, Tempo_text

# Creazione dell'oggetto animazione
ani = FuncAnimation(fig, update, frames=num_steps*2, init_func=init, blit=True, interval=30)

# Salvataggio dell'animazione come GIF
writer = PillowWriter(fps=30)
ani.save("two_body_problem.gif", writer=writer)

plt.show()