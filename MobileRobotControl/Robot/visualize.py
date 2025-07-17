import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Wczytywanie danych z pliku CSV
data = pd.read_csv('results.csv')
N = len(data)

# Parametry przeszkód (zgodne z modelem OPL)
obstacles = np.array([[2.0,1.5],[4.0,3.0],[6.0,1.0]])
safe_dist = 0.8

# 1. Wykres trajektorii robota i przeszkód
plt.figure(figsize=(12, 8))
plt.plot(data['r_x'], data['r_y'], 'k--', label='Trajektoria zadana')
plt.plot(data['y_x'], data['y_y'], 'b-', label='Trajektoria rzeczywista (iteracja K)')
for obs in obstacles:
    plt.scatter(obs[0], obs[1], s=200, c='red', marker='X', zorder=5, label='Przeszkoda' if obs[0]==obstacles[0][0] else '')
    circle = plt.Circle(obs, safe_dist, color='red', alpha=0.1, label='Strefa bezpieczeństwa' if obs[0]==obstacles[0][0] else '')
    plt.gca().add_patch(circle)
plt.title('Optymalna trajektoria robota i strefy bezpieczeństwa')
plt.xlabel('Pozycja X [m]')
plt.ylabel('Pozycja Y [m]')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.savefig('trajectory.png', dpi=300)

# 2. Wykres wejść sterujących (Fx, Fy) w czasie
plt.figure(figsize=(12, 6))
plt.plot(data['t'], data['u_x'], 'g-', label='Siła Fx')
plt.plot(data['t'], data['u_y'], 'r-', label='Siła Fy')
plt.title('Wejścia sterujące w czasie (iteracja K)')
plt.xlabel('Krok czasowy')
plt.ylabel('Siła [N]')
plt.grid(True)
plt.legend()
plt.savefig('control_inputs.png', dpi=300)

# 3. Wykres składowych błędu śledzenia (e_x, e_y) w czasie
plt.figure(figsize=(12, 6))
plt.plot(data['t'], data['e_x'], 'c-', label='Błąd e_x')
plt.plot(data['t'], data['e_y'], 'm-', label='Błąd e_y')
plt.title('Błąd śledzenia w czasie (iteracja K)')
plt.xlabel('Krok czasowy')
plt.ylabel('Błąd [m]')
plt.grid(True)
plt.legend()
plt.savefig('tracking_error.png', dpi=300)

# 4. Wykres naruszenia bezpieczeństwa w czasie
plt.figure(figsize=(12, 6))
plt.plot(data['t'], data['safety_viol'], 'orange', label='Naruszenie bezpieczeństwa')
plt.axhline(y=0, color='gray', linestyle='--', label='Brak naruszenia')
plt.title('Naruszenie bezpieczeństwa w czasie (iteracja K)')
plt.xlabel('Krok czasowy')
plt.ylabel('Naruszenie [m]')
plt.grid(True)
plt.legend()
plt.savefig('safety_violation.png', dpi=300)

# 5. Wykres składowych prędkości (vx, vy) w czasie
plt.figure(figsize=(12, 6))
plt.plot(data['t'], data['vx'], 'purple', label='Prędkość vx')
plt.plot(data['t'], data['vy'], 'brown', label='Prędkość vy')
plt.axhline(y=vel_max, color='red', linestyle=':', label='Max prędkość')
plt.axhline(y=-vel_max, color='red', linestyle=':')
plt.title('Prędkości robota w czasie (iteracja K)')
plt.xlabel('Krok czasowy')
plt.ylabel('Prędkość [m/s]')
plt.grid(True)
plt.legend()
plt.savefig('velocities.png', dpi=300)

plt.show()
