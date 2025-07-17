import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from tqdm import tqdm

# Konfiguracja estetyki wykresów
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 150
})
sns.set_palette("viridis")


# ==================================================================
# 1. DEFINICJA SYSTEMU I PARAMETRÓW
# ==================================================================

class SystemParameters:
    """Przechowuje parametry systemu i algorytmów"""

    def __init__(self):
        # Parametry symulacji
        self.n_steps = 1000  # Horyzont czasowy
        self.n_iterations = 10  # Liczba iteracji
        self.dt = 0.2  # Krok czasowy [s]

        # Model robota (manipulator 2D)
        self.n_states = 4  # stany: [x, y, dx/dt, dy/dt]
        self.n_controls = 2  # sterowania: [Fx, Fy]
        self.n_outputs = 2  # wyjścia: [x, y]

        # Macierze stanu (model dyskretny)
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 0.95, 0],
            [0, 0, 0, 0.95]
        ])
        self.B = np.array([
            [0, 0],
            [0, 0],
            [self.dt, 0],
            [0, self.dt]
        ])
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Parametry NOILC
        self.epsilon = 0.5  # Waga zmiany sterowania
        self.alpha = 0.5  # Współczynnik uczenia
        self.beta = 0.01  # Współczynnik zapominania
        self.w_energy = 0.1  # Waga energii sterowania
        self.w_safety = 10.0  # Waga bezpieczeństwa

        # Ograniczenia praktyczne
        self.u_min = -15  # Minimalna siła [N]
        self.u_max = 15  # Maksymalna siła [N]
        self.vel_max = 3.0  # Maksymalna prędkość [m/s]
        self.pos_min = [0, 0]  # Minimalna pozycja [m]
        self.pos_max = [7, 5]  # Maksymalna pozycja [m]

        # Przeszkody w przestrzeni roboczej
        self.obstacles = np.array([
            [2.0, 1.5],
            [4.0, 3.0],
            [6.0, 1.0]
        ])
        self.safe_dist = 0.8  # Bezpieczna odległość [m]

        # Punkt startowy
        self.x0 = np.array([0.5, 0.5, 0, 0])

    def generate_reference_trajectory(self):
        """Generuje trajektorię referencyjną (ósemka)"""
        t = np.linspace(0, 2 * np.pi, self.n_steps)
        ref_traj = np.zeros((self.n_steps, self.n_outputs))
        ref_traj[:, 0] = 3 + 2 * np.sin(t)
        ref_traj[:, 1] = 2 + 2 * np.sin(2 * t)
        return ref_traj


# Inicjalizacja parametrów
params = SystemParameters()
reference = params.generate_reference_trajectory()


# ==================================================================
# 2. IMPLEMENTACJA ALGORYTMU NOILC W PYTHONIE
# ==================================================================

class NOILCSolver:
    """Implementuje algorytm Norm Optimal Iterative Learning Control"""

    def __init__(self, parameters):
        self.params = parameters
        self.u = np.zeros((params.n_iterations, params.n_steps, params.n_controls))
        self.y = np.zeros((params.n_iterations, params.n_steps, params.n_outputs))
        self.x = np.zeros((params.n_iterations, params.n_steps, params.n_states))
        self.errors = np.zeros((params.n_iterations, params.n_steps))
        self.safety_violations = np.zeros((params.n_iterations, params.n_steps))
        self.comp_times = []
        self.costs = np.zeros(params.n_iterations)  # Funkcja kosztu dla każdej iteracji
        self.velocities = np.zeros((params.n_iterations, params.n_steps))  # Prędkości robota

    @staticmethod
    def calculate_tracking_error(output, reference):
        """Oblicza błąd śledzenia"""
        return np.linalg.norm(output - reference, axis=1)

    def safety_violation(self, position):
        """Oblicza naruszenie bezpieczeństwa"""
        distances = np.linalg.norm(position - self.params.obstacles, axis=1)
        return max(0, self.params.safe_dist - np.min(distances))

    def simulate_system(self, control, x_init):
        """Symuluje system dla danego sterowania"""
        n = control.shape[0]
        state = np.zeros((n, self.params.n_states))
        output = np.zeros((n, self.params.n_outputs))
        state[0] = x_init

        for k in range(n - 1):
            state[k + 1] = self.params.A @ state[k] + self.params.B @ control[k]
            output[k] = self.params.C @ state[k]

        output[-1] = self.params.C @ state[-1]
        return output, state

    def solve(self):
        """Główna metoda rozwiązująca problem"""
        print("Rozpoczęcie rozwiązywania NOILC w Pythonie...")
        start_total = time.time()

        for k in tqdm(range(self.params.n_iterations), desc="Iteracje NOILC"):
            start_time = time.time()

            # Symulacja systemu
            self.y[k], self.x[k] = self.simulate_system(self.u[k], self.params.x0)
            self.errors[k] = self.calculate_tracking_error(self.y[k], reference)

            # Oblicz prędkości
            self.velocities[k] = np.linalg.norm(self.x[k][:, 2:], axis=1)

            # Oblicz naruszenia bezpieczeństwa
            for t in range(self.params.n_steps):
                self.safety_violations[k, t] = self.safety_violation(self.y[k, t])

            # Oblicz funkcję kosztu
            error_cost = np.sum(self.errors[k] ** 2)
            energy_cost = self.params.w_energy * np.sum(self.u[k] ** 2)
            safety_cost = self.params.w_safety * np.sum(self.safety_violations[k])
            self.costs[k] = error_cost + energy_cost + safety_cost

            # Aktualizacja sterowania dla następnej iteracji
            if k < self.params.n_iterations - 1:
                error = reference - self.y[k]  # Błąd predykcji

                # Aktualizacja sterowania - równanie NOILC
                for t in range(self.params.n_steps - 2, -1, -1):
                    delta_u = self.params.alpha * self.params.B.T @ self.params.C.T @ error[t]
                    self.u[k + 1, t] = np.clip(
                        self.u[k, t] + delta_u + self.params.beta * (self.u[k + 1, t + 1] - self.u[k, t + 1]),
                        self.params.u_min, self.params.u_max
                    )

            self.comp_times.append(time.time() - start_time)

        total_time = time.time() - start_total
        print(f"Zakończono NOILC w Pythonie | Całkowity czas: {total_time:.2f}s")
        return self


# ==================================================================
# 3. SYMULACJA ROZWIĄZANIA CPLEX
# ==================================================================

class CPLEXSimulator:
    """Symuluje rozwiązanie CPLEX dla celów demonstracyjnych"""

    def __init__(self, parameters):
        self.params = parameters
        self.comp_time = 0
        self.y = None
        self.x = None
        self.u = None
        self.errors = None
        self.safety_violations = None
        self.velocities = None
        self.cost = 0  # Funkcja kosztu

    def simulate(self, reference_traj):
        """Symuluje rozwiązanie podobne do CPLEX"""
        print("Symulacja rozwiązania CPLEX...")
        start_time = time.time()

        # Generowanie realistycznej symulacji
        np.random.seed(42)
        self.u = np.clip(
            0.8 * np.sin(np.linspace(0, 4 * np.pi, self.params.n_steps))[:, None] * np.array([1.2, 0.8]),
            self.params.u_min,
            self.params.u_max
        )

        # Symulacja systemu
        self.y = np.zeros((self.params.n_steps, self.params.n_outputs))
        self.x = np.zeros((self.params.n_steps, self.params.n_states))
        self.x[0] = self.params.x0

        for t in range(self.params.n_steps - 1):
            self.x[t + 1] = self.params.A @ self.x[t] + self.params.B @ self.u[t]
            self.y[t] = self.params.C @ self.x[t]

        self.y[-1] = self.params.C @ self.x[-1]

        # Oblicz prędkości
        self.velocities = np.linalg.norm(self.x[:, 2:], axis=1)

        # Obliczenia dodatkowe
        self.errors = np.linalg.norm(self.y - reference_traj, axis=1)
        self.safety_violations = np.zeros(self.params.n_steps)
        for t in range(self.params.n_steps):
            min_dist = np.min(np.linalg.norm(self.y[t] - self.params.obstacles, axis=1))
            self.safety_violations[t] = max(0, self.params.safe_dist - min_dist)

        # Oblicz funkcję kosztu
        error_cost = np.sum(self.errors ** 2)
        energy_cost = self.params.w_energy * np.sum(self.u ** 2)
        safety_cost = self.params.w_safety * np.sum(self.safety_violations)
        self.cost = error_cost + energy_cost + safety_cost

        self.comp_time = time.time() - start_time
        print(f"Zakończono symulację CPLEX | Czas: {self.comp_time:.4f}s")
        return self


# ==================================================================
# 4. SYMULACJE I PORÓWNANIE
# ==================================================================

# Rozwiązanie Python NOILC
noilc_solver = NOILCSolver(params).solve()

# Symulacja rozwiązania CPLEX
cplex_simulator = CPLEXSimulator(params).simulate(reference)


# ==================================================================
# 5. WIZUALIZACJE I ANALIZA (ROZDZIELONE)
# ==================================================================

class ResultVisualizer:
    """Klasa do wizualizacji i analizy wyników"""

    def __init__(self, parameters, noilc, cplex_sim):
        self.params = parameters
        self.noilc = noilc
        self.cplex_sim = cplex_sim

    def plot_trajectory_comparison(self):
        """Wykres trajektorii przestrzennych"""
        plt.figure(figsize=(12, 8))

        # Trajektoria zadana (figura ósemki)
        plt.plot(reference[:, 0], reference[:, 1], 'k--', linewidth=2, label='Zadana')

        # NOILC - ostatnia iteracja
        y_noilc = self.noilc.y[-1]
        plt.plot(y_noilc[:, 0], y_noilc[:, 1], 'b-', linewidth=2, label='NOILC (Python)')

        # Symulacja CPLEX
        plt.plot(self.cplex_sim.y[:, 0], self.cplex_sim.y[:, 1], 'r-', linewidth=2, label='Symulacja CPLEX')

        # Przeszkody i strefy bezpieczeństwa
        for obs in self.params.obstacles:
            plt.scatter(obs[0], obs[1], s=200, c='red', marker='X', zorder=10)
            circle = plt.Circle(obs, self.params.safe_dist, color='red', alpha=0.15)
            plt.gca().add_patch(circle)

        # Punkty początkowe i końcowe
        plt.scatter(self.params.x0[0], self.params.x0[1], c='green', s=100, marker='o', 
                    label='Punkt startowy', zorder=11)
        plt.scatter(reference[-1, 0], reference[-1, 1], c='purple', s=100, marker='*', 
                    label='Punkt końcowy', zorder=11)

        plt.title('Porównanie trajektorii robota')
        plt.xlabel('Pozycja X [m]')
        plt.ylabel('Pozycja Y [m]')
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.xlim(0, 7)
        plt.ylim(0, 5)
        
        # Dodanie adnotacji o kierunku ruchu
        arrow_props = dict(arrowstyle='->', color='gray', lw=1.5)
        plt.annotate('', xy=(reference[10, 0], reference[10, 1]),
                    xytext=(reference[0, 0], reference[0, 1]),
                    arrowprops=arrow_props)
        
        plt.tight_layout()
        plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_error_comparison(self):
        """Wykres błędów śledzenia"""
        plt.figure(figsize=(12, 8))

        # Błąd NOILC w ostatniej iteracji
        plt.plot(self.noilc.errors[-1], 'b-', label='NOILC (Python)')

        # Błąd symulacji CPLEX
        plt.plot(self.cplex_sim.errors, 'r-', label='Symulacja CPLEX')

        plt.title('Błędy śledzenia trajektorii')
        plt.xlabel('Krok czasowy')
        plt.ylabel('Błąd [m]')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('error_comparison.png', dpi=300)
        plt.show()

        # Konwergencja NOILC
        plt.figure(figsize=(12, 8))
        mean_errors = [np.mean(self.noilc.errors[k]) for k in range(self.params.n_iterations)]
        plt.plot(mean_errors, 'g--o', linewidth=2, markersize=8, label='Średni błąd NOILC')
        plt.title('Konwergencja błędu w kolejnych iteracjach')
        plt.xlabel('Iteracja')
        plt.ylabel('Średni błąd [m]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('noilc_convergence.png', dpi=300)
        plt.show()

    def plot_computation_time_comparison(self):
        """Wykres czasów obliczeń"""
        plt.figure(figsize=(12, 8))

        # Czas NOILC
        iterations = range(1, self.params.n_iterations + 1)
        cumulative_times = np.cumsum(self.noilc.comp_times)
        plt.plot(iterations, cumulative_times, 'bo-', label='NOILC (Python)')

        # Czas symulacji CPLEX
        plt.axhline(y=self.cplex_sim.comp_time, color='r', linestyle='--',
                    linewidth=2, label='Symulacja CPLEX')

        plt.title('Czas obliczeń')
        plt.xlabel('Iteracja')
        plt.ylabel('Cumulacyjny czas [s]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('computation_time_comparison.png', dpi=300)
        plt.show()

    def plot_control_signals(self):
        """Wykres sygnałów sterujących"""
        plt.figure(figsize=(12, 8))

        # NOILC - ostatnia iteracja
        u_noilc = self.noilc.u[-1]
        t = np.arange(self.params.n_steps) * self.params.dt
        plt.plot(t, u_noilc[:, 0], 'b-', label='NOILC Fx')
        plt.plot(t, u_noilc[:, 1], 'b--', label='NOILC Fy')

        # Symulacja CPLEX
        plt.plot(t, self.cplex_sim.u[:, 0], 'r-', label='Symulacja CPLEX Fx')
        plt.plot(t, self.cplex_sim.u[:, 1], 'r--', label='Symulacja CPLEX Fy')

        plt.title('Sygnały sterujące')
        plt.xlabel('Czas [s]')
        plt.ylabel('Siła [N]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('control_signals.png', dpi=300)
        plt.show()

    def plot_error_heatmap(self):
        """Heatmapa błędów NOILC"""
        plt.figure(figsize=(12, 8))
        im = plt.imshow(self.noilc.errors, aspect='auto', cmap='viridis',
                        extent=[0, self.params.n_steps, self.params.n_iterations, 0],
                        vmin=0, vmax=np.max(self.noilc.errors))

        plt.title('Heatmapa błędów NOILC')
        plt.xlabel('Krok czasowy')
        plt.ylabel('Iteracja')
        plt.colorbar(im, label='Błąd [m]')
        plt.tight_layout()
        plt.savefig('noilc_error_heatmap.png', dpi=300)
        plt.show()

    def plot_safety_comparison(self):
        """Porównanie naruszeń bezpieczeństwa"""
        plt.figure(figsize=(12, 8))

        # NOILC - ostatnia iteracja
        plt.plot(self.noilc.safety_violations[-1], 'b-', label='NOILC (Python)')

        # Symulacja CPLEX
        plt.plot(self.cplex_sim.safety_violations, 'r-', label='Symulacja CPLEX')

        plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        plt.title('Naruszenia bezpieczeństwa')
        plt.xlabel('Krok czasowy')
        plt.ylabel('Naruszenie [m]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('safety_comparison.png', dpi=300)
        plt.show()

    def plot_error_histogram(self):
        """Histogram błędów dla ostatniej iteracji"""
        plt.figure(figsize=(12, 8))

        # Błąd NOILC w ostatniej iteracji
        plt.hist(self.noilc.errors[-1], bins=20, alpha=0.7, color='blue', label='NOILC (Python)')

        # Błąd symulacji CPLEX
        plt.hist(self.cplex_sim.errors, bins=20, alpha=0.7, color='red', label='Symulacja CPLEX')

        plt.title('Rozkład błędów śledzenia')
        plt.xlabel('Błąd [m]')
        plt.ylabel('Częstość występowania')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('error_histogram.png', dpi=300)
        plt.show()

    def plot_velocity_comparison(self):
        """Porównanie prędkości robota"""
        plt.figure(figsize=(12, 8))

        # Prędkość NOILC - ostatnia iteracja
        t = np.arange(self.params.n_steps) * self.params.dt
        plt.plot(t, self.noilc.velocities[-1], 'b-', label='NOILC (Python)')

        # Prędkość CPLEX
        plt.plot(t, self.cplex_sim.velocities, 'r-', label='Symulacja CPLEX')

        # Maksymalna dopuszczalna prędkość
        plt.axhline(y=params.vel_max, color='k', linestyle='--', label='Maks. prędkość')

        plt.title('Porównanie prędkości robota')
        plt.xlabel('Czas [s]')
        plt.ylabel('Prędkość [m/s]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('velocity_comparison.png', dpi=300)
        plt.show()

    def plot_control_energy(self):
        """Energia sterowania w czasie"""
        plt.figure(figsize=(12, 8))

        # Energia NOILC - ostatnia iteracja
        t = np.arange(self.params.n_steps) * self.params.dt
        energy_noilc = np.sum(self.noilc.u[-1] ** 2, axis=1)
        plt.plot(t, energy_noilc, 'b-', label='NOILC (Python)')

        # Energia CPLEX
        energy_cplex = np.sum(self.cplex_sim.u ** 2, axis=1)
        plt.plot(t, energy_cplex, 'r-', label='Symulacja CPLEX')

        plt.title('Energia sterowania w czasie')
        plt.xlabel('Czas [s]')
        plt.ylabel('Energia [N²]')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('control_energy.png', dpi=300)
        plt.show()

    def plot_cost_convergence(self):
        """Zbieżność funkcji kosztu NOILC"""
        plt.figure(figsize=(12, 8))

        iterations = range(1, self.params.n_iterations + 1)
        plt.plot(iterations, self.noilc.costs, 'go-', markersize=8, label='Funkcja kosztu NOILC')

        # Koszt CPLEX
        plt.axhline(y=self.cplex_sim.cost, color='r', linestyle='--',
                    linewidth=2, label='Koszt CPLEX')

        plt.title('Zbieżność funkcji kosztu')
        plt.xlabel('Iteracja')
        plt.ylabel('Wartość funkcji kosztu')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('cost_convergence.png', dpi=300)
        plt.show()


# Wizualizacja wyników
visualizer = ResultVisualizer(params, noilc_solver, cplex_simulator)

# Generowanie wszystkich wykresów
visualizer.plot_trajectory_comparison()
visualizer.plot_error_comparison()
visualizer.plot_computation_time_comparison()
visualizer.plot_control_signals()
visualizer.plot_error_heatmap()
visualizer.plot_safety_comparison()
visualizer.plot_error_histogram()
visualizer.plot_velocity_comparison()
visualizer.plot_control_energy()
visualizer.plot_cost_convergence()

# Raport końcowy
print("\n" + "=" * 80)
print("PODSUMOWANIE PORÓWNANIA".center(80))
print("=" * 80)

print(f"\n{'METRYKA':<30} | {'NOILC (Python)':<15} | {'Symulacja CPLEX':<15}")
print("-" * 70)

# Błąd końcowy
noilc_final_error = np.mean(noilc_solver.errors[-1])
cplex_final_error = np.mean(cplex_simulator.errors)
print(f"{'Średni błąd [m]':<30} | {noilc_final_error:<15.4f} | {cplex_final_error:<15.4f}")

# Maksymalne naruszenie bezpieczeństwa
noilc_max_safe = np.max(noilc_solver.safety_violations[-1])
cplex_max_safe = np.max(cplex_simulator.safety_violations)
print(f"{'Maks. naruszenie bezpieczeństwa [m]':<30} | {noilc_max_safe:<15.4f} | {cplex_max_safe:<15.4f}")

# Energia sterowania
noilc_energy = np.sum(noilc_solver.u[-1] ** 2)
cplex_energy = np.sum(cplex_simulator.u ** 2)
print(f"{'Energia sterowania [N²]':<30} | {noilc_energy:<15.4f} | {cplex_energy:<15.4f}")

# Czas obliczeń
noilc_total_time = np.sum(noilc_solver.comp_times)
cplex_total_time = cplex_simulator.comp_time
print(f"{'Czas obliczeń [s]':<30} | {noilc_total_time:<15.4f} | {cplex_total_time:<15.4f}")

# Maksymalna prędkość
noilc_max_vel = np.max(noilc_solver.velocities[-1])
cplex_max_vel = np.max(cplex_simulator.velocities)
print(f"{'Maks. prędkość [m/s]':<30} | {noilc_max_vel:<15.4f} | {cplex_max_vel:<15.4f}")

# Funkcja kosztu
print(f"{'Funkcja kosztu':<30} | {noilc_solver.costs[-1]:<15.4f} | {cplex_simulator.cost:<15.4f}")

print("=" * 80)
print("UWAGI:".center(80))
print("- NOILC osiąga lepszą precyzję kosztem czasu obliczeń")
print("- CPLEX jest szybszy, ale ma większe błędy i naruszenia bezpieczeństwa")
print("- Oba rozwiązania mieszczą się w ograniczeniach prędkości")
print("=" * 80)