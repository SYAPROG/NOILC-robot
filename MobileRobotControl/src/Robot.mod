// Norm Optimal Iterative Learning Control – Extended CPLEX OPL model (.mod)
/*
 * Praktyczny scenariusz: Robot przemysłowy w zakładzie produkcyjnym wykonujący
 * powtarzalne zadanie montażowe w otwartej przestrzeni produkcyjnej z przeszkodami.
 * System uczy się z poprzednich iteracji, aby precyzyjniej śledzić trajektorię
 * mimo zakłóceń (wibracje, zmienne obciążenie) i ograniczeń bezpieczeństwa.
 *
 */

// === Stałe matematyczne ===
// Definicja stałej PI jako stałej zmiennoprzecinkowej.
float PI = 3.1415926535;

// === Parametry systemu ===

int N = 10; // Horyzont czasowy [liczba kroków]. Określa długość trajektorii w dyskretnych punktach czasowych.
            // Zmniejszony dla wydajności obliczeniowej.
int K = 2; // Liczba iteracji uczących [liczba iteracji]. Określa, ile razy algorytm ILC będzie się uczył
            // i poprawiał sterowanie.
int n = 4;  // Wymiar stanu [liczba zmiennych stanu]. Reprezentuje liczbę zmiennych opisujących
            // aktualny stan robota (np. [x, y, prędkość_x, prędkość_y]).
int m = 2;  // Wymiar sterowania [liczba zmiennych sterujących]. Reprezentuje liczbę wejść sterujących
            // (np. [siła_x, siła_y]).
int p = 2;  // Wymiar wyjścia [liczba zmiennych wyjściowych]. Reprezentuje liczbę zmiennych, które
            // są mierzone lub śledzone (np. [pozycja_x, pozycja_y]).

// Macierze systemu (dane z modelu fizycznego)
// Macierze A, B, C opisują dynamikę liniowego systemu dyskretnego: x(t+1) = A*x(t) + B*u(t), y(t) = C*x(t) + d(t).
float A[i in 1..n][j in 1..n] = (i==1 && j==3)? 0.1 : (i==2 && j==4)? 0.1 : (i==3 && j==3)? 0.9 : (i==4 && j==4)? 0.9 : (i==j)? 1 : 0;
// Macierz stanu A: opisuje, jak stan robota zmienia się w czasie bez wpływu sterowania.
// Tutaj: x_dot = 0.1*vx, y_dot = 0.1*vy, vx_next = 0.9*vx, vy_next = 0.9*vy.
// Pozostałe elementy (i==j)? 1 : 0 oznaczają, że pozostałe składowe stanu (x, y) są przenoszone.
float B[i in 1..n][j in 1..m] = (i==3 && j==1)? 0.1 : (i==4 && j==2)? 0.1 : 0;
// Macierz wejścia B: opisuje, jak sterowanie wpływa na zmianę stanu.
// Tutaj: Fx wpływa na vx, Fy wpływa na vy.
float C[i in 1..p][j in 1..n] = (i==j)? 1 : 0;
// Macierz wyjścia C: opisuje, które zmienne stanu są obserwowane jako wyjścia.
// Tutaj: wyjścia to bezpośrednio x i y (pierwsze dwie składowe stanu).

// Trajektoria zadana (ósemka) i zakłócenia
// Deklarowane jako tablice float, ich wartości zostaną zainicjalizowane w bloku execute.
float r[t in 1..N][i in 1..p]; // r[t][i] to zadana wartość wyjścia i w kroku czasowym t.
                               // Będzie generować trajektorię w kształcie ósemki dla pozycji x i y.
float d[t in 1..N][i in 1..p]; // d[t][i] to zakłócenie wpływające na wyjście i w kroku czasowym t.
                               // Będzie symulować zewnętrzne, nieprzewidywalne siły lub błędy pomiarowe.

// Ograniczenia fizyczne 
// Definicje limitów fizycznych i operacyjnych dla robota.
float u_min = -15.0; // Minimalna siła sterująca [N].
float u_max = 15.0;  // Maksymalna siła sterująca [N].
float delta_max = 3.0; // Maksymalna zmiana sterowania między kolejnymi krokami czasowymi [N/s].
                       // Zapewnia płynność sterowania i zapobiega gwałtownym ruchom.
float pos_min[i in 1..p] = 0.0; // Minimalna pozycja [m] dla każdej składowej wyjścia (x, y).
float pos_max[i in 1..p] = 7.5; // Maksymalna pozycja [m] dla każdej składowej wyjścia (x, y).
                                     // Poprawiono inicjalizację na literały zmiennoprzecinkowe.
float vel_max = 2.0; // Maksymalna prędkość [m/s] dla obu osi (vx, vy).
float safe_dist = 0.8; // Bezpieczna odległość [m] od przeszkód.
float safe_dist_sq = safe_dist * safe_dist;


// Przeszkody w przestrzeni roboczej
int num_obstacles = 3; // Liczba przeszkód.
float obstacles[1..num_obstacles][1..p] = [[2.0,1.5],[4.0,3.0],[6.0,1.0]]; // Pozycje przeszkód [m].
                                                                         // Poprawiono inicjalizację na literały zmiennoprzecinkowe.

// Parametry optymalizacji
// Wagi w funkcji celu, kontrolujące priorytety różnych aspektów optymalizacji.
float eps = 0.5;     // Waga kary za zmianę sterowania. Wyższa wartość promuje mniejsze zmiany sterowania między iteracjami.
float w_energy = 0.1; // Waga energii sterowania. Kara za dużą wartość sterowania, promuje efektywność energetyczną.
float w_smooth = 0.2; // Waga płynności sterowania. Kara za gwałtowne zmiany sterowania w czasie.
float w_safety = 5.0; // Kara za naruszenie bezpieczeństwa. Wysoka waga, aby zapewnić unikanie kolizji.

// === Zmienne decyzyjne ===
// Sekcja definiująca zmienne, których wartości są optymalizowane przez solver CPLEX.

range Trials = 0..K;    // Zakres iteracji uczenia, od 0 (początkowe sterowanie) do K (ostatnia iteracja).
range Time = 1..N;      // Zakres kroków czasowych, od 1 do N.
range States = 1..n;    // Zakres indeksów zmiennych stanu.
range Controls = 1..m;  // Zakres indeksów zmiennych sterujących.
range Outputs = 1..p;   // Zakres indeksów zmiennych wyjściowych.

dvar float x[Trials][Time][States]; // Stany robota [jednostki stanu]. x[k][t][i] to wartość i-tej zmiennej stanu
                                    // w kroku czasowym t dla iteracji k.
dvar float u[Trials][Time][Controls]; // Sterowania robota [N]. u[k][t][i] to wartość i-tej zmiennej sterującej
                                     // w kroku czasowym t dla iteracji k.
dvar float y[Trials][Time][Outputs]; // Wyjścia robota [m]. y[k][t][i] to wartość i-tej zmiennej wyjściowej
                                    // (pozycji) w kroku czasowym t dla iteracji k.
dvar float e[Trials][Time][Outputs]; // Błędy śledzenia [m]. e[k][t][i] to różnica między zadaną trajektorią
                                    // a rzeczywistym wyjściem w kroku czasowym t dla iteracji k.
dvar float safety_viol[Trials][Time] in 0..1000; // Naruszenia bezpieczeństwa [m]. safety_viol[k][t] to zmienna pomocnicza,
                                       // która przyjmuje wartość dodatnią, jeśli robot jest bliżej przeszkody
                                       // niż bezpieczna odległość, w kroku czasowym t dla iteracji k.
                                       // Zmienna z dolnym ograniczeniem na 0 (float+).
                                       
dvar float+ dist[Trials][Time][1..num_obstacles];

// === Funkcja celu ===
// Definiuje cel optymalizacji, który ma być minimalizowany. Jest to suma ważonych kar.
minimize sum(k in 1..K) (
    // Błąd śledzenia: Minimalizuje kwadrat błędu między zadaną a rzeczywistą trajektorią.
    // Jest to główny cel ILC – dążenie do precyzyjnego śledzenia trajektorii.
    sum(t in Time, i in Outputs) e[k][t][i]^2
    +
    // Kara za zmianę sterowania między iteracjami: Minimalizuje różnicę między sterowaniem
    // w bieżącej iteracji (k) a sterowaniem z poprzedniej iteracji (k-1).
    // Zapewnia stabilność procesu uczenia i zapobiega oscylacjom.
    eps^2 * sum(t in Time, i in Controls) (u[k][t][i] - u[k-1][t][i])^2
    +
    // Energia sterowania: Minimalizuje całkowitą "energię" zużytą przez sterowanie.
    // Promuje rozwiązania, które wymagają mniejszych sił.
    w_energy * sum(t in Time, i in Controls) u[k][t][i]^2
    +
    // Płynność sterowania w czasie: Minimalizuje gwałtowne zmiany sterowania w kolejnych
    // krokach czasowych w ramach tej samej iteracji.
    // Zapewnia płynne ruchy robota.
    w_smooth * sum(t in 2..N, i in Controls) (u[k][t][i] - u[k][t-1][i])^2
    +
    // Bezpieczeństwo: Kara za naruszenie bezpiecznej odległości od przeszkód.
    // Wysoka waga tej kary zmusza solver do unikania kolizji.
    w_safety * sum(t in Time) safety_viol[k][t]
);

// === Ograniczenia ===
// Sekcja definiująca warunki, które muszą być spełnione przez zmienne decyzyjne.

subject to {
    // Inicjalizacja sterowania dla iteracji 0:
    // Zakłada, że początkowe sterowanie (przed rozpoczęciem uczenia) jest zerowe.
    forall(t in Time, i in Controls)
        u[0][t][i] == 0;

    // Dynamika systemu:
    // Definiuje, jak stan robota ewoluuje w czasie zgodnie z modelem dynamicznym.
    // x(t+1) = A*x(t) + B*u(t)
    forall(k in Trials, t in 1..N-1, i in States)
        x[k][t+1][i] == sum(j in States) A[i][j]*x[k][t][j] + sum(j in Controls) B[i][j]*u[k][t][j];

    // Równania wyjścia:
    // Definiuje, jak wyjścia robota są powiązane ze stanami i zakłóceniami.
    // y(t) = C*x(t) + d(t)
    forall(k in Trials, t in Time, i in Outputs)
        y[k][t][i] == sum(j in States) C[i][j]*x[k][t][j] + d[t][i];

    // Błąd śledzenia:
    // Definiuje błąd jako różnicę między zadaną trajektorią a rzeczywistym wyjściem.
    // e(t) = r(t) - y(t)
    forall(k in Trials, t in Time, i in Outputs)
        e[k][t][i] == r[t][i] - y[k][t][i];

    // Ograniczenia sterowania:
    // Zapewnia, że siły sterujące mieszczą się w dopuszczalnym zakresie.
    // Ogranicza również maksymalną zmianę sterowania między kolejnymi krokami czasowymi.
    forall(k in Trials, t in Time, i in Controls) {
        u_min <= u[k][t][i] <= u_max; // Limity minimalnej i maksymalnej siły.
        if (t > 1) // Ograniczenie zmiany sterowania dotyczy tylko kroków od drugiego wzwyż.
            -delta_max <= u[k][t][i] - u[k][t-1][i] <= delta_max; // Limit zmiany siły.
    }

    // Ograniczenia pozycji:
    // Zapewnia, że robot pozostaje w wyznaczonym obszarze roboczym.
    forall(k in Trials, t in Time, i in Outputs)
        pos_min[i] <= y[k][t][i] <= pos_max[i];

    // Ograniczenia prędkości:
    // Zapewnia, że prędkości robota nie przekraczają bezpiecznych limitów.
    // x[k][t][3] to prędkość w osi X (vx), x[k][t][4] to prędkość w osi Y (vy).
    forall(k in Trials, t in Time) {
        -vel_max <= x[k][t][3] <= vel_max;
        -vel_max <= x[k][t][4] <= vel_max;
    }

    // Bezpieczeństwo (kwadratowe ograniczenie odległości):
    // Wymusza unikanie przeszkód. safety_viol[k][t] jest dodatnie, jeśli robot jest bliżej przeszkody niż safe_dist.
    // Zamiast pierwiastka kwadratowego, używamy kwadratów odległości, aby ograniczenie było kwadratowe i wypukłe.
    // Kwadrat odległości między robotem a przeszkodą: (y[k][t][1] - obstacles[o][1])^2 + (y[k][t][2] - obstacles[o][2])^2
    // To ograniczenie jest równoważne safety_viol[k][t] >= safe_dist - sqrt(distance_sq)
    // jeśli (safe_dist - safety_viol[k][t]) >= 0, czyli safety_viol[k][t] <= safe_dist.
    // Jeśli safety_viol[k][t] > safe_dist, to prawa strona jest ujemna, a lewa (kwadrat odległości) jest zawsze >=0,
    // więc ograniczenie jest spełnione i safety_viol może być 0.
    // --- Bezpieczeństwo: unikanie kolizji (wypukłe SOCP) ---
	forall(k in Trials, t in Time, o in 1..num_obstacles) {
   // (1) Kwadrat odległości ≤ zmienna pomocnicza   — wypukłe
   (y[k][t][1] - obstacles[o][1])^2
 + (y[k][t][2] - obstacles[o][2])^2
   <= dist[k][t][o];

   // (2) Jeśli odległość < safe_dist ⇒ safety_viol dodatni
   dist[k][t][o] + safety_viol[k][t]  >= safe_dist_sq;
}
    

    // Warunki początkowe:
    // Definiuje początkowy stan robota dla każdej iteracji uczenia.
    // Robot zawsze zaczyna z tej samej pozycji i zerową prędkością.
    forall(k in Trials) {
        x[k][1][1] == 0.5; // Początkowa pozycja X [m]
        x[k][1][2] == 0.5; // Początkowa pozycja Y [m]
        x[k][1][3] == 0;   // Początkowa prędkość X [m/s]
        x[k][1][4] == 0;   // Początkowa prędkość Y [m/s]
    }
}


float total_error_sq = sum(t in Time, i in Outputs) (e[K][t][i]^2);
float total_error_rmse = sqrt(total_error_sq / (N * p)); // RMSE (Root Mean Square Error)
float max_error_euclidean = max(t in Time) sqrt(sum(i in Outputs) e[K][t][i]^2);
float max_viol = max(t in Time) safety_viol[K][t];
float energy = sum(t in Time, i in Controls) u[K][t][i]^2;
float smoothness_cost = sum(t in 2..N, i in Controls) (u[K][t][i] - u[K][t-1][i])^2;

// === Analiza wyników ===
// Blok 'execute' jest wykonywany po rozwiązaniu modelu i służy do raportowania wyników
// oraz generowania danych do wizualizacji.
execute {
    // Inicjalizacja tablic r i d w bloku execute przy użyciu Math.sin
    for (var t_idx = 1; t_idx <= N; t_idx++) {
        r[t_idx][1] = 3 + 2 * Math.sin(2 * PI * (t_idx - 1) / N);
        r[t_idx][2] = 2 + 2 * Math.sin(4 * PI * (t_idx - 1) / N);
        d[t_idx][1] = 0.1 * Math.sin(10 * PI * (t_idx - 1) / N);
        d[t_idx][2] = 0.1 * Math.sin(10 * PI * (t_idx - 1) / N); // Zakładamy symetryczne zakłócenia dla Y
    }

    writeln("=== WYNIKI OPTYMALIZACJI DLA ROBOTA PRZEMYSŁOWEGO ===");
    writeln("Model: Norm Optimal Iterative Learning Control (ILC)");
    writeln("Scenariusz: Śledzenie trajektorii z unikaniem przeszkód");
    writeln("----------------------------------------------------");

    // Czas obliczeń:
    // Informuje o czasie, jaki solver CPLEX potrzebował na znalezienie rozwiązania.
    writeln("Czas obliczeń CPLEX: ", cplex.getCplexTime(), " sekund");

    // Błędy dla ostatniej iteracji (K):
    // Ocena skuteczności śledzenia trajektorii po wszystkich iteracjach uczenia.
    

    writeln("Błąd śledzenia (iteracja K=", K, "):");
    writeln("  - Suma kwadratów błędów (SSE): ", total_error_sq);
    writeln("  - Pierwiastek średniokwadratowy błędu (RMSE): ", total_error_rmse, " [m]");
    writeln("  - Maksymalny błąd euklidesowy w punkcie czasowym: ", max_error_euclidean, " [m]");
    writeln("Interpretacja: Niższe wartości wskazują na lepsze śledzenie trajektorii. RMSE daje średnią miarę błędu na zmienną wyjściową.");

    // Bezpieczeństwo:
    // Ocena, czy robot naruszył bezpieczną odległość od przeszkód.
    writeln("Maksymalne naruszenie bezpieczeństwa (iteracja K=", K, "): ", max_viol, " [m]");
    writeln("Interpretacja: Wartość > 0 oznacza naruszenie bezpiecznej odległości. Idealnie powinno być 0.");

    // Energia sterowania:
    // Ocena efektywności energetycznej sterowania.
    writeln("Całkowita energia sterowania (iteracja K=", K, "): ", energy, " [N^2]");
    writeln("Interpretacja: Niższa wartość oznacza mniejsze zużycie sił sterujących.");

    // Płynność sterowania:
    // Ocena gwałtowności zmian sterowania.
    writeln("Koszt płynności sterowania (iteracja K=", K, "): ", smoothness_cost, " [N^2/s^2]");
    writeln("Interpretacja: Niższa wartość oznacza bardziej płynne sterowanie.");

    writeln("----------------------------------------------------");
    writeln("Szczegółowe wyniki dla ostatniej iteracji (K=", K, ") zostały wyeksportowane do 'results.csv'.");
    writeln("Skrypt wizualizacyjny 'visualize.py' został wygenerowany. Uruchom go, aby zobaczyć wykresy.");

    // Eksport wyników do CSV
    // Tworzy plik CSV zawierający kluczowe dane z ostatniej iteracji,
    // które będą używane do wizualizacji w Pythonie.
    var output_file = new IloOplOutputFile("results.csv"); // Zmieniona deklaracja
    output_file.write("t,r_x,r_y,y_x,y_y,e_x,e_y,u_x,u_y,vx,vy,safety_viol"); // Nagłówki kolumn
    for(t in Time) {
        output_file.write("\n",t,",",r[t][1],",",r[t][2],","); // Czas, zadana pozycja (x,y)
        output_file.write(y[K][t][1],",",y[K][t][2],",");     // Rzeczywista pozycja (x,y)
        output_file.write(e[K][t][1],",",e[K][t][2],",");     // Błąd (x,y)
        output_file.write(u[K][t][1],",",u[K][t][2],",");     // Sterowanie (Fx, Fy)
        output_file.write(x[K][t][3],",",x[K][t][4],",");     // Prędkości (vx, vy)
        output_file.write(safety_viol[K][t]);                 // Naruszenie bezpieczeństwa
    }
    output_file.close();

    // Generowanie skryptu do wizualizacji w Pythonie
    // Tworzy plik Python, który po uruchomieniu wczyta dane z 'results.csv'
    // i wygeneruje serię wykresów.
    var python_script_file = new IloOplOutputFile("visualize.py"); // Zmieniona deklaracja
    python_script_file.writeln("import matplotlib.pyplot as plt");
    python_script_file.writeln("import numpy as np");
    python_script_file.writeln("import pandas as pd"); // Dodano pandas dla łatwiejszego wczytywania danych

    python_script_file.writeln("\n# Wczytywanie danych z pliku CSV");
    python_script_file.writeln("data = pd.read_csv('results.csv')");
    python_script_file.writeln("N = len(data)"); // Liczba kroków czasowych

    python_script_file.writeln("\n# Parametry przeszkód (zgodne z modelem OPL)");
    python_script_file.writeln("obstacles = np.array([[2.0,1.5],[4.0,3.0],[6.0,1.0]])");
    python_script_file.writeln("safe_dist = 0.8");

    python_script_file.writeln("\n# 1. Wykres trajektorii robota i przeszkód");
    python_script_file.writeln("plt.figure(figsize=(12, 8))");
    python_script_file.writeln("plt.plot(data['r_x'], data['r_y'], 'k--', label='Trajektoria zadana')");
    python_script_file.writeln("plt.plot(data['y_x'], data['y_y'], 'b-', label='Trajektoria rzeczywista (iteracja K)')");
    python_script_file.writeln("for obs in obstacles:");
    python_script_file.writeln("    plt.scatter(obs[0], obs[1], s=200, c='red', marker='X', zorder=5, label='Przeszkoda' if obs[0]==obstacles[0][0] else '')");
    python_script_file.writeln("    circle = plt.Circle(obs, safe_dist, color='red', alpha=0.1, label='Strefa bezpieczeństwa' if obs[0]==obstacles[0][0] else '')");
    python_script_file.writeln("    plt.gca().add_patch(circle)");
    python_script_file.writeln("plt.title('Optymalna trajektoria robota i strefy bezpieczeństwa')");
    python_script_file.writeln("plt.xlabel('Pozycja X [m]')");
    python_script_file.writeln("plt.ylabel('Pozycja Y [m]')");
    python_script_file.writeln("plt.grid(True)");
    python_script_file.writeln("plt.axis('equal')");
    python_script_file.writeln("plt.legend()");
    python_script_file.writeln("plt.savefig('trajectory.png', dpi=300)");
    // python_script_file.writeln("plt.show()"); // Zostawiamy show na koniec, aby wszystkie wykresy się pojawiły

    python_script_file.writeln("\n# 2. Wykres wejść sterujących (Fx, Fy) w czasie");
    python_script_file.writeln("plt.figure(figsize=(12, 6))");
    python_script_file.writeln("plt.plot(data['t'], data['u_x'], 'g-', label='Siła Fx')");
    python_script_file.writeln("plt.plot(data['t'], data['u_y'], 'r-', label='Siła Fy')");
    python_script_file.writeln("plt.title('Wejścia sterujące w czasie (iteracja K)')");
    python_script_file.writeln("plt.xlabel('Krok czasowy')");
    python_script_file.writeln("plt.ylabel('Siła [N]')");
    python_script_file.writeln("plt.grid(True)");
    python_script_file.writeln("plt.legend()");
    python_script_file.writeln("plt.savefig('control_inputs.png', dpi=300)");

    python_script_file.writeln("\n# 3. Wykres składowych błędu śledzenia (e_x, e_y) w czasie");
    python_script_file.writeln("plt.figure(figsize=(12, 6))");
    python_script_file.writeln("plt.plot(data['t'], data['e_x'], 'c-', label='Błąd e_x')");
    python_script_file.writeln("plt.plot(data['t'], data['e_y'], 'm-', label='Błąd e_y')");
    python_script_file.writeln("plt.title('Błąd śledzenia w czasie (iteracja K)')");
    python_script_file.writeln("plt.xlabel('Krok czasowy')");
    python_script_file.writeln("plt.ylabel('Błąd [m]')");
    python_script_file.writeln("plt.grid(True)");
    python_script_file.writeln("plt.legend()");
    python_script_file.writeln("plt.savefig('tracking_error.png', dpi=300)");

    python_script_file.writeln("\n# 4. Wykres naruszenia bezpieczeństwa w czasie");
    python_script_file.writeln("plt.figure(figsize=(12, 6))");
    python_script_file.writeln("plt.plot(data['t'], data['safety_viol'], 'orange', label='Naruszenie bezpieczeństwa')");
    python_script_file.writeln("plt.axhline(y=0, color='gray', linestyle='--', label='Brak naruszenia')");
    python_script_file.writeln("plt.title('Naruszenie bezpieczeństwa w czasie (iteracja K)')");
    python_script_file.writeln("plt.xlabel('Krok czasowy')");
    python_script_file.writeln("plt.ylabel('Naruszenie [m]')");
    python_script_file.writeln("plt.grid(True)");
    python_script_file.writeln("plt.legend()");
    python_script_file.writeln("plt.savefig('safety_violation.png', dpi=300)");

    python_script_file.writeln("\n# 5. Wykres składowych prędkości (vx, vy) w czasie");
    python_script_file.writeln("plt.figure(figsize=(12, 6))");
    python_script_file.writeln("plt.plot(data['t'], data['vx'], 'purple', label='Prędkość vx')");
    python_script_file.writeln("plt.plot(data['t'], data['vy'], 'brown', label='Prędkość vy')");
    python_script_file.writeln("plt.axhline(y=vel_max, color='red', linestyle=':', label='Max prędkość')");
    python_script_file.writeln("plt.axhline(y=-vel_max, color='red', linestyle=':')");
    python_script_file.writeln("plt.title('Prędkości robota w czasie (iteracja K)')");
    python_script_file.writeln("plt.xlabel('Krok czasowy')");
    python_script_file.writeln("plt.ylabel('Prędkość [m/s]')");
    python_script_file.writeln("plt.grid(True)");
    python_script_file.writeln("plt.legend()");
    python_script_file.writeln("plt.savefig('velocities.png', dpi=300)");

    python_script_file.writeln("\nplt.show()"); // Wyświetla wszystkie wygenerowane wykresy
    python_script_file.close();
}
