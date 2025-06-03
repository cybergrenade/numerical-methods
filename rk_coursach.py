import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_3(f, t_span, y0, h):
    a, b = t_span
    t = np.arange(a, b + h, h)
    n_steps = len(t)
    n_eq = len(y0)
    y = np.zeros((n_steps, n_eq))
    y[0] = y0
    
    for i in range(n_steps - 1):
        t_n = t[i]
        y_n = y[i]
        
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h*k1/2)
        k3 = f(t_n + h, y_n - h*k1 + 2*h*k2)
        
        y[i+1] = y_n + h*(k1 + 4*k2 + k3)/6
    
    return t, y

def test_system(t, y):
    y1, y2 = y
    denom = np.sqrt(1 + np.exp(2*t))
    dy1_dt = -np.sin(t)/denom + y1*(y1**2 + y2**2 - 1)
    dy2_dt = np.cos(t)/denom + y2*(y1**2 + y2**2 - 1)
    return np.array([dy1_dt, dy2_dt])

def exact_solution(t):
    denom = np.sqrt(1 + np.exp(2*t))
    y1 = np.cos(t)/denom
    y2 = np.sin(t)/denom
    return y1, y2

    
def reactor_system(t, y, Da, k=1, gamma=20000, alpha=0.03, S=0.03, beta=4, theta_C=0, B=13):
    X, Y, theta = y
    
    def a(theta):
        return np.exp(theta / (1 + theta / gamma))
    
    Xa = X * a(theta)
    Ya = Y * a(theta)**k
    
    dX_dt = 1 - X - Da * Xa
    dY_dt = -Y + Da * Xa - Da * S * Ya
    dtheta_dt = -theta + Da * B * Xa - beta * (theta - theta_C) + Da * B * alpha * S * Ya
    
    return np.array([dX_dt, dY_dt, dtheta_dt])

def test_error_analysis():
    t_span = [0, 5]
    y0 = [1/np.sqrt(2), 0]  # y1(0) = cos(0)/sqrt(1+e^0) = 1/sqrt(2), y2(0) = sin(0)/sqrt(1+e^0) = 0
    h_values = np.linspace(0.001, 0.1, 10)
    errors = []
    errors_h3 = []
    
    for h in h_values:
        t, y = runge_kutta_3(test_system, t_span, y0, h)
        y1_exact, y2_exact = exact_solution(t)
        error = np.max(np.sqrt((y[:,0] - y1_exact)**2 + (y[:,1] - y2_exact)**2))
        errors.append(error)
        errors_h3.append(error / h**3)
    

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(h_values, errors, 'o-', label='Максимальная погрешность')
    plt.xlabel('Шаг h')
    plt.ylabel('e')
    plt.title('Зависимость погрешности e от h')
    plt.grid(True)
    plt.legend()
    
    log_h = np.log10(h_values)
    log_error = np.log10(errors)
    alpha_values = []

    for i in range(len(h_values) - 1):
        delta_log_h = log_h[i+1] - log_h[i]
        delta_log_error = log_error[i+1] - log_error[i]
        alpha = delta_log_error / delta_log_h
        alpha_values.append(alpha)
    
    plt.subplot(1, 2, 2)
    plt.plot(log_h, log_error, 'o-', label='log(error) vs log(h)')
    plt.xlabel('log(h)')
    plt.ylabel('log(error)')
    plt.title('Зависимость log(error) от log(h)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('log_error_vs_log_h.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(h_values, errors_h3,  label='e / h^3')
    plt.xlabel('Шаг h')
    plt.ylabel('e / h^3')
    plt.title('Зависимость ошибки/h^3 от h')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 0.1)
    plt.tight_layout()
    plt.savefig('c.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, y[:, 0], label='y1 (численное)')
    plt.plot(t, y1_exact, '--', label='y1 (аналитическое)')
    plt.plot(t, y[:, 1], label='y2 (численное)')
    plt.plot(t, y2_exact, '--', label='y2 (аналитическое)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('Сравнение численного и аналитического решений')
    plt.grid(True)
    plt.savefig('test_system_solutions.png')
    plt.close()
    
    print("Таблица 1: h и максимальная погрешность")
    print("| h      | Max Error      |")
    print("|--------|----------------|")
    for h, error in zip(h_values, errors):
        print(f" {h:.3f}  {error:.2e} ")
        
    print("\nТаблица 2: h, alpha и C(h)")
    print("| h      | alpha   | C(h)    |")
    print("|--------|---------|---------|")
    for i in range(len(h_values) - 1):
        h = h_values[i]
        alpha = alpha_values[i]
        C_h = errors_h3[i]
        print(f" {h:.3f}  {alpha:.3f}  {C_h:.3f} ")
    
    

def solve_reactor():
    t_span = [0, 100]  
    y0 = [0.4, 0.5, 1.3]  # X0, Y0, theta0
    h = 0.01
    Da_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for Da in Da_values:
        t, y = runge_kutta_3(lambda t, y: reactor_system(t, y, Da), t_span, y0, h)
        X, Y, theta = y[:,0], y[:,1], y[:,2]
        
      
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(X, theta)
        plt.xlabel('X')
        plt.ylabel('θ')
        plt.title(f'Фазовая траектория (X, θ), Da = {Da}')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(Y, theta)
        plt.xlabel('Y')
        plt.ylabel('θ')
        plt.title(f'Фазовая траектория (Y, θ), Da = {Da}')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'phase_trajectories_Da_{Da}.png')
        plt.close()
        
       
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, X, label='X(t)')
        plt.xlabel('t')
        plt.ylabel('X')
        plt.title(f'X(t), Da = {Da}')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(t, Y, label='Y(t)')
        plt.xlabel('t')
        plt.ylabel('Y')
        plt.title(f'Y(t), Da = {Da}')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(t, theta, label='θ(t)')
        plt.xlabel('t')
        plt.ylabel('θ')
        plt.title(f'θ(t), Da = {Da}')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'time_series_Da_{Da}.png')
        plt.close()
        


if __name__ == "__main__":
    test_error_analysis()

    solve_reactor()
   
