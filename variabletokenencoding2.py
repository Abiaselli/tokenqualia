import itertools
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_surface(terms, fn_label='Symbolic Model'):
    x_grid = torch.linspace(0, 4, 50)
    y_grid = torch.linspace(0, 4, 50)
    Z = np.zeros((len(y_grid), len(x_grid)))

    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            val = evaluate_hybrid_terms(terms, [x.item()], [y.item()])
            Z[i, j] = val.item()

    plt.imshow(Z, extent=[0, 4, 0, 4], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(f"{fn_label} Output Surface")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_decode(terms):
    terms = torch.tensor(terms, dtype=torch.int16, device=device)
    cx = (terms >> 12) & 0xF
    ex = (terms >> 8)  & 0xF
    cy = (terms >> 4)  & 0xF
    ey = terms & 0xF
    return cx, ex, cy, ey

def batch_decode_polar(terms):
    terms = torch.tensor(terms, dtype=torch.int16, device=device)
    term_type = (terms >> 12) & 0b11
    coeff     = (terms >> 8)  & 0xF
    exp       = terms & 0xF
    return term_type, coeff, exp


def evaluate_terms_batch(terms, x_vals, y_vals):
    # terms: (N,) ints
    cx, ex, cy, ey = batch_decode(terms)
    x = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, -1)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device).view(1, -1)

    # Shape: (N_terms, N_points)
    x_part = cx.unsqueeze(1) * (x ** ex.unsqueeze(1))
    y_part = cy.unsqueeze(1) * (y ** ey.unsqueeze(1))
    return x_part + y_part  # Total shape: (N_terms, N_points)

def score_combo(terms, x_vals, y_vals, target_values, weight=1.0):
    pred = evaluate_terms_batch(torch.tensor(terms, device=device), x_vals, y_vals).sum(dim=0)
    loss = ((pred - target_values) ** 2).sum()

    # Derivative penalty (can add here if needed)
    return loss.item()

# ========== Symbolic Term Infrastructure ==========
def encode_term(cx, ex, cy, ey):
    return (cx << 12) | (ex << 8) | (cy << 4) | ey

def decode_term(encoded):
    return ((encoded >> 12) & 0xF,
            (encoded >> 8) & 0xF,
            (encoded >> 4) & 0xF,
             encoded & 0xF)

def evaluate_term(encoded, x, y):
    cx, ex, cy, ey = decode_term(encoded)
    return (cx * (x ** ex)) + (cy * (y ** ey))

# ========== Target Function ==========
def target_function(x, y):
    return 4 * x**2 + 2 * y**2

# ========== Symbolic Fitting Core ==========
def generate_term_pool(max_coeff=4, max_exp=4):
    """Generate a list of encoded terms (symbolic 'neurons')"""
    pool = []
    for cx in range(max_coeff + 1):
        for ex in range(max_exp + 1):
            for cy in range(max_coeff + 1):
                for ey in range(max_exp + 1):
                    encoded = encode_term(cx, ex, cy, ey)
                    pool.append(encoded)
    return pool

def generate_polar_term_pool(max_coeff=4, max_exp=4):
    # term_type: 1 = r^n, 2 = sin(r^n), 3 = cos(r^n)
    pool = []
    for term_type in [1, 2, 3]:
        for coeff in range(max_coeff + 1):
            for exp in range(max_exp + 1):
                pool.append(encode_polar_term(term_type, coeff, exp))
    return pool

def polar_target_function(x, y):
    r = torch.sqrt(torch.tensor(x) ** 2 + torch.tensor(y) ** 2)
    return torch.sin(r ** 2)


def evaluate_combination(terms, x_vals, y_vals):
    """Evaluate sum of terms across x,y values"""
    total_loss = 0
    for x, y in zip(x_vals, y_vals):
        pred = sum(evaluate_term(term, x, y) for term in terms)
        target = target_function(x, y)
        total_loss += (pred - target) ** 2
    return total_loss

def search_best_combination(term_pool, x_vals, y_vals, combo_size=2):
    """Brute-force search for best combination of N terms"""
    best_loss = float('inf')
    best_combo = None
    for combo in itertools.combinations(term_pool, combo_size):
        loss = evaluate_combination(combo, x_vals, y_vals)
        if loss < best_loss:
            best_loss = loss
            best_combo = combo
    return best_combo, best_loss

def differentiate_term(encoded):
    cx, ex, cy, ey = decode_term(encoded)
    dx_coeff = cx * ex
    dx_exp   = max(ex - 1, 0)
    dy_coeff = cy * ey
    dy_exp   = max(ey - 1, 0)
    return encode_term(dx_coeff, dx_exp, 0, 0), encode_term(0, 0, dy_coeff, dy_exp)

def evaluate_term(encoded, x, y):
    cx, ex, cy, ey = decode_term(encoded)
    return (cx * (x ** ex)) + (cy * (y ** ey))

def evaluate_derivatives(terms, x_vals, y_vals):
    total_dx, total_dy = [], []
    for x, y in zip(x_vals, y_vals):
        dx = sum(evaluate_term(differentiate_term(term)[0], x, y) for term in terms)
        dy = sum(evaluate_term(differentiate_term(term)[1], x, y) for term in terms)
        total_dx.append(dx)
        total_dy.append(dy)
    return total_dx, total_dy

def target_function(x, y):
    return 4 * x**2 + 2 * y**2

def target_derivatives(x, y):
    dx = 8 * x
    dy = 4 * y
    return dx, dy

def evaluate_combination_with_grad(terms, x_vals, y_vals, weight=1.0):
    loss = 0
    for x, y in zip(x_vals, y_vals):
        pred = sum(evaluate_term(term, x, y) for term in terms)
        target = target_function(x, y)
        loss += (pred - target) ** 2

    pred_dx, pred_dy = evaluate_derivatives(terms, x_vals, y_vals)
    for i in range(len(x_vals)):
        target_dx, target_dy = target_derivatives(x_vals[i], y_vals[i])
        loss += weight * ((pred_dx[i] - target_dx) ** 2 + (pred_dy[i] - target_dy) ** 2)
    return loss

def search_best_combo_with_grad(term_pool, x_vals, y_vals, combo_size=3):
    best_loss = float('inf')
    best_combo = None
    for combo in itertools.combinations(term_pool, combo_size):
        loss = evaluate_combination_with_grad(combo, x_vals, y_vals)
        if loss < best_loss:
            best_loss = loss
            best_combo = combo
    return best_combo, best_loss

def encode_polar_term(term_type, coeff, exp):
    # 2 bits for type, 4 bits for coeff, 4 bits for exp
    return (term_type << 12) | (coeff << 8) | exp

def decode_polar_term(term):
    term_type = (term >> 12) & 0b11
    coeff = (term >> 8) & 0xF
    exp = term & 0xF
    return term_type, coeff, exp

def evaluate_polar_term_batch(terms, x_vals, y_vals):
    term_type, coeff, exp = batch_decode_polar(terms)
    x = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, -1)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device).view(1, -1)
    r = torch.sqrt(x ** 2 + y ** 2)

    result = torch.zeros((len(terms), len(x_vals)), device=device)

    for i in range(len(terms)):
        if term_type[i] == 1:
            result[i] = coeff[i] * (r[0] ** exp[i])
        elif term_type[i] == 2:
            result[i] = coeff[i] * torch.sin(r[0] ** exp[i])
        elif term_type[i] == 3:
            result[i] = coeff[i] * torch.cos(r[0] ** exp[i])
        else:
            result[i] = 0  # Treat '00' as empty or invalid
    return result

def evaluate_hybrid_terms(terms, x_vals, y_vals):
    xy_terms = [t[1] for t in terms if t[0] == 'xy']
    polar_terms = [t[1] for t in terms if t[0] == 'polar']

    preds = torch.zeros((len(x_vals),), device=device)
    if xy_terms:
        preds += evaluate_terms_batch(xy_terms, x_vals, y_vals).sum(dim=0)
    if polar_terms:
        preds += evaluate_polar_term_batch(polar_terms, x_vals, y_vals).sum(dim=0)
    return preds

def evaluate_hybrid_derivatives(terms, x_vals, y_vals):
    dx_total = torch.zeros(len(x_vals), device=device)
    dy_total = torch.zeros(len(x_vals), device=device)

    for kind, term in terms:
        if kind == 'xy':
            dx_term, dy_term = differentiate_term(term)
            dx_total += evaluate_term(dx_term, x_vals, y_vals)
            dy_total += evaluate_term(dy_term, x_vals, y_vals)
        elif kind == 'polar':
            # TODO: symbolic diff for polar functions — for now we'll skip or estimate numerically
            pass

    return dx_total, dy_total

def hybrid_loss(terms, x_vals, y_vals, target_fn, grad_weight=1.0):
    pred = evaluate_hybrid_terms(terms, x_vals, y_vals)
    target = target_fn(x_vals, y_vals).to(device)
    loss = ((pred - target) ** 2).sum()

    dx_pred, dy_pred = evaluate_hybrid_derivatives(terms, x_vals, y_vals)
    dx_target = torch.tensor([8 * x for x in x_vals], device=device)  # ∂/∂x of 4x²
    dy_target = torch.tensor([4 * y for y in y_vals], device=device)  # ∂/∂y of 2y²

    grad_loss = ((dx_pred - dx_target)**2 + (dy_pred - dy_target)**2).sum()
    return loss + grad_weight * grad_loss


# ('xy', encoded_xy_term) or ('polar', encoded_polar_term)
def generate_hybrid_term_pool():
    xy = [('xy', t) for t in generate_term_pool()]
    polar = [('polar', t) for t in generate_polar_term_pool()]
    return xy + polar


x_vals = [1, 2, 3]
y_vals = [1, 2, 3]
target_vals = polar_target_function(torch.tensor(x_vals), torch.tensor(y_vals)).to(device)
polar_pool = generate_polar_term_pool()

# Evaluate polar terms (output shape: [N_terms, N_points])
polar_results = evaluate_polar_term_batch(torch.tensor(polar_pool, device=device), x_vals, y_vals)

# Score combination (e.g., 2-term brute-force)
import itertools
min_loss = float("inf")
best_combo = None
for combo in itertools.combinations(range(len(polar_pool)), 2):
    pred = polar_results[list(combo)].sum(dim=0)
    loss = ((pred - target_vals) ** 2).sum()
    if loss < min_loss:
        min_loss = loss.item()
        best_combo = combo

print("Best polar term combo (loss =", min_loss, "):")
for idx in best_combo:
    print(decode_polar_term(polar_pool[idx]))
