import torch
import itertools
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === XY Encoding/Decoding ===
def encode_term(cx, ex, cy, ey):
    return (cx << 12) | (ex << 8) | (cy << 4) | ey

def decode_term(encoded):
    return ((encoded >> 12) & 0xF,
            (encoded >> 8) & 0xF,
            (encoded >> 4) & 0xF,
            encoded & 0xF)

def batch_decode_xy(terms):
    terms = torch.tensor(terms, dtype=torch.int16, device=device)
    cx = (terms >> 12) & 0xF
    ex = (terms >> 8)  & 0xF
    cy = (terms >> 4)  & 0xF
    ey = terms & 0xF
    return cx, ex, cy, ey

def evaluate_terms_batch(terms, x_vals, y_vals):
    cx, ex, cy, ey = batch_decode_xy(terms)
    x = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, -1)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device).view(1, -1)
    x_part = cx.unsqueeze(1) * (x ** ex.unsqueeze(1))
    y_part = cy.unsqueeze(1) * (y ** ey.unsqueeze(1))
    return x_part + y_part

def differentiate_term(encoded):
    cx, ex, cy, ey = decode_term(encoded)
    dx = encode_term(cx * ex, max(ex - 1, 0), 0, 0)
    dy = encode_term(0, 0, cy * ey, max(ey - 1, 0))
    return dx, dy

def evaluate_term(encoded, x_vals, y_vals):
    cx, ex, cy, ey = decode_term(encoded)
    x = torch.tensor(x_vals, dtype=torch.float32, device=device)
    y = torch.tensor(y_vals, dtype=torch.float32, device=device)
    return (cx * (x ** ex)) + (cy * (y ** ey))

# === Polar Encoding ===
def encode_polar_term(term_type, coeff, exp):
    return (term_type << 12) | (coeff << 8) | exp

def decode_polar_term(encoded):
    return ((encoded >> 12) & 0b11,
            (encoded >> 8) & 0xF,
            encoded & 0xF)

def batch_decode_polar(terms):
    terms = torch.tensor(terms, dtype=torch.int16, device=device)
    term_type = (terms >> 12) & 0b11
    coeff     = (terms >> 8)  & 0xF
    exp       = terms & 0xF
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
            result[i] = 0
    return result

# === Target Function ===
def target_function(x, y):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return 4 * x**2 + 2 * y**2


def generate_term_pool(max_coeff=4, max_exp=4):
    pool = []
    for cx in range(max_coeff + 1):
        for ex in range(max_exp + 1):
            for cy in range(max_coeff + 1):
                for ey in range(max_exp + 1):
                    pool.append(encode_term(cx, ex, cy, ey))
    return pool

def generate_polar_term_pool(max_coeff=4, max_exp=4):
    pool = []
    for typ in [1, 2, 3]:
        for c in range(max_coeff + 1):
            for e in range(max_exp + 1):
                pool.append(encode_polar_term(typ, c, e))
    return pool

def generate_hybrid_term_pool():
    return [('xy', t) for t in generate_term_pool()] + [('polar', t) for t in generate_polar_term_pool()]

# === Evaluation Logic ===
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
    dx = torch.zeros(len(x_vals), device=device)
    dy = torch.zeros(len(x_vals), device=device)
    for kind, term in terms:
        if kind == 'xy':
            dx_t, dy_t = differentiate_term(term)
            dx += evaluate_term(dx_t, x_vals, y_vals)
            dy += evaluate_term(dy_t, x_vals, y_vals)
        # Polar derivs can be added symbolically or estimated numerically
    return dx, dy

def hybrid_loss(terms, x_vals, y_vals, target_fn, grad_weight=1.0):
    pred = evaluate_hybrid_terms(terms, x_vals, y_vals)
    target = target_fn(torch.tensor(x_vals), torch.tensor(y_vals)).to(device)
    loss = ((pred - target) ** 2).sum()

    dx_pred, dy_pred = evaluate_hybrid_derivatives(terms, x_vals, y_vals)
    dx_target = torch.tensor([8 * x for x in x_vals], device=device)
    dy_target = torch.tensor([4 * y for y in y_vals], device=device)
    grad_loss = ((dx_pred - dx_target)**2 + (dy_pred - dy_target)**2).sum()

    return loss + grad_weight * grad_loss

# === Plotting ===
def plot_surface(terms, fn_label='Symbolic Output'):
    x_grid = torch.linspace(0, 4, 50)
    y_grid = torch.linspace(0, 4, 50)
    Z = np.zeros((50, 50))
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            val = evaluate_hybrid_terms(terms, [x.item()], [y.item()])
            Z[i, j] = val.item()
    plt.imshow(Z, extent=[0, 4, 0, 4], origin='lower', cmap='magma')
    plt.colorbar()
    plt.title(fn_label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# === Main ===
def main():
    x_vals = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
    y_vals = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)

    target_fn = lambda x, y: 4 * x**2 + 2 * y**2

    pool = generate_hybrid_term_pool()
    best_loss = float('inf')
    best_combo = None

    print("Searching combinations...")
    for combo in itertools.combinations(pool, 3):
        loss = hybrid_loss(combo, x_vals, y_vals, target_fn)
        if loss < best_loss:
            best_loss = loss
            best_combo = combo

    print("\nBest combo (loss = {:.4f}):".format(best_loss))
    for kind, term in best_combo:
        if kind == 'xy':
            print(" XY term:", decode_term(term))
        elif kind == 'polar':
            print("Polar term:", decode_polar_term(term))

    plot_surface(best_combo, "Best Symbolic Surface")

if __name__ == "__main__":
    main()
