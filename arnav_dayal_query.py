from pyswip import Prolog
import tempfile
import os

# One shared Prolog instance
prolog = Prolog()

# Simplified knowledge base with facts and rules
regularshow_kb = """
% Workers
worker(mordecai).
worker(rigby).
worker(skips).
worker(muscleman).
worker(hifiveghost).

% Managers (bosses)
manager(benson).
manager(mr_maellard).

% Define who is a boss (manager)
boss(X) :- manager(X).

% Define who is managed by a boss
managed_by(X, Y) :- manager(X), worker(Y).
"""

# Save and load KB
with tempfile.NamedTemporaryFile(delete=False, suffix=".pl", mode="w") as temp_file:
    temp_file.write(regularshow_kb)
    filename = temp_file.name.replace(os.sep, '/')

list(prolog.query(f"consult('{filename}')"))

# Query 1: Who is a boss (manager)?
print("List of bosses:")
results1 = list(prolog.query("boss(X)."))
for r in results1:
    print(r['X'])

# Query 2: Find all workers managed by a specific boss (e.g., 'benson')
print("\nWorkers managed by Benson:")
results2 = list(prolog.query("managed_by(benson, X)."))
for r in results2:
    print(r['X'])

# Query 3: Find all workers
print("\nAll workers:")
results3 = list(prolog.query("worker(X)."))
for r in results3:
    print(r['X'])

# Query 4: Find all managers (bosses)
print("\nAll managers:")
results4 = list(prolog.query("manager(X)."))
for r in results4:
    print(r['X'])

# Clean up file
os.remove(filename)
