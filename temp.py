# Load and clean pest class names
with open("dataset/classes.txt", "r") as f:
    raw_names = f.readlines()
    names = [line.strip().split(maxsplit=1)[-1] for line in raw_names]

# Print cleaned names
print(f"Cleaned class names ({len(names)}): {names}")
