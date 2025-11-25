pts = [0.1,0.3,0.6,0.8,0.95]
gammas = [0.0, 0.5, 1.0, 2.0, 5.0]

#create a table of all focal weights for each combination of pts and gammas formula (1 - pt)^gamma
print("Focal Weights Table:")
print("pt\t" + "\t".join([f"gamma={g}" for g in gammas]))
for p in pts:
    weights = [(1 - p) ** g for g in gammas]
    weights_str = "\t".join([f"{w:.4f}" for w in weights])
    print(f"{p:.2f}\t{weights_str}")

