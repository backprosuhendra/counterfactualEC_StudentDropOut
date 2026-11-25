import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

rows = []
popsize = [30, 60, 90]
crossOver = [0.5, 0.7, 0.9]
pertubationScale = [0.01, 0.05, 0.10]
for pop in popsize:
    for crossover in crossOver:
        for scale in pertubationScale:
            for seed in range(30):
                # Simulasi agar contoh dapat dijalankan secara mandiri.
                rng = np.random.default_rng([
                    seed, pop, round(crossover * 100), round(scale * 100)
                ])
                hv = rng.normal(loc=9.0, scale=0.2)

                rows.append({
                    "population_size": pop,
                    "crossover": crossover,
                    "perturbation_scale": scale,
                    "seed": seed,
                    "hv": hv,
                })

df = pd.DataFrame(rows)

print(df)
