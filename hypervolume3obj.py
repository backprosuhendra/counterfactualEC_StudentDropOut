import numpy as np
import pandas as pd
from time import perf_counter
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

class StudentProblem(Problem):
    def __init__(self):
            super().__init__(n_var=3, n_obj=3, n_ieq_constr=1, xl=0, xu=10, vtype=int)
        
    def _evaluate(self, X, out, **kwargs):
        score = X @ np.array([5, 3, 2]) / 100
        loss = 1 - score
        proximity = X.sum(axis=1) / 30
        sparsity = np.count_nonzero(X, axis=1)
        out["F"] = np.column_stack([loss, proximity, sparsity])
        out["G"] = (0.5 - score)[:, None]

class StepMutation(Mutation):
    def __init__(self, scale):
        super().__init__(prob=0.1)
        self.scale=scale
        

    def _do(self, problem, X, random_state=None, **kwargs):
        Y=X.copy()
        rows = np.arange(len(Y))
        cols = random_state.integers(problem.n_var,size=len(Y))
        steps = random_state.normal(0,10*self.scale,size=len(Y))
        Y[rows, cols]+= np.rint(steps).astype(int)
        return np.clip(Y,problem.xl,problem.xu).astype(int)

indicator = HV(ref_point=np.array([1.1,1.1,4]))
budget = 240

def simulate(pop_size, crossover_prob, scale, seed):
    algorithm = NSGA2(pop_size=pop_size,
                     sampling=IntegerRandomSampling(),
                     crossover=UniformCrossover(prob=crossover_prob),
                     mutation=StepMutation(scale),
                     eliminate_duplicates=False,
                     )
    result = minimize(StudentProblem(), algorithm, ("n_eval", budget), seed=seed)
    # assert result.algorithm.evaluator.n_eval == budget
    F = result.F if result.F is not None else np.empty((0, 3))
    hv = float(indicator.do(F)) if len(F) else 0.0
    return F, hv

F, hv = simulate(pop_size=24, crossover_prob=0.9, scale=0.10, seed=1)
print("First five distinct objective rows [loss, proximity, sparsity]")
print(np.unique(F, axis=0)[:5].round(3))
print(f"One HV for the entire feasible front: {hv:.4f}")

rng = np.random.default_rng()

n_experiment = 30


for i in range(n_experiment):
     settingParameters = pd.DataFrame({
        "popSize": rng.integers(10,101, size=n_experiment),
        "crossoverProb": rng.uniform(0.1, 0.9, size=n_experiment).round(1),
        "pertubationScale": rng.choice([0.01,0.10,0.50], size=n_experiment)
     })
seeds = [1,2,3,4,5]
rows =[]
fronts = {}
for experiment, params in enumerate(settingParameters.itertuples(index=False),start=1):
     for seed in seeds:
          start = perf_counter()
          F, hv = simulate(
               params.popSize,
               params.crossoverProb,
               params.pertubationScale,
               seed
          )
          fronts[(experiment,seed)]=F
          seconds = perf_counter() - start
          rows.append({
                "experiment": experiment,
                "popSize": params.popSize,
                "crossoverProb": params.crossoverProb,
                "pertubationScale": params.pertubationScale,
                "seed":seed,
                "hypervolume":hv,
                "time_seconds": seconds
            })

result = pd.DataFrame(rows)
print(result.head(10))

best_idx = result.groupby("experiment")["hypervolume"].idxmax()
best_runs = result.loc[best_idx].reset_index(drop=True)

best_idx_time = result.groupby("experiment")["time_seconds"].idxmax()
best_runs = result.loc[best_idx_time].reset_index(drop=True)

print(best_runs[["experiment","seed","hypervolume"]])
print(best_runs[["experiment","seed","time_seconds"]])
# summary = (
#      result.groupby(["experiment","popSize","crossoverProb","pertubationScale"],as_index=False)
#      .agg(mean_hv=("hypervolume","mean"),std_hv=("hypervolume","std")).sort_values("experiment",ascending=True)    
# )
# summary_time = (
#      result.groupby(["experiment"],as_index=False).agg(
#           mean_hv = ("hypervolume","mean"),
#           mean_time= ("time_seconds","mean")
#      )
# )

pareto_front = {}
for run in best_runs.itertuples(index=False):
    experiment_id = run.experiment
    best_seed = int(run.seed)

    pareto_front[experiment_id] = pd.DataFrame(
        fronts[(experiment_id, best_seed)],
        columns=["target_loss", "proximity", "sparsity"]
    )
   

for run in best_runs.itertuples(index=False):
    tabel = pareto_front[run.experiment].copy()
    tabel = tabel.sort_values(["target_loss", "proximity", "sparsity"])
    tabel.insert(0, "solusi", range(1, len(tabel) + 1))

    print(f"\nEksperimen {run.experiment} | Seed {run.seed} | HV {run.hypervolume:.4f}")
    print("-" * 55)
    print(tabel.to_string(
        index=False,
        formatters={
            "target_loss": "{:.2f}".format,
            "proximity": "{:.4f}".format,
            "sparsity": "{:.0f}".format,
        }
    ))



