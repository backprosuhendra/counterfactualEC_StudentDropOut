import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import shap

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from scipy.stats import median_abs_deviation

warnings.filterwarnings("default")
np.random.seed(42)

iris = load_iris(as_frame=True)
CLS = list(iris.target_names)
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, stratify=y, random_state=32)



model = RandomForestClassifier(n_estimators=200,random_state=42).fit(X_train,y_train)
predict = model.predict(X_test)
print(f"accuracy: {accuracy_score(y_test,predict):.3f}\n")
print(classification_report(y_test,predict, target_names=CLS))
ConfusionMatrixDisplay.from_predictions(y_test,predict,display_labels=CLS, cmap="Blues",colorbar=False)
plt.title("Confusion Matrix")
plt.show()

proba_test = model.predict_proba(X_test)
ok = np.where((y_test.values==1)&(proba_test.argmax(1)==1))[0]
i =int(ok[np.abs(proba_test[ok,1]-0.88).argmin()])
x0 = X_test.iloc[i]
pred_c = int(proba_test[i].argmax())
print("flower feature:\n",x0.to_string())
print("\n real label:", CLS[y_test.iloc[i]])
print("probability: ", dict(zip(CLS,proba_test[i].round(3))))
print("model predict: ", CLS[pred_c])

explainer = shap.TreeExplainer(model)
sv = explainer(X_test)
shap.plots.waterfall(sv[i,:,pred_c],show=False)
plt.title(f"why predict'{CLS[pred_c]}'?")
plt.show()

# shap.initjs()
# shap.force_plot(
#     sv.base_values[i,pred_c],
#     sv.values[i,:,pred_c],
#     X_test.iloc[i,:],
#     matplotlib=True,
#     show=False
# )
# plt.title(f"SHAP force plot {CLS[pred_c]}")
# plt.show()


def generate_population(pop_size):
    population = []
    perturbation_scale = 0.1
    for _ in range(pop_size):
        chromosome = x0.copy()
        for j in range(len(chromosome)):
            std = X_train.iloc[:,j].std()
            chromosome.iloc[j] += np.random.normal(
                loc=0,
                scale=perturbation_scale*std
            )
        population.append(chromosome.values)
    return population

# population = pd.DataFrame(generate_population(50), columns=X_train.columns)
# print("original x0")
# print(x0,"\n")
# print("initial population: \n")
# print(population)

popSize = 50
maxGenerations=200
lambda1, lambda2 = 5.0, 1.0

crossoverRate = 0.8
mutationRate = 0.3
mutationScale = 0.2

tournamentSize = 3
elitism = 2

patience = 30
tolerance = 1e-5
rng = np.random.default_rng(42)

featureMin = X_train.min().to_numpy()
featureMax = X_train.max().to_numpy()
xOriginal = x0.to_numpy()
targetClass = 2


assert targetClass != pred_c,(
    "target clas must differ from original prediction"
    )

# probaPopulation = model.predict_proba(population)
# targetLoss =  1 - probaPopulation[:,targetClass]
std = X_train.std().replace(0,1).to_numpy()
# mad = median_abs_deviation(X_train,axis=0,scale=1.0)
# proximity = np.sqrt(np.sum(((population.values-x0.values)/std)**2,axis=1))
# print(proximity)
population = pd.DataFrame(generate_population(popSize), columns=X_train.columns)
population = np.clip(
    population.to_numpy(),
    featureMin,
    featureMax
)
def FitnessFunction(pop):
    df = pd.DataFrame(pop, columns=X_train.columns)
    proba = model.predict_proba(df)
    predictions = model.predict(df)
    target = list(model.classes_).index(targetClass)
    targetLoss = 1-proba[:,target]
    proximity = np.sqrt(np.sum(((x0.values-pop)/std)**2,axis=1))
    # proximity = np.sum(np.abs(x0.values-pop)/np.maximum(mad,1e-8),axis=1)
    fitness = lambda1 * targetLoss + lambda2 * proximity
    validity = predictions == targetClass

    return fitness,targetLoss,proximity, validity

def TournamentSelection(pop, fitness):
    indices = rng.choice(
        len(pop),
        size= tournamentSize,
        replace=False
    )
    winner = indices[np.argmin(fitness[indices])]
    return pop[winner].copy()

def Crossover(parent1, parent2):
    if rng.random()<crossoverRate:
        alpha = rng.random()

        child1 = alpha*parent1+(1-alpha)*parent2
        child2 = alpha*parent2+(1-alpha)*parent1
        return child1,child2
    return parent1.copy(), parent2.copy()

def mutation(chromosome):
    child = chromosome.copy()

    for i in range(len(child)):
        if rng.random()<mutationRate:
            child[i] += rng.normal(
                loc=0,
                scale=mutationScale*std[i]
            )
    return np.clip(child, featureMin,featureMax)

history = []
bestFitness = np.inf
noImprovement = 0

bestValidCf = None
bestValidDistance = np.inf

for generation in range(maxGenerations):
    fitness,targetLoss,proximity,validity = FitnessFunction(population)

    bestIndividu = np.argmin(fitness)
    currentBest = fitness[bestIndividu]

    validIndices = np.where(validity)[0]

    if len(validIndices)>0:
        nearestValidInd = validIndices[
            np.argmin(proximity[validIndices])
        ]

        if proximity[nearestValidInd] < bestValidDistance:
            bestValidDistance = proximity[nearestValidInd]
            bestValidCf = population[nearestValidInd].copy()

    history.append({
        "generation": generation,
        "best_fitness": currentBest,
        "mean_fitness": np.mean(fitness),
        "valid_count": np.sum(validity),
        "best_valid_proximity": (
            bestValidDistance if bestValidCf is not None
            else np.nan
        )
    })

    print(
        f"gen {generation:03d}| "
        f" best fitness: {currentBest:.5f} |"
        f"valid: {np.sum(validity)}/{popSize} |"
        f"Max P(target): {np.max(1-targetLoss):.4f} |"
        f"Min proximity: {np.min(proximity):.4f} |"
        f"best valid proximity: "
        f"{bestValidDistance if bestValidCf is not None else float('nan'):.5f}"
    )

    if bestFitness - currentBest >tolerance:
        bestFitness=currentBest
        noImprovement=0
    else:
        noImprovement+=1

    if bestValidCf is not None and noImprovement >= patience:
        print(f"\n Convergence at generation{generation}")
        break

    elite_indices = np.argsort(fitness)[:elitism]
    newPopulation = [
        population[idx].copy() for idx in elite_indices
    ]

    while len(newPopulation)<popSize:
        parent1 = TournamentSelection(population,fitness)
        parent2 = TournamentSelection(population,fitness)

        child1, child2 = Crossover(parent1,parent2)
        child1 = mutation(child1)
        child2 = mutation(child2)
        newPopulation.append(child1)
        if len(newPopulation)<popSize:
            newPopulation.append(child2)
    population = np.array(newPopulation)

if bestValidCf is not None:
    cfDf = pd.DataFrame(
        [bestValidCf],
        columns=X_train.columns
    )

    print("\n===Best valid counterfactual===")
    print("\nOriginal:")
    print(x0)

    print("\n Counterfactual:")
    print(cfDf.iloc[0])

    print("\n orginanl prediction: ", CLS[model.predict(
        pd.DataFrame([xOriginal], columns=X_train.columns))[0]])

    print("Counterfactual prediction:", CLS[model.predict(cfDf)[0]])
    print("Proximity:", bestValidDistance)
else:
    print("\n no valid counterfactual found")

history_df = pd.DataFrame(history)

objective = np.column_stack((targetLoss,proximity))
isPareto = np.ones(len(objective), dtype=bool)

for i, point in enumerate(objective):
    dominate = np.any(
        np.all(objective<=point, axis=1) &
        np.any(objective<point, axis=1)
        )
    isPareto[i] = not dominate
paretoPoints = objective[isPareto]
paretoPoints = paretoPoints[
    np.argsort(paretoPoints[:,0])
]

plt.figure(figsize=(9,5))
plt.plot(
    history_df["generation"],
    history_df["best_fitness"],
    label = "best fitness"

)

plt.plot(
    history_df["generation"],
    history_df["mean_fitness"],
    label = "mean fitness"
)

plt.xlabel("generation")
plt.ylabel("fitness")
plt.title("GA fitness convergence")
plt.legend()

plt.grid(alpha=0.3)


# plt.figure(figsize=(9,5))
# plt.plot(
#     history_df["generation"],
#     history_df["valid_count"],
#     label="Valid counterfactuals"
# )

# plt.xlabel("generation")
# plt.ylabel("number of valid candidates")
# plt.title("Counterfactual validity during GA")
# plt.legend()



# plt.figure(figsize=(9, 5))

# plt.plot(
#     history_df["generation"],
#     history_df["best_valid_proximity"],
#     label="Best valid proximity"
# )

# plt.xlabel("Generation")
# plt.ylabel("Normalized Euclidean Distance")
# plt.title("Counterfactual Proximity During GA Evolution")
# plt.grid(alpha=0.3)
# plt.legend()
# plt.show()

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,5))
axes[0].plot(np.arange(popSize),targetLoss)
axes[0].set_title("validity-target Loss")
axes[0].set_xlabel("generation")
axes[0].set_ylabel("target loss")

axes[1].plot(np.arange(popSize),proximity)
axes[1].set_title("Proximity")
axes[1].set_xlabel("generation")
axes[1].set_ylabel("distance")
plt.show()

plt.scatter(
    targetLoss,
    proximity,
    alpha=0.4,
    label="All candidates"
)


plt.plot(
    paretoPoints[:, 0],
    paretoPoints[:, 1],
    'r-o',
    linewidth=2,
    label="Pareto front"
)

plt.xlabel("Validity Loss")
plt.ylabel("Proximity Distance")
plt.title("Pareto Front: Validity vs Proximity")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
print(fitness)
print(validity)