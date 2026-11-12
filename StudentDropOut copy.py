import ssl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import shap
import streamlit as st
import plotly.express as px
import networkx as nx


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors
from scipy.stats import median_abs_deviation
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair


ssl._create_default_https_context = ssl._create_unverified_context
from ucimlrepo import fetch_ucirepo 

encoder = LabelEncoder()
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

y['Target'] = encoder.fit_transform(y['Target'])

for index,label in enumerate(encoder.classes_):
    print(f"{label} = {index}")

st.set_page_config(
    page_title="counterfactual explanation",
    layout="wide",
)
st.title("Counterfactual explanation for student dropout prediction")
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Data","Model","SHAP Values","Counterfactuals","Multi-objective evaluation"])
with tab1:
    st.subheader("Dataset")
    st.dataframe(X.join(y), use_container_width=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=32)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train,y_train)
predict = model.predict(X_test)
colors = ['blue', 'orange', 'green']
CLS = list(encoder.classes_)
with tab2:
    st.subheader("Model performance")
    st.write(f"accuracy: {accuracy_score(y_test,predict):.3f}\n")
    st.subheader("Confusion Matrix")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predict, display_labels=CLS, cmap="Blues", colorbar=False)
    st.pyplot(disp.figure_, use_container_width=False) 
    plt.close(disp.figure_)  
        
# modelSVM = svm.SVC(decision_function_shape='ovo', probability=True).fit(X_train,y_train)
# print(f"accuracy score{accuracy_score(y_test,predict)}")


proba_test = model.predict_proba(X_test)
# ok = np.where((y_test.values==1)&(proba_test.argmax(1)==1))[0]
true_labels = y_test.to_numpy().ravel()

ok = np.flatnonzero(
    (true_labels == 1) &
    (proba_test.argmax(axis=1) == 1)
)

if len(ok) == 0:
    st.warning("Tidak ada instance kelas 1 yang diprediksi dengan benar.")
    st.stop()

i = int(ok[np.abs(proba_test[ok, 1] - 0.88).argmin()])

x0 = X_test.iloc[i]
pred_c = int(proba_test[i].argmax())
print("student feature:\n",x0.to_string())
print("\n real label:", encoder.inverse_transform([y_test.iloc[i]])[0])
print("probability: ", dict(zip(encoder.classes_,proba_test[i].round(3))))
print("model predict: ", encoder.inverse_transform([pred_c])[0])

with tab3:
    st.subheader("SHAP waterfall plot")
    explainer = shap.TreeExplainer(model)
    shapValues = explainer(X_test)
    bestFeatures = np.argsort(np.abs(shapValues.values[i, :, pred_c]))[::-1][:5]
    selected_features = X_test.columns[bestFeatures].tolist()
    shap.plots.waterfall(
    shapValues[i, :, pred_c],
    max_display=X_test.shape[1],
    show=True
    )

    fig = plt.gcf()
    fig.set_size_inches(12, 20)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    

    # axes.set_title(f"Scatter plot based on SHAP Values {X.columns[bestFeatureX]} vs {X.columns[bestFeatureY]}")
    # axes.set_xlabel(f"SHAP Value {X.columns[bestFeatureX]}")
    # axes.set_ylabel(f"SHAP Value {X.columns[bestFeatureY]}")
    # axes.legend()
    # st.pyplot(fig2)
    st.subheader("Selected instance(x0)")
    st.dataframe(pd.DataFrame(x0).T, use_container_width=True)
    st.subheader("Top 5 SHAP Features")

    st.dataframe(
    pd.DataFrame({
        "Feature": selected_features,
        "SHAP Value": shapValues.values[i, bestFeatures, pred_c]
    }),use_container_width=True)

    # X_test_scaled = X_test.copy()
    categorical_features = [ "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
    ]
    binary_features= [
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International",
        "Daytime/evening attendance"
        ]

    ec_features = X.columns.tolist()


    graph_features = X_train.columns.tolist()

    X_train_graph = X_train[graph_features].copy()
    X_test_graph = X_test[graph_features].copy()
    x0_graph = x0[graph_features].to_frame().T

    graph_categorical = [col for col in graph_features if col in categorical_features]
    graph_binary = [col for col in graph_features if col in binary_features]
    graph_numerical = [col for col in graph_features if col not in categorical_features + binary_features]

    transformers = []

    if graph_numerical:
        transformers.append(
            ("num", StandardScaler(), 
             graph_numerical))

    if graph_categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             graph_categorical))

    if graph_binary:
        transformers.append(
        ("bin", "passthrough", graph_binary)
    )

    preprocessing = ColumnTransformer(
        transformers=transformers
    )
    
    X_train_scaled = preprocessing.fit_transform(X_train_graph)
    X_test_scaled = preprocessing.transform(X_test_graph)
    X0_scaled = preprocessing.transform(x0_graph)

    if hasattr(X_train_scaled, "toarray"):
        X_train_scaled = X_train_scaled.toarray()

    if hasattr(X_test_scaled, "toarray"):
        X_test_scaled = X_test_scaled.toarray()

    if hasattr(X0_scaled, "toarray"):
        X0_scaled = X0_scaled.toarray()

    k = 5
    knn = NearestNeighbors(
        n_neighbors=k+1,
        metric='euclidean',
        algorithm='auto'
        )
    knn.fit(X_train_scaled)
    distances, indices = knn.kneighbors(X_train_scaled)

# Calculate average KNN distances for training instances


    distances_train = distances[:, 1:]

    mean_knn_distances = distances_train.mean(axis=1)


    percentile = 90
    tau = np.percentile(mean_knn_distances, percentile)

    st.subheader("KNN Distance Distribution")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(
        mean_knn_distances,
        bins=20,
        color="cornflowerblue",
        edgecolor="white",
        alpha=0.85
    )


    ax.axvline(
        tau,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold (P{percentile}) = {tau:.3f}"
    )

    ax.set_title("Distribution of Average KNN Distances")
    ax.set_xlabel("Average KNN Distance")
    ax.set_ylabel("Number of Training Instances")

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.write(f"Plausibility threshold (τ): {tau:.4f}")

    G = nx.Graph()
    G.add_nodes_from(range(len(X_train_scaled)))
    for node_idx in range(len(X_train_scaled)):
        neighbors_added = 0
        for j, dist in zip(indices[node_idx], distances[node_idx]):
            if node_idx==j:
                continue
            G.add_edge(node_idx, int(j), weight=float(dist))
            neighbors_added += 1
            if neighbors_added == k:
               break
    distances_x0, indices_x0 = knn.kneighbors(X0_scaled, n_neighbors=k)
    G.add_node("x0")
    for j, dist in zip(indices_x0[0], distances_x0[0]):
        G.add_edge("x0", int(j), weight=float(dist))

    st.write("connected components:", nx.number_connected_components(G))

    st.subheader("KNN Graph Visualization")

    pca = PCA(n_components=2, svd_solver="full")

    train_pca = pca.fit_transform(X_train_scaled)
    x0_pca = pca.transform(X0_scaled)

    st.write(
    "Total explained variance:",
    pca.explained_variance_ratio_.sum()
    )
    
    pos = {
        node_idx: train_pca[node_idx]
        for node_idx in range(len(X_train_scaled))
    }
    pos["x0"] = x0_pca[0]

    train_labels = np.asarray(y_train).ravel()

    class_colors = {
        "Dropout": "#2556ce",
        "Enrolled": "#e53e3e",
        "Graduate": "#28a16a",
    }

    fig, ax = plt.subplots(figsize=(14, 8))

   
    normal_edges = [
        (u, v)
        for u, v in G.edges()
        if u != "x0" and v != "x0"
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=normal_edges,
        edge_color="gray",
        width=0.6,
        alpha=0.18,
        ax=ax,
    )

    x0_edges = list(G.edges("x0"))

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=x0_edges,
        edge_color="gold",
        width=2.5,
        alpha=1.0,
        ax=ax,
    )

    for class_id, class_name in enumerate(encoder.classes_):
        mask = train_labels == class_id

        ax.scatter(
            train_pca[mask, 0],
            train_pca[mask, 1],
            color=class_colors.get(class_name, "gray"),
            s=22,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.85,
            label=class_name,
            zorder=3,
        )

    ax.scatter(
        x0_pca[0, 0],
        x0_pca[0, 1],
        marker="*",
        s=300,
        color="#ffd23f",
        edgecolors="black",
        linewidths=1.3,
        label=r"$x_0$",
        zorder=5,
    )

    ax.annotate(
        r"$x_0$",
        xy=x0_pca[0],
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        zorder=6,
    )

    variance = pca.explained_variance_ratio_

    ax.set_xlabel(
        f"PC1 ({variance[0]:.1%} explained variance)"
    )
    ax.set_ylabel(
        f"PC2 ({variance[1]:.1%} explained variance)"
    )

    ax.set_title(
        f"Full-feature KNN Graph | "
        f"{len(graph_features)} original features | k={k}"
    )

    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")

    # NetworkX dapat menyembunyikan tick sumbu.
    ax.set_axis_on()
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    df_x0_edges = pd.DataFrame({
        "Source": ["x0"] * len(indices_x0[0]),
        "Neighbor": indices_x0[0],
        "Weight": distances_x0[0],
    })

    st.subheader(f"Weight x₀ to {len(df_x0_edges)} closest neighbors")

    st.dataframe(
        df_x0_edges.style.format({"Weight": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )

    shap_x0 = shapValues.values[i, :, pred_c]
    abs_shap = np.abs(shap_x0)
    actionable_mask = np.ones(len(ec_features), dtype=bool)
    eplsilon = 1e-6
    weights = actionable_mask * (abs_shap + eplsilon)
    mutation_probabilities = weights / weights.sum()

    st.dataframe(
        pd.DataFrame({
            "Feature": ec_features,
            "SHAP Value": shap_x0,
            "Absolute SHAP Value": abs_shap,
            "Mutation Probability": mutation_probabilities,
        }),
        use_container_width=True,
    )

immutable_features = [
    "Marital Status",
    "Gender",
    "Age at enrollment",
    "Application mode",
    "Course",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]
increase_Feature = [
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)"
]
decrease_Feature = [
    "Curricular units 1st sem (without evaluation)",
    "Curricular units 2nd sem (without evaluation)"
]
categorical_features = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]
binary_features= [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Daytime/evening attendance"
]
actionable_features = np.array([feature not in immutable_features for feature in ec_features], dtype=bool)
increaseFeature = np.array([feature in increase_Feature for feature in ec_features], dtype=bool)
decreaseFeature = np.array([feature in decrease_Feature for feature in ec_features], dtype=bool)
shap_x0 = shapValues.values[i, :, pred_c]
abs_shap = np.abs(shap_x0)
actionable_mask = np.ones(len(ec_features), dtype=bool)
eplsilon = 1e-6
weights = actionable_features * (abs_shap + eplsilon)
mutation_probabilities = weights / weights.sum()
def generate_population(pop_size):
    population = []
    perturbation_scale = 0.1
    for _ in range(pop_size):
        chromosome = x0.copy()
        for j in range(len(chromosome)):
            if chromosome.index[j] not in immutable_features:
                std = X_train.iloc[:,j].std()
                chromosome.iloc[j] += np.random.normal(
                loc=0,
                scale=perturbation_scale*std
            )
        population.append(chromosome.values)
    return population

#NSGA2
class FeatureDomain:
    def __init__(self, X_train,x0,immutable,increase, decrease):
        self.names = X_train.columns.tolist()
        self.x0 = np.asarray(x0, dtype=float).reshape(-1)
        self.n = len(self.names)
        for label, group in [("immutable",immutable),("increase",increase),("decrease",decrease)]:
            unknown = set(group) - set(self.names)
            if unknown:
                raise ValueError(f"Unknown features in {label}: {unknown}")
        self.mutable = np.array([f not in immutable for f in self.names],dtype=bool)
        self.increase = np.array([f in increase for f in self.names],dtype=bool)
        self.decrease = np.array([f in decrease for f in self.names],dtype=bool)
        if np.any(self.increase & self.decrease):
            raise ValueError("Features cannot be both in increase and decrease sets.")
        if np.any((self.increase|self.decrease)&~self.mutable):
            raise ValueError("Features in increase or decrease sets must be mutable.")
        self.cat_names = set(categorical_features) & set(self.names)
        self.binary_names = set(binary_features) & set(self.names)
        self.discrete_names = self.cat_names | self.binary_names
        self.integer_names = {
            f for f in self.names
            if f not in self.discrete_names and pd.api.types.is_integer(X_train[f])
        }
        self.continous_names =(
            set(self.names) - self.discrete_names - self.integer_names
        )
        self.allowed = {
            f:np.sort(X_train[f].dropna().to_numpy(dtype=float).copy())
            for f in self.discrete_names
        }
        self.allowed = {f:np.unique(v) for f, v in self.allowed.items()}
        self.lower= X_train.min().to_numpy(dtype=float)
        self.upper= X_train.max().to_numpy(dtype=float)
        self.std = X_train.std().to_numpy(dtype=float)

        self.lower = np.maximum(self.lower,self)
        self.upper = np.minimum(self.upper,self)
        for j,f in enumerate(self.names):
            if f in self.discrete_names:
               self.allowed[f] = np.unique(np.append(self.allowed[c],self.x0[j]))
        self.allowed_by_index = {}
        for j,f in enumerate(self.names):
            if f in self.discrete_names:
                vals = self.allowed[f]
                if self.increase[j]:
                    vals = vals[vals>=self.x0[j]]
                if self.decrease[j]:
                    vals = vals[vals<=self.x0[j]]
                if not lens (vals):
                    raise ValueError(f"No allowed values for feature {f} after applying increase/decrease constraints.")
                self.allowed_by_index[j] = vals

    def repair(self, X):
        Y = np.asarray(X, dtype=float).copy()
        if Y.ndim == 1:
            Y = Y[None, :]
        if Y.shape[1] != self.n:
            raise ValueError(f"Input array must have {self.n} features, got {Y.shape[1]}.")
        if not np.all(np.isfinite(Y)):
            raise ValueError("Input array contains non-finite values.")
        Y=np.clip(Y, self.lower, self.upper)
        Y[:,~self.mutable]= self.x0[~self.mutable]
        for j, name in enumerate(self.names):
            if not self.mutable[j]:
                continue
            if self.increase[j]:
                Y[:,j] = np.maximum(Y[:,j], self.x0[j])
            if self.decrease[j]:
                Y[:,j] = np.minimum(Y[:,j], self.x0[j])
            if j in self.allowed_by_index:
                choices = self.allowed_by_index[j]
                Y[:,j] = choices[np.abs(Y[:,j,None]-choice[None,:]).argmin(axis=1)]
            elif name in self.integer_names:
                Y[:,j] = np.rint(Y[:,j])
        return Y

    def is_valid(self, X, atol=1e-8):
        Y = np.atleast_2d(np.asarray(X, dtype=float))
        valid = np.all(np.isclose(Y[:, ~self.mutable], self.x0[~self.mutable], atol=atol, rtol=0), axis=1)
        valid &= np.all(Y >= self.lower - atol, axis=1) & np.all(Y <= self.upper + atol, axis=1)
        valid &= np.all(Y[:, self.increase] >= self.x0[self.increase] - atol, axis=1)
        valid &= np.all(Y[:, self.decrease] <= self.x0[self.decrease] + atol, axis=1)
        for j, name in enumerate(self.names):
            if j in self.allowed_by_index:
                valid &= np.isin(Y[:, j], self.allowed_by_index[j])
            elif name in self.integer_names:
                valid &= np.isclose(Y[:, j], np.rint(Y[:, j]), atol=atol, rtol=0)
        return valid

class CostumSampling(Sampling):
    def __init__(self,x0,actionable_mask,feature_std,increase_mask,decrease_mask,perturbation_scale=0.1):
        super().__init__()
        self.x0 = np.asarray(x0, dtype=float)
        self.actionable_mask = np.asarray(actionable_mask,dtype=bool)
        self.feature_std = np.asarray(feature_std, dtype=float)
        self.increase_mask = np.asarray(increase_mask, dtype=bool)
        self.decrease_mask = np.asarray(decrease_mask, dtype=bool)
        self.perturbation_scale = perturbation_scale
        
    def _do(self,problem,n_samples, **kwargs):
        samples = np.tile(self.x0,(n_samples,1))
        for i in range(n_samples):
            for j in range(len(self.x0)):
                if not self.actionable_mask[j]:
                    continue

                std = self.feature_std[j]
                if not np.isfinite(std) or std <= 0:
                    continue
                delta = np.random.normal(
                    loc=0,
                    scale=self.perturbation_scale*std
                )
                if self.increase_mask[j]:
                    delta = abs(delta)
                elif self.decrease_mask[j]:
                    delta = -abs(delta)
                samples[i,j] += delta
                # samples[i,j] += np.random.normal(
                #     loc=0,
                #     scale=self.perturbation_scale*std
                # )
        return samples
class ShapMutation(Mutation):
    def __init__(self, mutation_probabilities, actionable_mask,feature_std,perturbation_scale=0.1):
        super().__init__()
        
        self.mutation_probabilities = np.asarray(mutation_probabilities, dtype=float)
        self.actionable_mask = np.asarray(actionable_mask, dtype=bool)
        self.feature_std = np.asarray(feature_std, dtype=float)
        self.perturbation_scale = perturbation_scale

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i in range(len(Y)):
            j= np.random.choice(
                X.shape[1],
                p=self.mutation_probabilities
            )
            Y[i,j] += np.random.normal(
                loc=0,
                scale=(self.perturbation_scale*self.feature_std[j])
            )
        return Y
    
class DirectionalRepair(Repair):
    def __init__(self,x0,increaseFeature,decreaseFeature):
        super().__init__()
        self.x0 = np.asarray(x0, dtype=float)
        self.increaseFeature = increaseFeature
        self.decreaseFeature = decreaseFeature
        
    def _do(self,problem,X, **kwargs):
        repaired = X.copy()
        for i in self.increaseFeature:
            repaired[:,i] = np.maximum(
                repaired[:,i],
                self.x0[i]
            )
        for i in self.decreaseFeature:
            repaired[:,i] = np.minimum(
                repaired[:,i],
                self.x0[i]
            )
        return repaired

class DomainRepair(Repair):
    def __init__(self,domain):
        super().__init__()
        self.domain = domain

    def _do(self,problem,X,**kwargs):
        return self.domain.repair(X)

class MixedCrossover(Crossover):
    def __init__(self, crossover_probabilities=0.9):
        super().__init__(2,2)
        self.crossover_probabilities = crossover_probabilities

    def _do(self, problem, X, **kwargs):
        n_matings = 
sampling = CostumSampling(
    x0=x0.values,
    actionable_mask=actionable_features,
    feature_std=X_train.std().values,
    increase_mask=increaseFeature,
    decrease_mask=decreaseFeature,
    perturbation_scale=0.1,
   
)
population_ = sampling._do(None, 50)    

with tab4:
    population = generate_population(50)
    st.dataframe(pd.DataFrame(population, columns=X.columns),
                 use_container_width=True,
                )
    print ("population:\n", pd.DataFrame(population, columns=X.columns))
    st.write(f"actionable features: {actionable_features}")

    st.subheader("population from custom sampling")
    st.dataframe(pd.DataFrame(population_, columns=X.columns),
                 use_container_width=True,
                )
    st.subheader("mutable features")
    st.dataframe(pd.DataFrame(
        population_, columns=X.columns).loc[:,actionable_features],
        use_container_width=True,
    )
    # immutable_mask = ~actionable_features

    # immutable_valid = np.all(
    #     np.isclose(
    #         population_[:, immutable_mask],
    #         x0[immutable_mask],
    #         rtol=0,
    #         atol=1e-10
    #     )
    # )
    # st.write(f"Are immutable features unchanged? {immutable_valid}")

    # probabilities = np.asarray(mutation_probabilities, dtype=float).copy()
    # probabilities[~actionable_features] = 0
    # if not np.all(np.isfinite(probabilities)):
    #     st.warning("Some mutation probabilities are not finite.")
    # if np.any(probabilities <0) or probabilities.sum() <= 0:
    #     raise ValueError("Mutation probabilities must be non-negative and sum to a positive value.")
    # probabilities /= probabilities.sum()

    # mutated_population = ShapMutation(
    #     mutation_probabilities=probabilities,
    #     actionable_mask=actionable_features,
    #     feature_std=X_train.std().values,
    #     perturbation_scale=0.1
    # )._do(None, population_)

    # st.subheader("mutated population")
    # st.dataframe(pd.DataFrame(mutated_population, columns=X.columns),
    #              use_container_width=True,
    #             )

target_class = 2
df = pd.DataFrame(population, columns=X.columns)
proba_population = model.predict_proba(df)
predict_population = model.predict(df)
target = list(model.classes_).index(target_class)
targetLoss = 1-proba_population[:,target]
print("\nprobability:\n", pd.DataFrame(proba_population, columns=encoder.classes_))
print("\nprediction:\n", pd.DataFrame(predict_population, columns=['prediction']))
print("\ntarget loss:\n", pd.DataFrame(targetLoss, columns=['target_loss'])) 
validity = predict_population==target_class
print("\nvalidity:\n", pd.DataFrame(validity, columns=['validity'])) 
# st.dataframe(pd.DataFrame(validity),
#              use_container_width=True,
#             )

mad = median_abs_deviation(X_train,axis=0,scale=1.0)
proximity = np.sum(np.abs(x0.values-population)/np.maximum(mad,1e-8),axis=1)


with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("target loss and validity")
        st.write(f"target class: {target_class} ({encoder.inverse_transform([target_class])[0]})")
        st.dataframe(pd.DataFrame({'target Loss': np.asarray(targetLoss).ravel(),
                               'validity': np.asarray(validity).ravel()}),
                 use_container_width=False,
                )
    with col2:
        st.subheader("proximity")
        st.dataframe(pd.DataFrame({'proximity': np.asarray(proximity).ravel()}),
                 use_container_width=False,
                )
    st.subheader("valid counterfactuals")
    for index, is_valid in enumerate(validity):
        if is_valid:
            print(f"\nStudent features {index}:")
            print(population[index])
            st.dataframe(pd.DataFrame(population[index].reshape(1,-1), 
                              columns=X.columns),
                              use_container_width=True,
                )
            print("\nReal label:",
              encoder.inverse_transform([y_test.iloc[i]])[0])

            print("Probability:",
              dict(zip(encoder.classes_,
                       proba_population[index].round(3))))

            print("Model predict:",
              encoder.inverse_transform([predict_population[index]])[0])

