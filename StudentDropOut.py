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
from pymoo.indicators.hv import HV
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.core.crossover import Crossover
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting



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
tab1,tab2,tab3,tab4,tab5, tab6 = st.tabs([
    "Data",
    "Model",
    "SHAP Values",
    "Counterfactuals",
    "Multi-objective evaluation",
    "Performance"
    ])
with tab1:
    st.subheader("Dataset")
    st.dataframe(X.join(y), use_container_width=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=32)

model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train,y_train)
predict = model.predict(X_test)
colors = ['blue', 'orange', 'green']
CLS = list(encoder.classes_)

# modelFeatureImportance = model.fit(X_train,y_train)
# featureImportance = modelFeatureImportance.feature_importances_
# indicesFeature = np.argsort(featureImportance)[::-1]
# thresholdFeature = 0.05
# importantFeatureIndices = [i for i in range(len(featureImportance)) if featureImportance[i]>=thresholdFeature]
# X_important = X_train[:,importantFeatureIndices]
# X_test_important = X_test[:,importantFeatureIndices]


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
    df_x0 = x0.to_frame().T.assign(target=encoder.inverse_transform([pred_c])[0])

    st.dataframe(df_x0, use_container_width=True)  
   
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
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)"
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

continuous_feature = [
    "Previous qualification (grade)",
    "Admission grade",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
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
    def __init__(self, X_train,x0,immutable,increase, decrease,categorical_features, binary_features,continuous_features):
        self.names = X_train.columns.tolist()
        self.x0 = np.asarray(x0, dtype=float).reshape(-1)
        self.n = len(self.names)
        feature_groups = {
            "immutable": immutable,
            "increase": increase,
            "decrease": decrease,
            "categorical": categorical_features,
            "binary": binary_features,
            "continuous": continuous_features
        }

        for label, group in feature_groups.items():
            unknown = set(group) - set(self.names)
            if unknown:
                raise ValueError(
                    f"Unknown features in {label}: {unknown}"
                )
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
        self.continuous_names =(set(continuous_features) & set(self.names))
        self.integer_names = (set(self.names) - self.discrete_names - self.continuous_names)
        groups = [
            self.cat_names,
            self.binary_names,
            self.continuous_names
        ]
        if any(groups[i] & groups[j]
            for i in range(len(groups))
                for j in range(i+1,len(groups))):
            raise ValueError("Feature type must not overlap")

        self.allowed = {
            f:np.sort(X_train[f].dropna().to_numpy(dtype=float))
            for f in self.discrete_names
        }

        self.allowed = {f:np.unique(v) for f, v in self.allowed.items()}
        self.lower= X_train.min().to_numpy(dtype=float)
        self.upper= X_train.max().to_numpy(dtype=float)
        self.std = X_train.std().to_numpy(dtype=float)

        self.lower = np.minimum(self.lower,self.x0)
        self.upper = np.maximum(self.upper,self.x0)
        for j,f in enumerate(self.names):
            if f in self.discrete_names:
               self.allowed[f] = np.unique(np.append(self.allowed[f],self.x0[j]))
        
        self.allowed_by_index = {}
        for j,f in enumerate(self.names):
            if f in self.discrete_names:
                vals = self.allowed[f]
                if self.increase[j]:
                    vals = vals[vals>=self.x0[j]]
                if self.decrease[j]:
                    vals = vals[vals<=self.x0[j]]
                if not len(vals):
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
                Y[:,j] = choices[np.abs(Y[:,j,None]-choices[None,:]).argmin(axis=1)]
            elif name in self.integer_names:
                Y[:,j] = np.rint(Y[:,j])
        return Y

    def is_valid(self, X, atol=1e-8):
        Y = np.atleast_2d(np.asarray(X, dtype=float))
        valid = np.all(np.isfinite(Y),axis=1)
        valid &= np.all(
            np.isclose(
                Y[:,~self.mutable],
                self.x0[~self.mutable],
                atol=atol,
                rtol=0
            ),axis=1
        )
        valid &=np.all(
            Y>=self.lower-atol,
            axis=1
        )
        valid &=np.all(
            Y<=self.upper+atol,
            axis=1
        )
        #directional
        valid &=np.all(
            Y[:,self.increase]>=self.x0[self.increase]-atol,
            axis=1
        )
        valid &=np.all(
             Y[:,self.decrease]<=self.x0[self.decrease]+atol,
             axis=1
        )

        for j,name in enumerate(self.names):
            if j in self.allowed_by_index:
                valid &= np.isin(
                    Y[:,j],
                    self.allowed_by_index[j]
                )
            elif name in self.integer_names:
                valid &= np.isclose(
                    Y[:,j],
                    np.rint(Y[:,j]),
                    atol=atol,
                    rtol=0
                )
        return valid

domain = FeatureDomain(
    X_train=X_train,
    x0=x0,
    immutable=immutable_features,
    increase=increase_Feature,
    decrease=decrease_Feature,
    categorical_features=categorical_features,
    binary_features=binary_features,
    continuous_features=continuous_feature
)

# class CostumSampling(Sampling):
#     def __init__(self,x0,actionable_mask,feature_std,increase_mask,decrease_mask,perturbation_scale=0.1):
#         super().__init__()
#         self.x0 = np.asarray(x0, dtype=float)
#         self.actionable_mask = np.asarray(actionable_mask,dtype=bool)
#         self.feature_std = np.asarray(feature_std, dtype=float)
#         self.increase_mask = np.asarray(increase_mask, dtype=bool)
#         self.decrease_mask = np.asarray(decrease_mask, dtype=bool)
#         self.perturbation_scale = perturbation_scale
        
#     def _do(self,problem,n_samples, **kwargs):
#         samples = np.tile(self.x0,(n_samples,1))
#         for i in range(n_samples):
#             for j in range(len(self.x0)):
#                 if not self.actionable_mask[j]:
#                     continue

#                 std = self.feature_std[j]
#                 if not np.isfinite(std) or std <= 0:
#                     continue
#                 delta = np.random.normal(
#                     loc=0,
#                     scale=self.perturbation_scale*std
#                 )
#                 if self.increase_mask[j]:
#                     delta = abs(delta)
#                 elif self.decrease_mask[j]:
#                     delta = -abs(delta)
#                 samples[i,j] += delta
#                 # samples[i,j] += np.random.normal(
#                 #     loc=0,
#                 #     scale=self.perturbation_scale*std
#                 # )
#         return samples

class CostumSampling(Sampling):
    def __init__(self, domain, perturbation_scale=0.1):
        super().__init__()
        self.domain = domain
        self.perturbation_scale = perturbation_scale

    def _do(self,problem,n_samples, **kwargs):
        rng = kwargs.get("random_state")
        if rng is None:
            rng = np.random.default_rng()
        samples = np.tile(self.domain.x0,(n_samples,1))
        for i in range(n_samples):
            for j,name in enumerate(self.domain.names):
                if not self.domain.mutable[j]:
                    continue
                if j in self.domain.allowed_by_index:
                    # Categorical codes are labels: sample choices, not numeric distances.
                    if rng.random() < 0.2:
                        samples[i,j] = rng.choice(self.domain.allowed_by_index[j])
                    continue
                std = self.domain.std[j]
                if not np.isfinite(std) or std <= 0:
                    continue
                delta = rng.normal(0,self.perturbation_scale*std)
                if self.domain.increase[j]:
                    delta = abs(delta)
                elif self.domain.decrease[j]:
                    delta = -abs(delta)
                samples[i,j] += delta
        samples[0] = self.domain.x0
        return self.domain.repair(samples)

class ShapMutation(Mutation):
    def __init__(self,domain,mutation_probabilities,perturbation_scale=0.1):
        super().__init__(prob=1.0)
        self.domain = domain
        self.perturbation_scale = perturbation_scale
        probabilities = np.asarray(mutation_probabilities,dtype=float).copy()
        probabilities[~domain.mutable] = 0
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Mutation probabilities must be finite and non-negative.")
        if probabilities.sum() == 0 and np.any(domain.mutable):
            probabilities = domain.mutable.astype(float)
        self.mutation_probabilities = probabilities / probabilities.sum() if probabilities.sum() else probabilities

    def _do(self,problem,X, **kwargs):
        rng = kwargs.get("random_state")
        if rng is None:
            rng = np.random.default_rng()
        Y = X.copy()
        if not np.any(self.domain.mutable):
            return Y
        for i in range(len(Y)):
            j = rng.choice(X.shape[1],p=self.mutation_probabilities)
            # Reversion encourages fewer changed features without a third objective.
            if rng.random() < 0.1:
                Y[i,j] = self.domain.x0[j]
            elif j in self.domain.allowed_by_index:
                choices = self.domain.allowed_by_index[j]
                choices = choices[choices != Y[i,j]]
                if len(choices):
                    Y[i,j] = rng.choice(choices)
            else:
                scale = self.perturbation_scale*self.domain.std[j]
                if np.isfinite(scale) and scale > 0:
                    delta = rng.normal(0,scale)
                    if self.domain.names[j] in self.domain.integer_names:
                        delta = np.rint(delta)
                        if delta == 0:
                            delta = rng.choice([-1,1])
                    Y[i,j] += delta
        return self.domain.repair(Y)

class DomainRepair(Repair):
    def __init__(self,domain):
        super().__init__()
        self.domain = domain

    def _do(self,problem,X,**kwargs):
        return self.domain.repair(X)

class MixedCrossover(Crossover):
    def __init__(self,domain,crossover_probabilities=0.9):
        super().__init__(2,2,prob=crossover_probabilities)
        self.domain = domain

    def _do(self,problem,X,**kwargs):
        # X shape: (2 parents, number of matings, number of features).
        n_matings = X.shape[1]
        rng = kwargs.get("random_state")
        if rng is None:
            rng = np.random.default_rng()
        Y = X.copy()
        for j,name in enumerate(self.domain.names):
            if name in self.domain.discrete_names:
                swap = rng.random(n_matings) < 0.5
                Y[0,swap,j] = X[1,swap,j]
                Y[1,swap,j] = X[0,swap,j]
            else:
                # Arithmetic crossover for numerical features; repair rounds integers.
                alpha = rng.random(n_matings)
                Y[0,:,j] = alpha*X[0,:,j] + (1-alpha)*X[1,:,j]
                Y[1,:,j] = (1-alpha)*X[0,:,j] + alpha*X[1,:,j]
        return self.domain.repair(Y.reshape(-1,self.domain.n)).reshape(Y.shape)

class CounterfactualProblem(Problem):
    def __init__(self,model,x0,target_class,domain,preprocessing,knn,k,tau,X_train):
        super().__init__(
            n_var=domain.n,
            n_obj=3,
            n_ieq_constr=2,
            xl=domain.lower,
            xu=domain.upper
        )
        self.model = model
        self.x0 = np.asarray(x0,dtype=float)
        self.target_class = target_class
        self.target = list(model.classes_).index(target_class)
        self.domain = domain
        self.preprocessing = preprocessing
        self.knn = knn
        self.k = k
        self.tau = tau
        mad = median_abs_deviation(X_train,axis=0,scale=1.0)
        # Hybrid Gower-MAD: Gower aggregation/mismatch + Wachter numeric scaling.
        # epsilon is a numerical safeguard, not a formula attributed to either paper.
        self.mad_epsilon = 1e-8
        self.distance_scale = np.maximum(mad,self.mad_epsilon)
        self.numeric_indices = [j for j,f in enumerate(domain.names) if f not in domain.discrete_names]
        self.discrete_indices = list(domain.allowed_by_index)

    def get_metrics(self,X):
        Y = np.atleast_2d(np.asarray(X,dtype=float))
        df = pd.DataFrame(Y,columns=self.domain.names)
        proba_population = self.model.predict_proba(df)
        predict_population = self.model.classes_[proba_population.argmax(axis=1)]
        targetLoss = 1-proba_population[:,self.target]
        # Modified Gower: average per-feature distance; numeric denominator is MAD, not range.
        proximity = np.sum(
            np.abs(Y[:,self.numeric_indices]-self.x0[self.numeric_indices]) /
            self.distance_scale[self.numeric_indices],axis=1
        )
        proximity += np.sum(Y[:,self.discrete_indices] != self.x0[self.discrete_indices],axis=1)
        proximity = proximity / self.domain.n
        transformed = self.preprocessing.transform(df)
        if hasattr(transformed,"toarray"):
            transformed = transformed.toarray()
        distances,_ = self.knn.kneighbors(transformed,n_neighbors=self.k)
        mean_knn = distances.mean(axis=1)
        domain_valid = self.domain.is_valid(Y)
        validity = predict_population == self.target_class
        sparsity = np.sum(~np.isclose(Y,self.x0,atol=1e-8,rtol=0),axis=1)
        return pd.DataFrame({
            "target_loss": targetLoss,
            "proximity": proximity,
            "sparsity": sparsity,
            "target_probability": proba_population[:,self.target],
            "prediction": predict_population,
            "validity": validity,
            "mean_knn_distance": mean_knn,
            "plausible": mean_knn <= self.tau,
            "domain_valid": domain_valid
        })

    def _evaluate(self,X,out,*args,**kwargs):
        metrics = self.get_metrics(X)
        out["F"] = metrics[["target_loss","proximity","sparsity"]].to_numpy()
        # pymoo inequality constraints are feasible when G <= 0.
        out["G"] = np.column_stack([
            (metrics["mean_knn_distance"]-self.tau)/max(self.tau,1e-8),
            (~metrics["domain_valid"]).astype(float)
        ])

class EvolutionHistory(Callback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def notify(self,algorithm):
        feasible = np.all(algorithm.pop.get("G") <= 0,axis=1)
        F = algorithm.pop.get("F")
        self.rows.append({
            "generation": algorithm.n_gen,
            "feasible_count": int(feasible.sum()),
            "minimum_constraint_violation": float(algorithm.pop.get("CV").min()),
            "best_feasible_target_loss": float(F[feasible,0].min()) if feasible.any() else np.nan,
            "best_feasible_proximity": float(F[feasible,1].min()) if feasible.any() else np.nan,
            "best_feasible_sparsity": float(F[feasible,2].min()) if feasible.any() else np.nan
        })

with tab4:
    st.subheader("NSGA-II parameters")
    with st.form("nsga2_parameters"):
        target_class = st.selectbox(
            "Target class",[int(c) for c in model.classes_ if c != pred_c],
            format_func=lambda c: encoder.inverse_transform([c])[0],
            index=len(model.classes_)-2
        )
        pop_size = st.number_input("Population size",min_value=20,max_value=1000,value=100,step=10)
        n_generations = st.number_input("Generations",min_value=1,max_value=1000,value=100,step=10)
        crossover_probabilities = st.number_input("Crossover Probability", min_value=0.1,max_value=0.9, step=0.05)
        perturbation_scale = st.number_input("Perturbation scale",min_value=0.01,max_value=2.0,value=0.1,step=0.05)
        seed = st.number_input("Random seed",min_value=0, value=42, step=1)
        run_nsga2 = st.form_submit_button("Run NSGA-II")
    st.subheader("selected instance(X0)")
    st.dataframe(df_x0, use_container_width=True)
    st.caption("Immutable features")
    st.dataframe(pd.DataFrame({
        "Feature": ec_features,
        "SHAP Value": shap_x0,
        "Mutable": domain.mutable,
        "Mutation Probability": mutation_probabilities
    }),use_container_width=True)
    objective_names = ["target_loss","validity","proximity"]
    if run_nsga2:
        problem = CounterfactualProblem(
            model=model,x0=x0.values,target_class=target_class,domain=domain,
            preprocessing=preprocessing,knn=knn,k=k,tau=tau,X_train=X_train
        )
        sampling = CostumSampling(domain=domain,perturbation_scale=perturbation_scale)
        population_ = sampling._do(problem,int(pop_size),random_state=np.random.default_rng(int(seed)))
        algorithm = NSGA2(
            pop_size=int(pop_size),
            sampling=population_,
            crossover=MixedCrossover(domain=domain,crossover_probabilities=crossover_probabilities),
            mutation=ShapMutation(domain=domain,mutation_probabilities=mutation_probabilities,
                                  perturbation_scale=perturbation_scale),
            repair=DomainRepair(domain),
            eliminate_duplicates=True
        )
        with st.spinner("Running NSGA-II..."):
            result = minimize(
                problem,algorithm,termination=("n_gen",int(n_generations)),
                seed=int(seed), callback= EvolutionHistory(), verbose=False
            )
        
        final_population = result.pop.get("X")
        final_metrics = problem.get_metrics(final_population)
        feasible = final_metrics[["plausible","domain_valid"]].all(axis=1).to_numpy()
        feasible_indices = np.flatnonzero(feasible)
        if len(feasible_indices):
            objectives = final_metrics.loc[feasible,["target_loss","proximity","sparsity"]].to_numpy()
            front = NonDominatedSorting().do(objectives,only_non_dominated_front=True)
            pareto_indices = feasible_indices[front]
            pareto_indices = pareto_indices[np.argsort(final_metrics.iloc[pareto_indices]["proximity"])]
        else:
            pareto_indices = np.array([],dtype=int)
        pareto_X = final_population[pareto_indices]
        pareto_metrics = final_metrics.iloc[pareto_indices].reset_index(drop=True)
        pareto_df = pd.concat([
            pd.DataFrame(pareto_X,columns=X.columns),pareto_metrics
        ],axis=1)
        pareto_df.insert(0,"candidate",[f"CF_{j+1}" for j in range(len(pareto_df))])
        st.session_state["nsga2_result_3obj"] = {
            "initial": population_,"final_metrics": final_metrics,
            "pareto_X": pareto_X,"pareto_df": pareto_df,
            "history": pd.DataFrame(result.algorithm.callback.rows),
            "target_class": target_class,
            "parameters": {"population_size": int(pop_size),"generations": int(n_generations),
                           "perturbation_scale": perturbation_scale}
        }

    if "nsga2_result_3obj" in st.session_state:
        st.subheader("Initial population")
        st.dataframe(pd.DataFrame(st.session_state["nsga2_result_3obj"]["initial"],columns=X.columns),use_container_width=True)

with tab5:
    
    # st.subheader("Constraint-feasible Pareto front")
    if "nsga2_result_3obj" not in st.session_state:
        st.info("Open Counterfactuals and click Run NSGA-II.")
    else:
        saved = st.session_state["nsga2_result_3obj"]
        pareto_df = saved["pareto_df"]
        final_metrics = saved["final_metrics"]
        st.write("Target:",encoder.inverse_transform([saved["target_class"]])[0])
        # st.write("Run parameters:",saved["parameters"],"Seed:",saved["seed"])
        # st.caption("This is an approximate Pareto front from the final population. Three minimized objectives: target loss, Gower-MAD proximity and sparsity. Feasible means domain + KNN; check validity for target-class success. Path feasibility is not evaluated here.")
        if pareto_df.empty:
            st.warning("No candidate satisfies KNN plausibility and feature-domain rules. No feasible Pareto front was found in this run.")
        else:
            st.success(f"{len(pareto_df)} non-dominated candidates.")
            st.dataframe(pareto_df,use_container_width=True)
            valid_front = pareto_df[pareto_df["validity"]]
            max_proximity = valid_front["proximity"].max()
            # st.write(max_proximity)
            # proximity_ref = max_proximity+ max(0.10 * max_proximity, 1e-8)
            # ref_point = np.array([1.01,proximity_ref,len(domain.names)+1])
            # indicator = HV(ref_point=ref_point)
            # hypervolume = indicator(valid_front)
            # st.write(f"HV: {hypervolume}")
            st.write(f"Target-valid candidates on this front: {len(valid_front)} / {len(pareto_df)}")
            if valid_front.empty:
                st.warning("The front contains no target-valid counterfactual in this run.")
            # Non-dominance is computed in all three objectives, not in this 2D projection.
            fig,ax = plt.subplots(figsize=(8,4))
            points = ax.scatter(pareto_df["proximity"],pareto_df["target_loss"],
                                c=pareto_df["sparsity"],cmap="viridis",s=50)
            fig.colorbar(points,ax=ax,label="Sparsity (changed features)")
            ax.set_xlabel("Gower-MAD proximity")
            ax.set_ylabel("Target loss (1 - target probability)")
            ax.set_title("Projection of the three-objective Pareto front")
            ax.grid(alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8,4))
            ax = fig.add_subplot(111,projection="3d")
            ax.scatter(pareto_df["proximity"],pareto_df["target_loss"],
                       pareto_df["sparsity"],c=pareto_df["sparsity"],cmap="viridis")
            ax.set_xlabel("Gower-MAD proximity")
            ax.set_ylabel("Target loss")
            ax.set_zlabel("Sparsity")
            ax.set_title("Three-objective Pareto front")
            st.pyplot(fig)
            plt.close(fig)
            st.subheader("Target-valid subset of the Pareto front")
            st.dataframe(valid_front,use_container_width=True)
            # st.download_button("Download target-valid Pareto subset (CSV)",
            #     valid_front.to_csv(index=False).encode("utf-8"),
            #     file_name="valid_pareto_counterfactuals.csv",mime="text/csv")
            # st.download_button("Download Pareto candidates (CSV)",
            #     pareto_df.to_csv(index=False).encode("utf-8"),
            #     file_name="pareto_counterfactuals.csv",mime="text/csv")
            st.subheader("Feature changes from x0")
            selected_cf = st.selectbox("Candidate",range(len(pareto_df)),
                                      format_func=lambda j: pareto_df.iloc[j]["candidate"])
            values = saved["pareto_X"][selected_cf]
            changed = ~np.isclose(values,x0.values,atol=1e-8,rtol=0)
            st.dataframe(pd.DataFrame({"Feature": X.columns[changed],
                "x0": x0.values[changed],"counterfactual": values[changed]}),use_container_width=True)
        st.subheader("Final population evaluation")
        st.dataframe(final_metrics,use_container_width=True)
        st.subheader("Evolution history")
        st.dataframe(saved["history"],use_container_width=True)
       