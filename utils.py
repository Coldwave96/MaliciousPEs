import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from collections.abc import Sequence
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier

# The family names of malware samples
classes = ("Ramnit", "Lollipop", "Kelihos_ver3", "Vundo", "Simda", "Tracur", "Kelihos_ver1", "Obfuscator.ACY", "Gatak")

# Datasets directory
data_dir = Path("Datasets")
# Train dataset directory
train_dir = data_dir.joinpath("train")
# Test dataset directory
test_dir = data_dir.joinpath("test")
# Sample dataset directory
sample_dir = data_dir.joinpath("dataSample")

# Statistic directory.
stats_dir = Path("Stats")
if not stats_dir.exists():
    stats_dir.mkdir()

# Feature directory
feature_dir = Path("Features")
if not feature_dir.exists():
    feature_dir.mkdir()

# Read different kinds of content from a sample file
class Read:
    # Read a sample's byte content
    @staticmethod
    def bytes(dir: Path, id: str) -> list[str]:
        with dir.joinpath(id + ".bytes").open() as file:
            data = file.read()
        items = data.split()
        byte_list = []
        for item in items:
            if len(item) == 2 and item != "??":
                byte_list.append(item)
        return byte_list
    
    # Read a sample's disassembly content as a large string
    @staticmethod
    def asm(dir: Path, id: str) -> str:
        with dir.joinpath(id + ".asm").open(encoding="utf-8", errors="ignore") as file:
            return file.read()
        
    # Read a sample's disassembly content as a list of lines
    @staticmethod
    def asm_lines(dir: Path, id: str) -> list[str]:
        with dir.joinpath(id + ".asm").open(encoding="utf-8", errors="ignore") as file:
            return file.readlines()

# Manage the saving and loading of a `.csv` feature file
class CSVFeature:
    def __init__(self, file: str) -> None:
        """
        The constructor.

        -- PARAMETERS --
        file: A `.csv` file name.
        """
        self.file: str = file
    
    def save(self, data: pd.DataFrame) -> None:
        """
        Save a dataframe into a `.csv` file.

        -- PARAMETERS --
        data: A dataframe.
        """
        data.to_csv(feature_dir.joinpath(self.file))
    
    def load(self) -> pd.DataFrame | None:
        """
        Load a dataframe from a `.csv` file.

        -- RETURNS --
        A dataframe or `None` if the file does not exist.
        """
        path = feature_dir.joinpath(self.file)
        return pd.read_csv(path).set_index("ID") if path.exists() else None

# Manage the loading of a `.npz` feature file
class NPZFeature:
    def __init__(self, file: str) -> None:
        """
        The constructor.

        -- PARAMETERS --
        file: A `.npz` file name.
        """
        self.file: str = file
    
    def load(self) -> pd.DataFrame | None:
        """
        Load a dataframe from a `.npz` file.

        -- RETURNS --
        A dataframe or `None` if the file does not exist.
        """
        path = feature_dir.joinpath(self.file)
        return pd.DataFrame(load_npz(path).toarray()) if path.exists() else None

# Manage the saving and loading of a list file
class List:
    def __init__(self, file: str) -> None:
        """
        The constructor.

        -- PARAMETERS --
        file: A list file name.
        """
        self.file = file
    
    def save(self, data: Sequence[str]) -> None:
        """
        Save a sequence into a list file.

        -- PARAMETERS --
        data: A list.
        """
        path = feature_dir.joinpath(self.file)
        with path.open("w", encoding="utf-8") as file:
            for line in data:
                file.write(f"{str(line)}\n")
    
    def load(self) -> list[str] | None:
        """
        Load a sequence from a list file.

        -- RETURNS --
        A list or `None` if the file does not exist.
        """
        path = feature_dir.joinpath(self.file)
        if path.exists():
            with path.open(encoding="utf-8") as file:
                return file.read().splitlines("\n")
        else:
            return None

# Save a section's attributions
class Section:
    def __init__(self) -> None:
        self.name: str = ""
        self.virtual_size: int = 0
        self.raw_size: int = 0
        self.executable: bool = False
        self.writable: bool = False

# Model Names
BASE = "Baseline"
KNN = "K-Nearest"
SVM = "Support Vector"
RF = "Random Forest"

# Feature Names
FILE_SIZE = "File Size"
SCTN_SIZE = "Section Size"
RWE_SIZE = "Section Permission"
API_NGRAM = "API 4-gram"
OPCODE_NGRAM = "Opcode 4-gram"
CMPLXTY = "Content Complexity"
IMP_LIB = "Import Library"

# Manage test scores
class IndividualScoreStats:
    def __init__(self, file: str) -> None:
        """
        Load or create a new score dataframe.
        """
        self._path: Path = stats_dir.joinpath(file)
        if self._path.exists():
            self.df: pd.DataFrame = pd.read_csv(self._path).set_index("Feature")
        else:
            self.df: pd.DataFrame = pd.DataFrame(columns=["Feature", BASE, KNN, SVM, RF, "Dimension"]).set_index("Feature")
    
    def new_feature(self, name: str, dimension: str | int) -> None:
        self.df.at[name, "Dimension"] = dimension

    def update(self, ftr: str, mod: str, score: float) -> None:
        """
        Update a score.

        -- PARAMETERS --
        ftr: A feature name.
        mod: A model name.
        score: A score value.
        """
        self.df.at[ftr, mod] = round(score, 4)

    def save(self) -> None:
        """
        Save scores to a `.csv` file.
        """
        self.df.to_csv(self._path)

MEMO_LIMIT = 1024 * 10
TIME_LIMIT = 20

def automl_cross_val(ftr: str, mod: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    if mod == KNN:
        include = ["k_nearest_neighbors"]
    elif mod == SVM:
        include = ["libsvm_svc"]
    elif mod == RF:
        include = ["random_forest"]
    elif mod is None or len(mod) == 0:
        include = None
    else:
        assert False, "Unknown Model"

    if include is None:
        automl = AutoSklearnClassifier(
            memory_limit = MEMO_LIMIT,
            time_left_for_this_task = TIME_LIMIT * 60, 
            ensemble_size = 1, 
            resampling_strategy = "cv", 
            resampling_strategy_arguments = {"folds": 5}
        )
    else:
        automl = AutoSklearnClassifier(
            memory_limit = MEMO_LIMIT, 
            time_left_for_this_task = TIME_LIMIT * 60, 
            ensemble_size = 1, 
            include_estimators = include, 
            resampling_strategy = "cv", 
            resampling_strategy_arguments = {"folds": 5}
        )
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test).astype(int)
    return accuracy_score(y_test, y_pred), automl

def best_model(automl: AutoSklearnClassifier) -> dict:
    """
    Get the best model from an `AutoSklearnClassifier` object.
    """
    return automl.cv_results_["params"][np.argmax(automl.cv_results_["mean_test_score"])]

class IndividualScoreStats:
    """
    Manage test scores.
    """
    def __init__(self, file: str) -> None:
        """
        Load or create a new score dataframe.
        """
        self._path: Path = stats_dir.joinpath(file)
        if self._path.exists():
            self.df: pd.DataFrame = pd.read_csv(self._path).set_index("Feature")
        else:
            self.df: pd.DataFrame = pd.DataFrame(columns=["Feature", BASE, KNN, SVM, RF, "Dimension"]).set_index("Feature")
    
    def new_feature(self, name: str, dimension: str | int) -> None:
        self.df.at[name, "Dimension"] = dimension
    
    def update(self, ftr: str, mod: str, score: float) -> None:
        """
        Update a score.

        -- PARAMETERS --
        ftr: A feature name.
        mod: A model name.
        score: A score value.
        """
        self.df.at[ftr, mod] = round(score, 4)

    def save(self) -> None:
        """
        Save scores to a `.csv` file.
        """
        self.df.to_csv(self._path)

"""
Integrated Feature Names
"""
SCTN_RWE_CMPLXTY = "Section Size + Section Permission + Content Complexity"
SCTN_RWE_CMPLXTY_LIB = "Section Size + Section Permission + Content Complexity + Import Library"
FILE_API_OPCODE = "File Size + API 4-gram + Opcode 4-gram"
ALL = "All Features"

def integrated_features(ftrs: set[str], labels: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate features.
    """
    def load_npz(file: str, prefix: str) -> pd.DataFrame:
        df = NPZFeature(file).load()
        df["ID"] = labels.index
        df.set_index("ID", inplace=True)
        df.columns = [f"{prefix}-{str(col)}" for col in df.columns]
        return df
    
    file_sizes = CSVFeature("file_sizes.csv").load()
    sctn_sizes = CSVFeature("section_sizes.csv").load()
    rwe_sizes = CSVFeature("section_permissions.csv").load()
    cmplxty = CSVFeature("content_complexity.csv").load()

    api_4grams = load_npz("api_4grams.npz", "api")
    opcode_4grams = load_npz("opcode_4grams.npz", "opcode")
    lib_1grams = load_npz("lib_1grams.npz", "lib")

    dfs = []
    if FILE_SIZE in ftrs:
        dfs.append(file_sizes)
    
    if SCTN_SIZE in ftrs:
        dfs.append(sctn_sizes)
    
    if RWE_SIZE in ftrs:
        dfs.append(rwe_sizes)
    
    if CMPLXTY in ftrs:
        dfs.append(cmplxty)
    
    if API_NGRAM in ftrs:
        dfs.append(api_4grams)
    
    if OPCODE_NGRAM in ftrs:
        dfs.append(opcode_4grams)
    
    if IMP_LIB in ftrs:
        dfs.append(lib_1grams)
    
    return pd.concat(dfs, axis="columns")

class IntegratedScoreStats:
    """
    Manage test scores.
    """
    def __init__(self, file: str) -> None:
        """
        Load or create a new score dataframe.
        """
        self._path: Path = stats_dir.joinpath(file)
        if self._path.exists():
            self._df: pd.DataFrame = pd.read_csv(self._path).set_index("Features")
        else:
            self._df: pd.DataFrame = pd.DataFrame(columns=["Feature", "Dimension", "Best Accuracy", "Best Model"])
    
    def new_feature(self, name: str, dimension: str | int) -> None:
        self.df.at[name, "Dimension"] = dimension
    
    def update(self, ftr: str, mod: str, score: float) -> None:
        """
        Update a score.

        -- PARAMETERS --
        ftr: A feature name.
        mod: A model name.
        score: A score value.
        """
        if score > self.df.at[ftr, "Best Accuracy"]:
            self.df.at[ftr, "Best Accuracy"] = round(score, 4)
            self.df.at[ftr, "Best Model"] = mod
        
    def save(self) -> None:
        """
        Save scores to a `.csv` file.
        """
        self.df.to_csv(self._path)
