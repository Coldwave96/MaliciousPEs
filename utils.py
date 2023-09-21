import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from collections.abc import Sequence

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
