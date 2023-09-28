import os
import re
import zlib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.sparse import csr_matrix
from collections.abc import Sequence, Callable
from sklearn.feature_extraction.text import CountVectorizer

import utils

SeqReader = Callable[[str], str]

def ngram_extract(ids: Sequence[str], dir: Path, seq_reader: SeqReader, n: int) -> tuple[CountVectorizer, csr_matrix]:
    """
    Extract samples' n-grams.

    -- PARAMETERS --
    ids: Sample IDs.
    dir: Samples' directory.
    seq_reader: A callback function used to extract a sequence content from a sample.
    n: The N value.

    -- RETURNS --
    A fitted `CountVectorizer` model.
    A sparse count matrix.
    """
    class Reader:
        """
        A wrapper for file-like input of `CountVectorizer`.
        """
        def __init__(self, id: str, dir: Path) -> None:
            self._id = id
            self._dir = dir
        
        def read(self) -> str:
            return seq_reader(self._id, self._dir)
    
    seqs = [Reader(id, dir) for id in ids]
    # Name mangling should be considered, `token_pattern` cannot be the default.
    ngrm_vct = CountVectorizer(ngram_range=(n, n), stop_words=None, token_pattern=r"(?u)\b[\w@?]{2,}\b", lowercase=False, input="file")
    ngrms = ngrm_vct.fit_transform(seqs)
    return ngrm_vct, ngrms

def fileSize_extract(ids: Sequence[str], dir: Path) -> pd.DataFrame:
    """
    Extract samples' file sizes:

    -- PARAMETERS --
    ids: Sample IDs.
    dir: Samples' directory.

    -- RETURNS --
    A dataframe having the following columns:
      - The disassembly size.
      - The byte size.
      - The ratio of the disassembly size and byte size.
    """
    df = pd.DataFrame(columns=["ID", "Asm_Size", "Byte_Size", "Ratio"], dtype=float).set_index("ID")
    for id in tqdm(ids):
        df.at[id, "Asm_Size"] = os.path.getsize(dir.joinpath(id + ".asm"))
        df.at[id, "Byte_Size"] = os.path.getsize(dir.joinpath(id + ".bytes"))
    
    df[["Asm_Size", "Byte_Size"]] = df[["Asm_Size", "Byte_Size"]].astype(int)
    df["Ratio"] = (df["Asm_Size"] / df["Byte_Size"]).round(5)
    return df

def load_sections(id: str, dir: Path) -> list[utils.Section]:
        """
        Extract a sample's section attributions.
        """
        # The separator between words such as "Virtual size" can be a space or a tab. So the regular expression should use `Virtual\s+size` instead of `Virtual size``
        sctn_attr_rgx = re.compile(r"([A-Za-z]+):[\dA-F]{8}\s+;\s+Virtual\s+size\s+:\s+[\dA-F]{8}\s+\(\s*(\d+)\.[\s\S]+?Section\s+size\s+in\s+file\s+:\s+[\dA-F]{8}\s+\(\s*(\d+)\.[\s\S]+?;\s+Flags\s+[\dA-F]{8}:[\s\S]+?\s+((?:Executable|Readable|Writable|\s)+)", flags=re.IGNORECASE)

        asm = utils.Read.asm(dir, id)
        sctns = []
        for attr in sctn_attr_rgx.findall(asm):
            sctn = utils.Section()
            sctn.name, sctn.virtual_size, sctn.raw_size, access = attr[0].lower(), int(attr[1]), int(attr[2]), set(attr[3].split())
            if "Executable" in access:
                sctn.executable = True
            if "Writable" in access:
                sctn.writable = True
            sctns.append(sctn)
        return sctns

def sectionAttr_extract(ids: Sequence[str], dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract samples' section attributions.

    -- PARAMETERS --
    ids: Sample IDs.
    dir: Samples' directory.

    -- RETURNS --
    Two dataframes.
    The 1st contains each section's sizes.
      - The virtual size.
      - The raw size.
      - The ratio of the raw size and virtual size.
    The 2nd contains the total sizes of sections grouped by access properties.
      - The virtual sizes and raw sizes of executable, readable and writable data.
      - The ratio of the raw size and virtual size.

    For each section or access property `<X>`, there will be three columns named `<X>-Virtual`, `<X>-Raw` and `<X>-Ratio`.
    """
    rwe_cols = [f"{i}-{j}" for i in ["Executable", "Writable", "Readable"] for j in ["Virtual", "Raw", "Ratio"]]
    rwe = pd.DataFrame(columns=["ID"] + rwe_cols, dtype=float).set_index("ID")
    sctn = pd.DataFrame(columns=["ID"], dtype=float).set_index("ID")

    # Save the names of integer columns.
    sctn_int_cols = set()

    for id in tqdm(ids):
        rwe.loc[id, :] = 0
        sctn.loc[id, :] = 0
        for attr in load_sections(id, dir):
            # Save sizes by section names
            sctn.at[id, f"{attr.name}-Virtual"] = attr.virtual_size
            sctn.at[id, f"{attr.name}-Raw",] = attr.raw_size
            sctn.at[id, f"{attr.name}-Ratio",] = round(attr.raw_size / attr.virtual_size, 5)
            sctn_int_cols.update([f"{attr.name}-Virtual", f"{attr.name}-Raw"])

            # Save sizes by access properties
            rwe.at[id, "Readable-Virtual"] += attr.virtual_size
            rwe.at[id, "Readable-Raw"] += attr.raw_size
            if attr.writable:
                rwe.at[id, "Writable-Virtual"] += attr.virtual_size
                rwe.at[id, "Writable-Raw"] += attr.raw_size
            if attr.executable:
                rwe.at[id, "Executable-Virtual"] += attr.virtual_size
                rwe.at[id, "Executable-Raw"] += attr.raw_size
    
    # Convert column types into the integer.
    sctn.fillna(0, inplace=True)
    sctn_int_cols = list(sctn_int_cols)
    sctn[sctn_int_cols] = sctn[sctn_int_cols].astype(int)

    rwe.fillna(0, inplace=True)
    rwe_int_cols = [i for i in rwe.columns if "-Virtual" in i or "-Raw" in i]
    rwe[rwe_int_cols] = rwe[rwe_int_cols].astype(int)

    # Calculate the size ratio for different access properties.
    rwe["Readable-Ratio"] = (rwe["Readable-Raw"] / rwe["Readable-Virtual"]).round(5)
    rwe["Writable-Ratio"] = (rwe["Writable-Raw"] / rwe["Writable-Virtual"]).round(5)
    rwe["Executable-Ratio"] = (rwe["Executable-Raw"] / rwe["Executable-Virtual"]).round(5)
    return sctn, rwe.fillna(0)

def syscallSequence_extract(id: str, dir: Path) -> str:
    """
    Extract a sample's system call sequence.

    -- PARAMETERS --
    id: A sample ID.
    dir: Sample's directory.

    -- RETURNS --
    A system call sequence.
    """
    call_rgx = re.compile(r"\scall\s+(?:ds:)(?:__imp_)?([^\s]+)", flags=re.IGNORECASE)
    api_rgx = re.compile(r"extrn\s+(?:__imp_)?([^\s:]+)", flags=re.IGNORECASE)
    asm = utils.Read.asm(dir, id)
    apis = set(api_rgx.findall(asm))
    calls = call_rgx.findall(asm)
    syscalls = [i for i in calls if i in apis]
    return " ".join(syscalls)

def opcodeSequence_extract(id: str, dir: Path) -> str:
    """
    Extract a sample's operation code sequence.

    -- PARAMETERS --
    id: A sample ID.
    dir: Sample's directory.

    -- RETURNS --
    An operation code sequence.
    """
    opcode_rgx = re.compile(r"\s[\dA-F]{2}(?:\+)?\s+([a-z]+)\s")
    opcodes = []
    for line in utils.Read.asm_lines(dir, id):
        for opcode in opcode_rgx.findall(line):
            opcodes.append(opcode.lower())
    return " ".join(opcodes)

def contentComplexity_extract(ids: Sequence[str], dir: Path) -> pd.DataFrame:
    """
    Extract samples' content complexity.

    -- PARAMETERS --
    ids: Sample IDs.
    dir: Samples' directory.

    -- RETURNS --
    A dataframe having the following columns:
      - The disassembly string length.
      - The compressed disassembly string length.
      - The disassembly compression ratio.
      - The byte string length.
      - The compressed byte string length.
      - The byte compression ratio.
    """
    df = pd.DataFrame(columns=["ID", "Asm-Len", "Zip-Asm-Len", "Asm-Zip-Ratio", "Byte-Len", "Zip-Byte-Len", "Byte-Zip-Ratio"], dtype=float).set_index("ID")
    for id in tqdm(ids):
        asm = utils.Read.asm(dir, id).encode("utf-8")
        bytes = utils.Read.bytes(dir, id)
        bytes = " ".join([str[byte] for byte in bytes]).encode("utf-8")
        df.at[id, "Asm-Len"] = len(asm)
        df.at[id, "Zip-Asm-Len"] = len(zlib.compress(asm))
        df.at[id, "Byte-Len"] = len(bytes)
        df.at[id, "Zip-Byte-Len"] = len(zlib.compress(bytes))

    df[["Asm-Len", "Zip-Asm-Len", "Byte-Len", "Zip-Byte-Len"]] = df[["Asm-Len", "Zip-Asm-Len", "Byte-Len", "Zip-Byte-Len"]].astype(int)
    df["Asm-Zip-Ratio"] = (df["Asm-Len"] / df["Zip-Asm-Len"]).round(5)
    df["Byte-Zip-Ratio"] = (df["Byte-Len"] / df["Zip-Byte-Len"]).round(5)
    
    return df

def libSet_extract(id: str, dir: Path) -> set:
    """
    Extract a sample's import library set.

    -- PARAMETERS --
    id: A sample ID.
    dir: Sample's directory

    -- RETURNS --
    An import library set.
    """
    lib_rgx = re.compile(r"Imports\s+from\s+(.+).dll", flags=re.IGNORECASE)
    asm = utils.Read.asm(dir, id)
    return set([i.lower() for i in set(lib_rgx.findall(asm))])

def libSet_class_extract(type: int, labels: pd.DataFrame, dir: Path) -> set:
    """
    Extract a class's import library set.

    -- PARAMETERS --
    type: A malware class.

    -- RETURNS --
    An import library set.
    """
    samples = [i for i in labels.index if labels.at[i, "Class"] == type]
    libs = set()
    for i in samples:
        libs.update(libSet_extract(i, dir))
    return libs

def libSequence_extract(id: str, dir: Path) -> str:
    """
    Extract a sample's import library sequence.

    -- PARAMETERS --
    id: A sample ID.
    dir: Sample's directory.

    -- RETURNS --
    An import library sequence.
    """
    libs = libSet_extract(id, dir)
    return " ".join(list(libs))
