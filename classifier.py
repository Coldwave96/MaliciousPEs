import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz, load_npz, csr_matrix

import utils
import feature_extraction

print("[*] Loading labels...")
# Load labels
labels = pd.read_csv(utils.data_dir.joinpath("trainLabels.csv"))
labels["ID"] = labels["Id"].astype(str)
labels["Class"] = labels["Class"].astype("category")
labels.set_index("ID", inplace=True)

print("[*] Done!\n\n[*] Extracting features - File Size")
###### File Size
# Extract each sample's disassembly size and byte size, then calculate the ratio
file_sizes = feature_extraction.fileSize_extract(labels.index, utils.train_dir)
assert (file_sizes.index == labels.index).all()
# Save all the file sizes
utils.CSVFeature("file_sizes.csv").save(file_sizes)
######

print("[*] Done!\n\n[*] Extracting features - Section Attributions")
###### Section Attributions
"""
Create two dataframes for section attributions:
The 1st saves each section's sizes.
The 2nd saves the total sizes of sections grouped by access properties.

For each section or access property <X>, there will be three columns named <X>-Virtual, <X>-Raw and <X>-Ratio storing its virtual size, raw size and size ratio.
"""
sctn_sizes, rwe_sizes = feature_extraction.sectionAttr_extract(labels.index, utils.train_dir)
assert (sctn_sizes.index == labels.index).all()
assert (rwe_sizes.index == labels.index).all()

utils.CSVFeature("section_permissions.csv").save(rwe_sizes)

# Dimensionality Reduction
# Use Random Forest to choose the most important section columns, remaining the top 25
X_train, X_test, y_train, y_test = train_test_split(sctn_sizes, labels, test_size=0.2, random_state=42, stratify=labels)

rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train.values.ravel())

top_ftr_idx = np.argsort(rnd_clf.feature_importances_)[::-1][:25]
top_ftrs = X_train.columns[top_ftr_idx]

# Delete other columns
top_sctn_sizes = sctn_sizes[top_ftrs]
# Save the top sections to files
utils.CSVFeature("section_sizes.csv").save(top_sctn_sizes)
######

print("[*] Done!\n\n[*] Extracting features - API 4-grams")
###### API 4-grams
api_ngram_vct, api_ngrams = feature_extraction.ngram_extract(labels.index, utils.train_dir, feature_extraction.syscallSequence_extract, 4)
joblib.dump(api_ngram_vct, utils.feature_dir.joinpath("api_4gram.joblib"))

# Dimensionality Reduction
# Calculate the sum of columns and get the columns with the highest frequency, remaining the top 5000
# Calculate the sum
col_sum = api_ngrams.sum(axis=0).A1
# Get the index of the top columns and sort by frequency
top_frqn_ftr_idx = np.argsort(col_sum)[::-1][:5000]

# Save the sparse matrix to `.npz` files
save_npz(utils.feature_dir.joinpath("api_4grams.npz"), api_ngrams[:, top_frqn_ftr_idx])
# Save the feature names to a list file
utils.List("api_4gram_names.txt").save(top_frqn_ftr_idx)
######

print("[*] Done!\n\n[*] Extracting features - Opcode 4-grams")
###### Opcode 4-grams
opcode_ngram_vct, opcode_ngrams = feature_extraction.ngram_extract(labels.index, utils.train_dir, feature_extraction.opcodeSequence_extract, 4)
joblib.dump(opcode_ngram_vct, utils.feature_dir.joinpath("opcode_4gram.joblib"))

# Dimensionality Reduction
# Calculate the sum of columns and get the columns with the highest frequency, remaining the top 5000
# Calculate the sum
col_sum = opcode_ngrams.sum(axis=0).A1
# Get the index of the top columns and sort by frequency
top_frqn_ftr_idx = np.argsort(col_sum)[::-1][:5000]

# Save the sparse matrix to `.npz` files
save_npz(utils.feature_dir.joinpath("opcode_4grams.npz"), opcode_ngrams[:, top_frqn_ftr_idx])
# Save the feature names to a list file
utils.List("opcode_4gram_names.txt").save(top_frqn_ftr_idx)
######

print("[*] Done!\n\n[*] Extracting features - Content Complexity")
###### Content Complexity
complexity = feature_extraction.contentComplexity_extract(labels.index, utils.train_dir)
assert (complexity.index == labels.index).all()

utils.CSVFeature("content_complexity.csv").save(complexity)
######

print("[*] Done!\n\n[*] Extracting features - Import Libraries")
###### Import Libraries
lib_ngram_vct, lib_ngrams = feature_extraction.ngram_extract(labels.index, utils.train_dir, feature_extraction.libSequence_extract, 1)
joblib.dump(lib_ngram_vct, utils.feature_dir.joinpath("lib_1gram.joblib"))

lib_ngrams = csr_matrix(lib_ngrams, dtype=np.int16)

# Dimensionality Reduction
# Calculate the sum of columns and get the columns with the highest frequency, remaining the top 300
# Calculate the sum
col_sum = lib_ngrams.sum(axis=0).A1
# Get the index of the top columns
top_frqn_ftr_idx = np.argsort(col_sum)[::-1][:300]

# Save the sparse matrix to a `.npz` file
save_npz(utils.feature_dir.joinpath("lib_1grams.npz"), lib_ngrams[:, top_frqn_ftr_idx])
# Save the feature names to a list file
utils.List("lib_1gram_names.txt").save([lib_ngram_vct.get_feature_names_out()[i] for i in top_frqn_ftr_idx])
######

print("[*] Done!\n\n[*] Individual Feature Experiments - File Sizes")
X = utils.CSVFeature("file_sizes.csv").load()

