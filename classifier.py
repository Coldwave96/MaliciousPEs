import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
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
labels = labels.drop(columns=["Id"], axis=1)
labels.set_index("ID", inplace=True)

print("[*] Done!\n\n[*] Extracting features - File Size")
###### File Size
# Extract each sample's disassembly size and byte size, then calculate the ratio
if utils.feature_dir.joinpath("file_sizes.csv").exists():
    file_sizes = pd.read_csv(utils.feature_dir.joinpath("file_sizes.csv"))
else:
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
if utils.feature_dir.joinpath("section_permissions.csv").exists() and utils.feature_dir.joinpath("section_sizes_raw.csv").exists():
    rwe_sizes = pd.read_csv(utils.feature_dir.joinpath("section_permissions.csv"))
    sctn_sizes = pd.read_csv(utils.feature_dir.joinpath("section_sizes_raw.csv"))
else:
    sctn_sizes, rwe_sizes = feature_extraction.sectionAttr_extract(labels.index, utils.train_dir)
    assert (sctn_sizes.index == labels.index).all()
    assert (rwe_sizes.index == labels.index).all()

    sctn_sizes.set_index("ID", inplace=True)

    utils.CSVFeature("section_permissions.csv").save(rwe_sizes)
    utils.CSVFeature("section_sizes_raw.csv").save(sctn_sizes)

    # Dimensionality Reduction
    # Use Random Forest to choose the most important section columns, remaining the top 25
    X_train, X_test, y_train, y_test = train_test_split(sctn_sizes, labels, test_size=0.2, random_state=42)

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
if utils.feature_dir.joinpath("api_4gram.joblib").exists() and utils.feature_dir.joinpath("api_4grams.npz").exists():
    api_ngram_vct = joblib.load(utils.feature_dir.joinpath("api_4gram.joblib"))
    api_ngrams = load_npz(utils.feature_dir.joinpath("api_4grams.npz"))
else:
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
if utils.feature_dir.joinpath("opcode_4gram.joblib").exists() and utils.feature_dir.joinpath("opcode_4grams.npz").exists():
    opcode_ngram_vct = joblib.load(utils.feature_dir.joinpath("opcode_4gram.joblib"))
    opcode_ngrams = load_npz(utils.feature_dir.joinpath("opcode_4grams.npz"))
else:
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
if utils.feature_dir.joinpath("content_complexity.csv").exists():
    complexity = pd.read_csv(utils.feature_dir.joinpath("content_complexity.csv"))
else:
    complexity = feature_extraction.contentComplexity_extract(labels.index, utils.train_dir)
    assert (complexity.index == labels.index).all()

    utils.CSVFeature("content_complexity.csv").save(complexity)
######

print("[*] Done!\n\n[*] Extracting features - Import Libraries")
###### Import Libraries
if utils.feature_dir.joinpath("lib_1gram.joblib").exists() and utils.feature_dir.joinpath("lib_1grams.npz").exists():
    lib_ngram_vct = joblib.load(utils.feature_dir.joinpath("lib_1gram.joblib"))
    lib_ngrams = load_npz(utils.feature_dir.joinpath("lib_1grams.npz"))
else:
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
    utils.List("lib_1gram_names.txt").save([lib_ngram_vct.get_feature_names()[i] for i in top_frqn_ftr_idx])
######

###### Individual Feature Experiments
slgn_stats = utils.IndividualScoreStats("individual_socres.csv")

# Baseline
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(labels, labels["Class"])
y_pred = dummy_clf.predict(labels)
base_score = accuracy_score(labels, y_pred)

### Fize Size
print("[*] Done!\n\n[*] Individual Feature Experiments - File Sizes")
X = utils.CSVFeature("file_sizes.csv").load()
slgn_stats.new_feature(utils.FILE_SIZE, len(X.columns))
slgn_stats.update(utils.FILE_SIZE, utils.BASE, base_score)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.FILE_SIZE, utils.KNN, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.FILE_SIZE, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# SVM
score, auto_svm = utils.automl_cross_val(utils.FILE_SIZE, utils.SVM, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.FILE_SIZE, utils.SVM, score)

print(f"Best {utils.SVM} model:\n{utils.best_model(auto_svm)}\n")
print(auto_svm.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.FILE_SIZE, utils.RF, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.FILE_SIZE, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())

# Summary
slgn_stats.save()
summary = slgn_stats.df.loc[utils.FILE_SIZE, [utils.BASE, utils.KNN, utils.SVM, utils.RF]].sort_values(ascending=False)
print(f"Summary:\n{summary}\n")
###

### PE Section Sizes
print("[*] Individual Feature Experiments - PE Section Sizes")
X = utils.CSVFeature("section_sizes.csv").load()
slgn_stats.new_feature(utils.SCTN_SIZE, f"{len(sctn_sizes.columns)} -> {len(X.columns)}")
slgn_stats.update(utils.SCTN_SIZE, utils.BASE, base_score)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.SCTN_SIZE, utils.KNN, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.SCTN_SIZE, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# SVM
score, auto_svm = utils.automl_cross_val(utils.SCTN_SIZE, utils.SVM, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.SCTN_SIZE, utils.SVM, score)

print(f"Best {utils.SVM} model:\n{utils.best_model(auto_svm)}\n")
print(auto_svm.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.SCTN_SIZE, utils.RF, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.SCTN_SIZE, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())

# Summary
slgn_stats.save()
summary = slgn_stats.df.loc[utils.SCTN_SIZE, [utils.BASE, utils.KNN, utils.SVM, utils.RF]].sort_values(ascending=False)
print(f"Summary:\n{summary}\n")
###

### PE Section Permissions
print("[*] Individual Feature Experiments - PE Section Permissions")
X = utils.CSVFeature("section_sizes.csv").load()
slgn_stats.new_feature(utils.RWE_SIZE, len(X.columns))
slgn_stats.update(utils.RWE_SIZE, utils.BASE, base_score)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.RWE_SIZE, utils.KNN, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.RWE_SIZE, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# SVM
score, auto_svm = utils.automl_cross_val(utils.RWE_SIZE, utils.SVM, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.RWE_SIZE, utils.SVM, score)

print(f"Best {utils.SVM} model:\n{utils.best_model(auto_svm)}\n")
print(auto_svm.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.RWE_SIZE, utils.RF, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.RWE_SIZE, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())

# Summary
slgn_stats.save()
summary = slgn_stats.df.loc[utils.RWE_SIZE, [utils.BASE, utils.KNN, utils.SVM, utils.RF]].sort_values(ascending=False)
print(f"Summary:\n{summary}\n")
###

### API 4-grams
print("[*] Individual Feature Experiments - API 4-grams")
X = utils.NPZFeature("api_4grams.npz").load().set_index(labels.index)
slgn_stats.new_feature(utils.API_NGRAM, f"{len(X.columns)}")
slgn_stats.update(utils.API_NGRAM, utils.BASE, base_score)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.API_NGRAM, utils.KNN, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.API_NGRAM, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# SVM
score, auto_svm = utils.automl_cross_val(utils.API_NGRAM, utils.SVM, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.API_NGRAM, utils.SVM, score)

print(f"Best {utils.SVM} model:\n{utils.best_model(auto_svm)}\n")
print(auto_svm.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.API_NGRAM, utils.RF, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.API_NGRAM, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())

# Summary
slgn_stats.save()
summary = slgn_stats.df.loc[utils.API_NGRAM, [utils.BASE, utils.KNN, utils.SVM, utils.RF]].sort_values(ascending=False)
print(f"Summary:\n{summary}\n")
###

### Opcode 4-grams
print("[*] Individual Feature Experiments - Opcode 4-grams")
X = utils.NPZFeature("opcode_4grams.npz").load().set_index(labels.index)
slgn_stats.new_feature(utils.OPCODE_NGRAM, f"{len(X.columns)}")
slgn_stats.update(utils.OPCODE_NGRAM, utils.BASE, base_score)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.OPCODE_NGRAM, utils.KNN, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.OPCODE_NGRAM, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# SVM
score, auto_svm = utils.automl_cross_val(utils.OPCODE_NGRAM, utils.SVM, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.OPCODE_NGRAM, utils.SVM, score)

print(f"Best {utils.SVM} model:\n{utils.best_model(auto_svm)}\n")
print(auto_svm.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.OPCODE_NGRAM, utils.RF, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.OPCODE_NGRAM, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())

# Summary
slgn_stats.save()
summary = slgn_stats.df.loc[utils.OPCODE_NGRAM, [utils.BASE, utils.KNN, utils.SVM, utils.RF]].sort_values(ascending=False)
print(f"Summary:\n{summary}\n")
###

### Content Complexity
print("[*] Individual Feature Experiments - Content Complexity")
X = utils.CSVFeature("content_complexity.csv").load()
slgn_stats.new_feature(utils.CMPLXTY, len(X.columns))
slgn_stats.update(utils.CMPLXTY, utils.BASE, base_score)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.CMPLXTY, utils.KNN, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.CMPLXTY, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# SVM
score, auto_svm = utils.automl_cross_val(utils.CMPLXTY, utils.SVM, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.CMPLXTY, utils.SVM, score)

print(f"Best {utils.SVM} model:\n{utils.best_model(auto_svm)}\n")
print(auto_svm.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.CMPLXTY, utils.RF, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.CMPLXTY, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())

# Summary
slgn_stats.save()
summary = slgn_stats.df.loc[utils.CMPLXTY, [utils.BASE, utils.KNN, utils.SVM, utils.RF]].sort_values(ascending=False)
print(f"Summary:\n{summary}\n")
###

### Import Libraries
print("[*] Individual Feature Experiments - Import Libraries")
X = utils.NPZFeature("lib_1grams.npz").load().set_index(labels.index)
slgn_stats.new_feature(utils.IMP_LIB, f"{lib_ngrams.shape[1]} -> {len(X.columns)}")
slgn_stats.update(utils.IMP_LIB, utils.BASE, base_score)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.IMP_LIB, utils.KNN, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.IMP_LIB, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# SVM
score, auto_svm = utils.automl_cross_val(utils.IMP_LIB, utils.SVM, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.IMP_LIB, utils.SVM, score)

print(f"Best {utils.SVM} model:\n{utils.best_model(auto_svm)}\n")
print(auto_svm.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.IMP_LIB, utils.RF, X_train, X_test, y_train, y_test)
slgn_stats.update(utils.IMP_LIB, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())

# Summary
slgn_stats.save()
summary = slgn_stats.df.loc[utils.IMP_LIB, [utils.BASE, utils.KNN, utils.SVM, utils.RF]].sort_values(ascending=False)
print(f"Summary:\n{summary}\n")
###
######

###### Integrated Feature Experiments
intgr_stats = utils.IntegratedScoreStats("integrated_scores.csv")

### PE Section Sizes + PE Section Permissions + Content Complexity
print("[*] Individual Feature Experiments - PE Section Sizes + PE Section Permissions + Content Complexity")
X = utils.integrated_features((utils.SCTN_SIZE, utils.RWE_SIZE, utils.CMPLXTY), labels)
raw_dim = len(sctn_sizes.columns) + int(slgn_stats.df.at[utils.RWE_SIZE, "Dimension"]) + int(slgn_stats.df.at[utils.CMPLXTY, "Dimension"])
intgr_stats.new_feature(utils.SCTN_RWE_CMPLXTY, f"{raw_dim} -> {len(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.SCTN_RWE_CMPLXTY, utils.KNN, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.SCTN_RWE_CMPLXTY, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.SCTN_RWE_CMPLXTY, utils.RF, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.SCTN_RWE_CMPLXTY, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())
###

### PE Section Sizes + PE Section Permissions + Content Complexity + Import Libraries
print("[*] Individual Feature Experiments - PE Section Sizes + PE Section Permissions + Content Complexity + Import Libraries")
X = utils.integrated_features((utils.SCTN_SIZE, utils.RWE_SIZE, utils.CMPLXTY, utils.IMP_LIB), labels)
raw_dim = len(sctn_sizes.columns) + int(slgn_stats.df.at[utils.RWE_SIZE, "Dimension"]) + int(slgn_stats.df.at[utils.CMPLXTY, "Dimension"]) + lib_ngrams.shape[1]
intgr_stats.new_feature(utils.SCTN_RWE_CMPLXTY_LIB, f"{raw_dim} -> {len(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.SCTN_RWE_CMPLXTY_LIB, utils.KNN, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.SCTN_RWE_CMPLXTY_LIB, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.SCTN_RWE_CMPLXTY_LIB, utils.RF, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.SCTN_RWE_CMPLXTY_LIB, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())
###

### File Sizes + API 4-grams + Opcode 4-grams
X = utils.integrated_features((utils.FILE_SIZE, utils.API_NGRAM, utils.OPCODE_NGRAM), labels)
raw_dim = int(slgn_stats.df.at[utils.FILE_SIZE, "Dimension"]) + api_ngrams.shape[1] + opcode_ngrams.shape[1]
intgr_stats.new_feature(utils.FILE_API_OPCODE, f"{raw_dim} → {len(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.FILE_API_OPCODE, utils.KNN, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.FILE_API_OPCODE, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.FILE_API_OPCODE, utils.RF, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.FILE_API_OPCODE, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())
###

### All Features
X = utils.integrated_features((utils.FILE_SIZE, utils.SCTN_SIZE, utils.RWE_SIZE, utils.CMPLXTY, utils.API_NGRAM, utils.OPCODE_NGRAM, utils.IMP_LIB), labels)
raw_dim = int(slgn_stats.df.at[utils.FILE_SIZE, "Dimension"]) + len(sctn_sizes.columns) + int(slgn_stats.df.at[utils.RWE_SIZE, "Dimension"]) + int(slgn_stats.df.at[utils.CMPLXTY, "Dimension"]) + api_ngrams.shape[1] + opcode_ngrams.shape[1] + lib_ngrams.shape[1]
intgr_stats.new_feature(utils.ALL, f"{raw_dim} → {len(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# KNN
score, auto_knn = utils.automl_cross_val(utils.ALL, utils.KNN, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.ALL, utils.KNN, score)

print(f"Best {utils.KNN} model:\n{utils.best_model(auto_knn)}\n")
print(auto_knn.sprint_statistics())

# RF
score, auto_rf = utils.automl_cross_val(utils.ALL, utils.RF, X_train, X_test, y_train, y_test)
intgr_stats.update(utils.ALL, utils.RF, score)

print(f"Best {utils.RF} model:\n{utils.best_model(auto_rf)}\n")
print(auto_rf.sprint_statistics())
###

### Summary
intgr_stats.save()
summary = intgr_stats.df
print(f"Summary:\n{summary}\n")
###
######

###### Summary For All Experiments
stats = intgr_stats.df.copy()
for ftr, row in slgn_stats.df.iterrows():
    stats.at[ftr, "Dimension"] = row["Dimension"]
    stats.at[ftr, "Best Accuracy"] = row[[utils.KNN, utils.SVM, utils.RF]].max()
    stats.at[ftr, "Best Model"] = (utils.KNN, utils.SVM, utils.RF)[row[[utils.KNN, utils.SVM, utils.RF]].values.argmax()]

stats.sort_values(by="Best Accuracy", ascending=False, inplace=True)
stats.to_csv(utils.stats_dir.joinpath("scores.csv")) 
summary = stats.df
print(f"Summary for all expriments:\n{summary}\n")
######
