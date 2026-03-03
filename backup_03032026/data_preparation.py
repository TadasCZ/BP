import numpy as np
import pandas as pd
from scipy import stats
from pandas_plink import read_plink
from phenotype_levels import phenotype_levels_used

def snp_matrix(prefix):
    (bim, fam, G) = read_plink(prefix, verbose=False)

    X = G.compute().T.astype(np.float32) # sample x snp
    missing_count = np.isnan(X).sum()
    print(f"Celkem chybějících genotypů: {missing_count}")

    # nahrazení NaN za modus
    if missing_count > 0:
        modes = stats.mode(X, axis=0, nan_policy='omit', keepdims=True).mode[0]
        modes = np.nan_to_num(modes, nan=0)
        
        inds = np.where(np.isnan(X))
        X[inds] = np.take(modes, inds[1])
    
    X = X.astype(np.int8)
    print("Genotype matrix shape:", X.shape)

    return X # bim, fam

def prepare_phenotype_data(pheno_path, fam_path, phenotype):
    pheno = pd.read_csv(pheno_path, sep="\t")
    fam = pd.read_csv(
        fam_path, 
        sep=r"\s+", 
        header=None, 
        names=["FID", "IID", "father", "mother", "sex", "trait"]
    )
    
    pheno_subset = pheno[["3K_DNA_IRIS_UNIQUE_ID", phenotype]].rename(
        columns={"3K_DNA_IRIS_UNIQUE_ID": "IID", phenotype: "VALUE"}
    )
    
    fam["IDX"] = range(len(fam))

    merged_data = fam[["IID", "IDX"]].merge(pheno_subset, on="IID").dropna(subset=["VALUE"])

    merged_data = merged_data[["IDX", "IID", "VALUE"]]
    merged_data["VALUE"] = merged_data["VALUE"].astype(int)

    print(f"Počet vzorků '{phenotype}' po filtraci: {len(merged_data)}")
    
    return merged_data

def analyze_phenotype_distribution(df, phenotype_name):
    print(f"\nDistribuce hodnot pro: {phenotype_name}")

    counts = df["VALUE"].value_counts().sort_index()
    print(counts)
    
    total = len(df)
    majority_count = counts.max()
    apriori_acc = majority_count / total
    
    print(f"Celkem: {total} vzorků")
    print(f"Apriori accuracy (majority class): {apriori_acc:.4f}")
    
    return apriori_acc

def onehot_encode_phenotype(df, levels, prefix):
    onehot_data = pd.get_dummies(df["VALUE"], prefix=prefix, dtype=np.float32)

    for lvl in levels:
        col = f"{prefix}_{lvl}"
        if col not in onehot_data.columns:
            onehot_data[col] = 0

    column_order = [f"{prefix}_{l}" for l in levels]
    onehot_data = onehot_data[column_order]

    final_df = pd.concat([
        df[["IDX", "IID"]].reset_index(drop=True), 
        onehot_data.reset_index(drop=True)
    ], axis=1)

    print(f'\nOne-hot kódování provedeno.')

    return final_df

def get_clean_labels(df_onehot, levels, prefix):
    label_cols = []
    for l in levels:
        label_cols.append(prefix + "_" + str(l))
    y = df_onehot[label_cols].values
    
    print(f"Tvar matice fenotypů: {y.shape}")
    print(y)
    return y

def align_genotypes_to_phenotypes(X, df_pheno):
    keep_indices = df_pheno["IDX"].values

    print(f"\nPůvodní počet vzorků v X: {X.shape[0]}")
    print(f"Počet vzorků po filtraci: {len(keep_indices)}")
    print(f"Prvních 10 vybraných indexů řádků: {keep_indices[:10]}")
    
    X_filtered = X[keep_indices, :]
    
    print(f"Nová velikost matice X: {X_filtered.shape}")
    
    return X_filtered

def onehot_encode_genotypes(X):
    num_samples, num_snps = X.shape
    X_onehot = np.zeros((num_samples, num_snps, 3), dtype=np.int8)
    
    for val in [0, 1, 2]:
        X_onehot[:, :, val] = (X == val)

    return X_onehot.reshape(num_samples, -1)

def run_phenotype_pipeline(phenotype_name, X_full_matrix, pheno_path, fam_path):
    if phenotype_name not in phenotype_levels_used:
        print(f"Fenotyp {phenotype_name} neexistuje.")
        return None
    
    levels = phenotype_levels_used[phenotype_name]
    
    # připravím si jen hodnoty jednoho fenotypu
    df_pheno = prepare_phenotype_data(pheno_path, fam_path, phenotype_name)
    
    # analýza phenotypu
    apriori_acc = analyze_phenotype_distribution(df_pheno, phenotype_name)
    
    # převedení fenotypových dat na onehot
    df_onehot = onehot_encode_phenotype(df_pheno, levels, phenotype_name)
    #print(df_onehot.dtypes)

    # připravené čisté fenotypy
    y_final = get_clean_labels(df_onehot, levels, phenotype_name)
    
    # seřazené vzorky dle fenotypů
    X_ready_for_onehot = align_genotypes_to_phenotypes(X_full_matrix, df_pheno)
    
    # genotypy na onehot
    #X_onehot_final = onehot_encode_genotypes(X_ready_for_onehot)
    
    return X_ready_for_onehot, y_final, levels, apriori_acc