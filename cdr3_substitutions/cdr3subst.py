import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import MDS
from collections import defaultdict, Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot, init_notebook_mode

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_2_POS = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
AA_N = 20

def read_slim_df(f):
    vdjdb_slim_df = pd.read_csv(f, sep='\t')
    
    # Filter referencies from 10xgenomics.com
    vdjdb_slim_df = vdjdb_slim_df[~vdjdb_slim_df['reference.id'].str.startswith('https://www.10xgenomics.com')]

    # Filter bad positions of j.start and v.end
    vdjdb_slim_df = vdjdb_slim_df[(vdjdb_slim_df['v.end'] < vdjdb_slim_df['j.start']) & (vdjdb_slim_df['v.end'] > 0)]

    # add CDR3 len
    vdjdb_slim_df['cdr3_len'] = vdjdb_slim_df.cdr3.str.len()
    return vdjdb_slim_df

def filter_good_epitopes(vdjdb_slim_df):
    ept_df = vdjdb_slim_df.groupby(['species', 'gene', 'antigen.epitope', 'cdr3_len'])['cdr3'].count().reset_index()
    ept_df.columns = ['species', 'gene', 'antigen.epitope', 'cdr3_len', 'samples_cnt']
    good_epitopes = ept_df[ept_df.samples_cnt >= 30][['species', 'gene', 'antigen.epitope', 'cdr3_len', 'samples_cnt']]
    vdjdb_slim_df.merge(good_epitopes, how='inner', suffixes=['', '_r'],
                                    left_on=['species', 'gene', 'antigen.epitope', 'cdr3_len'],
                                    right_on=['species', 'gene', 'antigen.epitope', 'cdr3_len'])
    return vdjdb_slim_df, good_epitopes

def filter_slim_df(df, species, epitopes, chains, cdr3_len):
    return df[(df['species'].isin(species)) &
              (df['antigen.epitope'].isin(epitopes)) &
              (df['gene'].isin(chains)) &
              (df['cdr3_len'] == cdr3_len)].copy()

def get_concervative_pos(df, q, vend_tag='v.end', jstart_tag='j.start'):
    start_pos = np.percentile(df[vend_tag], q=q, interpolation='nearest')
    end_pos = np.percentile(df[jstart_tag], q=(100-q), interpolation='nearest') - 1
    return start_pos, end_pos

def get_blosum_matrix(df, start_pos, end_pos):
    cdr3_full = df.cdr3.apply(lambda x: pd.Series(list(x))).values
    cdr3 = cdr3_full[:, start_pos:(end_pos+1)]

    # Frequency table
    F_ij = np.zeros([AA_N, AA_N])
    for j in range(0, cdr3.shape[1]):
        column_acids = cdr3[:, j]
        cnt = Counter(column_acids)
        for i_acid, i in AA_2_POS.items():
            for j_acid, j in AA_2_POS.items():
                if (i_acid == j_acid) and cnt[i_acid] > 1:
                    F_ij[i, j] += ((cnt[i_acid] * (cnt[i_acid] - 1)) / 2)
                else:
                    F_ij[i, j] += cnt[i_acid] * cnt[j_acid]
    
    # Observed probability
    lower_Tr = np.tril(np.ones([AA_N, AA_N]))
    np.fill_diagonal(lower_Tr, 1)
    total_pairs = np.sum(lower_Tr * F_ij)
    Q_ij = F_ij / total_pairs

    # Expected probability
    P_i = np.zeros(AA_N)
    for i_acid, i in AA_2_POS.items():
        P_i[i] = Q_ij[i, i] + (np.sum(Q_ij[i, :]) - Q_ij[i, i])/2
    E_ij = np.zeros([AA_N, AA_N])
    for i in np.arange(0, AA_N):
        for j in np.arange(0, AA_N):
            if i == j:
                E_ij[i, j] = P_i[i] * P_i[i]
            else:
                E_ij[i, j] = 2 * P_i[i] * P_i[j]
    
    # The log-odds ration
    L = np.zeros([AA_N, AA_N])
    for i in np.arange(0, AA_N):
        for j in np.arange(0, AA_N):
            if (Q_ij[i, j] == 0) or (E_ij[i, j] == 0):
                L[i, j] = 1
            else:
                L[i, j] = Q_ij[i, j] / E_ij[i, j]
    L = np.round(np.log2(L) * 2)
    return L

def build_all_matrices(vdjdb_slim_df, good_epitopes):
    matrices = {}
    for r in good_epitopes.to_records():
        species, chain, epitope, cdr3_len = r[1], r[2], r[3], r[4]
        df = filter_slim_df(vdjdb_slim_df, [species], [epitope], [chain], cdr3_len)
        if chain == 'TRA':
            start_pos, end_pos = get_concervative_pos(df, 62)
        else:
            start_pos, end_pos = get_concervative_pos(df, 80)
        if (end_pos - start_pos + 1) == 0:
            continue
        L = get_blosum_matrix(df, start_pos, end_pos)
        result = pd.DataFrame(L, index=AMINO_ACIDS, columns=AMINO_ACIDS)
        matrices[(species, chain, epitope, cdr3_len)] = result
    
    print(f"Total good epitoped: {good_epitopes.shape[0]}")
    print(f"Selected epitopes with conservative region: {np.round(len(matrices) / good_epitopes.shape[0] * 100, 2)}%")
    return matrices
    
def get_avg_matrix(matrices, epitopes, epitope=None, species='HomoSapiens', gene='TRB', verbose=False): 
    if epitope:
        selected_epitopes = epitopes[(epitopes['antigen.epitope'] == epitope) & (
            epitopes.species == species) & (epitopes.gene == gene)].values
    else:
        selected_epitopes = epitopes[(epitopes.species == species) & (epitopes.gene == gene)].values
        
    avg_matrix = pd.DataFrame(index=AMINO_ACIDS, columns=AMINO_ACIDS).fillna(0)
    found_num = 0
    for species, chain, epitope, e_len, n_samples in selected_epitopes:
        if (species, chain, epitope, e_len) in matrices:
            avg_matrix += matrices[(species, chain, epitope, e_len)]
            found_num += 1
    avg_matrix = (avg_matrix / found_num).round()
    if verbose:
        print(f"{gene}, epitope: {epitope}. Found {found_num} variants of CDR3 length")
    return avg_matrix

def spearman_corr(M1, M2):
    return stats.spearmanr(M1.values.flatten(), M2.values.flatten())

def pearson_corr(M1, M2):
    return stats.pearsonr(M1.values.flatten(), M2.values.flatten())

IMGT_classes = [
    ('acidic', ['D', 'E']),
    ('aliphatic', ['A', 'I', 'L', 'V']),
    ('amide', ['N', 'Q']),
    ('aromatic', ['F', 'W', 'Y']),
    ('basic', ['H', 'K', 'R']),
    ('G', ['G']),
    ('hydroxyl', ['S', 'T']),
    ('P', ['P']),
    ('sulfur', ['C', 'M'])
]

IMGT_colors =  [
    '#1f77b4', '#ff7f0e', '#2ca02c',
    '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22'
]
    
def get_mds_imgt_traces(matrice, showlegend=True):
    mds = MDS(n_components=2, dissimilarity='euclidean')
    components = mds.fit_transform(matrice.values)
    aa2pos = {aa: pos for pos, aa in enumerate(matrice.columns)}
    class2x = defaultdict(list)
    class2y = defaultdict(list)
    for imgt_class, aa_v in IMGT_classes:
        for aa in aa_v:
            if aa in aa2pos:
                x, y = components[aa2pos[aa]]
                class2x[imgt_class].append(x)
                class2y[imgt_class].append(y)
    traces = []
    for num, (imgt_class, aa_v) in enumerate(IMGT_classes):
        if imgt_class in class2x:
            traces.append(go.Scatter(x=class2x[imgt_class], y=class2y[imgt_class],
                                     mode='markers+text', text=aa_v,
                                     marker={'size': 40, 'opacity':0.5},
                                     line={'color': IMGT_colors[num]},
                                     name=imgt_class,
                                     showlegend=showlegend))
    return traces

def plot_mds_IMGT(matrice2params, rows, cols, titles, exclude=None,
                  width=None, height=None, hspacing=0.05, vspacing=0.1,
                  bmargin=10, tmargin=25, lmargin=10, rmargin=10):
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=titles,
                        horizontal_spacing=hspacing,
                        vertical_spacing=vspacing)
    for matrice, row, col in matrice2params:
        if exclude is not None:
            aa_columns = [a for a in AMINO_ACIDS if a not in exclude]
            matrice = matrice[aa_columns].loc[matrice.index.isin(aa_columns)]
        if row == 1 and col == 1:
            traces = get_mds_imgt_traces(matrice)
        else:
            traces = get_mds_imgt_traces(matrice, showlegend=False)
        for t in traces:
            fig.append_trace(t, row=row, col=col)
    for row in range(rows):
        for col in range(cols):
            pass
#             fig.update_xaxes(title_text="mds_x", row=row+1, col=col+1)
#             fig.update_yaxes(title_text="mds_y", row=row+1, col=col+1)
    fig.update_traces(textfont_size=24)
    fig.update_layout(legend_orientation="h",
                      plot_bgcolor='rgb(248,248,248)',
                      margin=dict(l=lmargin, r=rmargin, b=bmargin, t=tmargin))
    if width and height:
        fig.update_layout(autosize=False, width=width, height=800)
    iplot(fig)