import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import subprocess


class EpitopeDataSlim:

    def __init__(self, epitopes, folder, duplicate=False, predict=False, output='output',
                 prediction_path='prediction.csv'):
        self.epitopes = epitopes
        self.data = self.filter_data(self.read(f'{folder}/vdjdb.slim.txt'))
        self.duplicate = self.add_duplicates() if duplicate else None
        self.make_prediction = predict
        self.output = output
        self.prediction_path = prediction_path

    def read(self, data):
        data = pd.read_csv(data, sep='\t')
        return data

    def filter_data(self, data, hla='HLA-A*02', species='HomoSapiens'):
        mapper = {'cdr3': 'CDR3', 'antigen.epitope': 'Antigen', 'mhc.a': 'HLA'}
        data = data.loc[(data.species == species) & (data['antigen.epitope'].isin(self.epitopes)) &
                        (data['mhc.a'].str.startswith(hla))]

        data = data[['cdr3', 'antigen.epitope', 'mhc.a']].rename(mapper=mapper, axis=1)

        # removing 'HLA-' part as in testing_data.csv from pMTnet
        data.HLA = data.HLA.str[4:].str.split(',').apply(lambda x: 'A*02:01' if 'A*02:01' in x else 'A*02')
        data.drop_duplicates(subset=['CDR3', 'Antigen'], inplace=True)
        return data

    def save_data(self, name, index=False):
        self.data.to_csv(name, index=index)

    def add_duplicates(self):
        duplicate_data = []
        for cdr, cdr_data in self.data.groupby('CDR3'):
            if len(cdr_data) > 1:
                continue
            for x in cdr_data.itertuples(index=False):
                duplicate_data.append({'CDR3': x.CDR3, 'Antigen': (set(self.epitopes) - {x.Antigen}).pop(),
                                       'HLA': x.HLA})
        return pd.DataFrame(data=duplicate_data)

    def predict(self, input_data):
        command = f'python pMTnet.py -input {input_data} -library library -output {self.output} ' \
                  f'-output_log {self.output}/output.log'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, errors = process.communicate()
        if errors:
            print(errors)
        if output:
            print(output)

    def prepare_prediction(self):
        merge_list = ['CDR3', 'Antigen', 'HLA']

        if len(self.duplicate):
            self.duplicate['label'] = 0
            self.data = self.data.merge(self.duplicate, left_on=merge_list, right_on=merge_list, how='outer')
            self.data.loc[self.data.label != 0, 'label'] = 1
        else:
            self.data['label'] = 1

        name = '_'.join(self.epitopes)
        name = f'duplicate_{name}.csv'
        if not len(self.duplicate):
            name = 'no_' + name
        self.save_data(name=name)
        if self.make_prediction:
            self.predict(name)
        output = pd.read_csv(f'{self.output}/{self.prediction_path}')
        self.data = self.data.merge(output.drop('Unnamed: 0', axis=1), left_on=merge_list, right_on=merge_list,
                                    how='outer')
        self.save_data(name=name)

    def roc(self, minus_rank=True):
        i = 0
        fig, axes = plt.subplots(1, len(self.epitopes), figsize=(4 * len(self.epitopes), 2 * len(self.epitopes)))

        self.prepare_prediction()
        title = 'Using' if len(self.duplicate) else 'Not using'
        plt.suptitle(f'{title} data with duplicates')
        for epitope, data in self.data.groupby('Antigen'):
            ax = axes[i]
            rank = 1 - data.Rank if minus_rank else data.Rank
            fpr, tpr, _ = roc_curve(data.label, rank)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', c='darkorange')
            ax.set_title(epitope)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.legend(loc='lower right')
            i += 1
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
