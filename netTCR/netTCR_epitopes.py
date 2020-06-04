import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import subprocess


class EpitopeSlimNetTCR:

    def __init__(self, epitopes, chain, folder, duplicate=True, predict=False, prediction_path='prediction.csv'):
        """

        Args:
            chain (str): specifies which chain to use
            epitopes (list): list of epitopes to plot ROC and predict
            folder (str): path to folder of data file (now it has a default name)
            duplicate (bool): whether to add duplicates to data (will be removed and will stay True)
            predict (bool): whether there is a need to predict values for dataset
            output (str): output folder for prediction
            prediction_path (str): path to file with predictions
        """
        self.epitopes = epitopes
        self.chain = chain
        self.data = self.filter_data(self.read(f'{folder}/vdjdb.slim.txt'))
        self.duplicate = self.add_duplicates() if duplicate else None
        self.make_prediction = predict
        self.prediction_path = prediction_path

    def read(self, data):
        """
        Reading data for one chain
        Args:
            data (str): path to data file

        Returns:
            pd.DataFrame with read data for one chain
        """
        data = pd.read_csv(data, sep='\t')

        # filtering by chain
        return data.loc[data.gene == self.chain]

    def filter_data(self, data, hla='HLA-A*02', species='HomoSapiens'):
        """
        This function filters HLA and species and mostly prepares data for pMTnet input.
        Args:
            data (pd.DataFrame): dataframe for one chain
            hla (str): HLA that should be kept
            species (str): species to keep

        Returns:
            pd.DataFrame ready for pMTnet
        """

        # this is a mapping to rename columns
        mapper = {'cdr3': 'CDR3', 'antigen.epitope': 'Antigen'}

        # filtering data by species, HLA and epitopes
        data = data.loc[(data.species == species) & (data['antigen.epitope'].isin(self.epitopes)) &
                        (data['mhc.a'].str.contains(hla, regex=False))]

        # renaming columns
        data = data[['cdr3', 'antigen.epitope']].rename(mapper=mapper, axis=1)

        # removing duplicated data
        data.drop_duplicates(subset=['Antigen', 'CDR3'], inplace=True)
        return data

    def save_data(self, name, index=False, header=True, subset=None, sep=','):
        """
        This function saves a dataframe with a given name
        Args:
            subset (list):
            header (bool): #TODO
            name (str): name with which a dataframe will be saved
            index (bool): flag to keep index
        """
        if subset:
            self.data[subset].to_csv(name, index=index, header=header, sep=sep)
        else:
            self.data.to_csv(name, index=index, header=header, sep=sep)

    def add_duplicates(self):
        """
        Creating a dataframe with false rows for prediction

        Returns:
            pd.DataFrame with only false (negative) duplicated data
        """
        # all duplicates will be stored here
        duplicate_data = []

        # for each CDR3
        for cdr, cdr_data in self.data.groupby('CDR3'):

            # if there are already all duplicates, then skipping this CDR
            if len(cdr_data) == len(self.epitopes):
                continue

            # else for each row with CDR - Antigen
            for x in cdr_data.itertuples(index=False):

                # a set of epitopes which are missing
                # TODO: this step may add exhaustive duplicates which are removed in the end
                rest_epitopes = set(self.epitopes) - {x.Antigen}

                # for each epitope that is missing (see TODO)
                for epitope in rest_epitopes:

                    # adding a dict with the current CDR3 and false epitope
                    duplicate_data.append({'CDR3': x.CDR3, 'Antigen': epitope})
        return pd.DataFrame(data=duplicate_data).drop_duplicates()[['Antigen', 'CDR3']]

    def predict(self, input_data):
        """
        Running prediction by pMTnet on prepared data
        Args:
            input_data (str): path to input data
        """

        # command line exec
        command = f'python scripts/netTCR.py -infile {input_data} -outfile {self.prediction_path}'

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, errors = process.communicate()
        if errors:
            print(errors)
        if output:
            print(output)

    def prepare_prediction(self):
        """
        Saving a file with labels and predictions
        """

        # a list of feature names to merge on
        merge_list = ['Antigen', 'CDR3']

        # if there are duplicates prepared
        if len(self.duplicate):

            # setting label to 0 as these rows are false
            self.duplicate['label'] = 0

            # merging true data with duplicates
            self.data = self.data.merge(self.duplicate, left_on=merge_list, right_on=merge_list, how='outer')

            # labeling all true rows with 1
            self.data.loc[self.data.label != 0, 'label'] = 1

        # if there are no duplicates
        # TODO: remove this branch
        else:
            # all rows are true, so setting label to 1
            self.data['label'] = 1

        # creating a name to save the dataset
        name = '_'.join(self.epitopes)
        name = f'duplicate_{name}_{self.chain}.csv'
        if not len(self.duplicate):
            name = 'no_' + name

        # saving data
        self.save_data(name=name, header=False, subset=['Antigen', 'CDR3'], sep='\t')

        # if there is a need to predict by pMTnet
        if self.make_prediction:
            self.predict(name)

        # reading prediciton data
        output = pd.read_csv(f'{self.prediction_path}', header=1, sep='\t', skipfooter=1)

        mapper = {'peptide': 'Antigen', 'tcr': 'CDR3'}

        output = output.rename(mapper=mapper, axis=1)

        # merging data with labels and prediction scores
        self.data = self.data.merge(output, left_on=merge_list, right_on=merge_list,
                                    how='outer')

        # saving this data
        self.save_data(name=name)

    def roc(self):
        """
        Plotting ROC by epitope
        """
        i = 0

        # creating subplots for each epitope. They will be plotted in one line
        fig, axes = plt.subplots(1, len(self.epitopes), figsize=(4 * len(self.epitopes), 2 * len(self.epitopes)))

        # getting true labels and predicted scores
        self.prepare_prediction()

        # creating a title
        title = 'Using' if len(self.duplicate) else 'Not using'
        plt.suptitle(f'{title} data with duplicates, {self.chain}')

        # for each epitope
        for epitope, data in self.data.groupby('Antigen'):
            ax = axes[i]

            # predicted rank = 1 - probability of binding
            rank = data.prediction

            fpr, tpr, _ = roc_curve(data.label, rank)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})\nPositive = {len(data.loc[data.label == 1])}\n'
                                    f'Negative = {len(data.loc[data.label == 0])}',
                    c='darkorange')
            ax.set_title(epitope)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.legend(loc='lower right')
            i += 1
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
