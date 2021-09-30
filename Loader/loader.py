import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json


class Loader():
    def __init__(self, folder_path, split_pct, save=True):
        self.folder_path = folder_path

        # Init label_df
        self.label_df = pd.read_csv(os.path.join(folder_path, 'data.csv'))
        self.train_path = os.path.join(folder_path, 'train.csv')
        self.val_path = os.path.join(folder_path, 'val.csv')

        _, self.club_ranking_dict = self.init_club_rank()

        f = open(os.path.join(folder_path, "attr_groups.json"), "r")
        self.attribute_group = json.load(f)
        f.close()

        # Preprocess label_df
        self.preprocess()
        self.train_df, self.val_df = self.split_train_val(split_pct)

        self.cat_var = ["Position", "Club_ranking"]

        if save:
            self.label_df.to_csv(os.path.join(folder_path, 'data_final.csv'), index=False)
            self.train_df.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
            self.val_df.to_csv(os.path.join(folder_path, 'val.csv'), index=False)
    
    def group_pos(self, new_col_name, col_group):
        self.label_df[new_col_name] = 0
        for col in col_group:
            self.label_df[new_col_name] += round((self.label_df[col]/len(col_group)), 1)
    
    def count_null(self):
        col_list = self.label_df.columns
        null_stats = {col: 0 for col in col_list}
        for i in null_stats.keys():
            if len(self.label_df[self.label_df[i].isnull()==True]) > 0:
                null_stats[i] = len(self.label_df[self.label_df[i].isnull()==True])
        return null_stats

    def init_club_rank(self):
        club_rankings = pd.read_csv(os.path.join(self.folder_path, "clubs.csv"))
        inval_club = []
        for club in self.label_df["Club"].to_list():
            if club not in club_rankings["Club"].to_list() and club not in inval_club:
                inval_club.append(club)
        invalid_name = {
            "Club": inval_club
        }

        club_ranking_dict = club_rankings.set_index('Club').to_dict()['Overall']

        return invalid_name, club_ranking_dict

    def preprocess_release_clause(self):
        # This function replaced NAN value in release clause with the average release clause 
        # of players with the same rating.
        clause_fill = {'Release Clause': -1}
        df_rm_nan_clause = self.label_df.dropna(subset=['Release Clause'])
        self.label_df = self.label_df.fillna(clause_fill)

        df_rm_nan_clause['Release Clause'] = df_rm_nan_clause['Release Clause'].apply(lambda x : self.convert_to_million(x))

        average_clause_rating = {rating: 0 for rating in list(set(self.label_df['Overall'].to_list()))}
        for rating in average_clause_rating.keys():
            average_clause_rating[rating] = round(df_rm_nan_clause[df_rm_nan_clause["Overall"] == rating]["Release Clause"].mean(), 
                                                1)
        
        clause_list = self.label_df['Release Clause'].to_list()
        overall = self.label_df['Overall'].to_list()

        for i, (ovr, price) in enumerate(zip(overall, clause_list)):
            if price == -1:
                clause_list[i] = str("L"+str(average_clause_rating[ovr])+"M")
        self.label_df = self.label_df.drop(['Release Clause'], axis = 1)
        self.label_df['Release Clause'] = clause_list
        self.label_df['Release Clause'] = self.label_df['Release Clause'].apply(self.convert_to_million)
    
    def split_train_val(self, pct):
        train, val = train_test_split(self.label_df,
                                      test_size=pct,
                                      random_state=30,
                                      shuffle=True)
        return train, val

    def prepare_train(self, one_hot):
        train_prepared, val_prepared = self.train_df, self.val_df
        if one_hot:
            train_prepared[self.cat_var] = train_prepared[self.cat_var].astype("category")
            val_prepared[self.cat_var] = val_prepared[self.cat_var].astype("category")
            train_prepared = pd.get_dummies(train_prepared, columns=self.cat_var)
            val_prepared = pd.get_dummies(val_prepared, columns=self.cat_var)
        return train_prepared, val_prepared


     # Preprocess data
    def preprocess(self):
        position_fill = {pos: "0+0" for pos in self.attribute_group["Position"]}
        # Fill NaN clubs
        position_fill["Club"] = "None"

        # Fill GK NaN attributes such as LS, ST, etc with 0
        self.label_df = self.label_df.fillna(position_fill)
        # Coach
        self.label_df = self.label_df.dropna(subset=['Position'])

        # Reformat Wage and Value
        self.label_df["Wage"] = self.label_df["Wage"].apply(self.convert_to_million)
        self.label_df["Value"] = self.label_df["Value"].apply(self.convert_to_million)

        # Update club rankings
        self.label_df["Club_ranking"] = 0
        self.label_df["Club_ranking"] = self.label_df["Club"].apply(lambda x : self.club_rank(x,
                                                                                        self.club_ranking_dict))

        # Random range of players' positions
        for pos in self.attribute_group["Position"]:
            self.label_df[pos] = self.label_df[pos].apply(self.random_attributes)
        
        # Height
        self.label_df["Height"] = self.label_df["Height"].apply(self.feet_to_m)

        # Weight
        self.label_df["Weight"] = self.label_df["Weight"].apply(lambda x : int(x.split("lbs")[0]))

        # Foot
        self.label_df["Preferred Foot"] = self.label_df["Preferred Foot"].apply(lambda x : 0 if x == "Right" else 1)

        # Work Rate
        self.label_df["Work Rate"] = self.label_df["Work Rate"].apply(self.work_rate)

        # Group attributes
        for k, v in self.attribute_group["Attributes"].items():
            self.group_pos(k, v)
        
        position_df, pos_index = pd.factorize(self.label_df["Position"].to_list())
        self.label_df = self.label_df.drop(['Position'], axis = 1)
        self.label_df['Position'] = position_df

        position_index = {}
        for i, p in enumerate(pos_index):
            position_index[p] = i

        self.attribute_group["Position_index"] = position_index
        self.label_df = self.label_df[self.attribute_group["Keep"]]

        # Fill NaN release clause based rating
        self.preprocess_release_clause()

        # Update Json File
        f = open(os.path.join(self.folder_path, "attr_groups.json"), "w")
        json.dump(self.attribute_group, f)
        f.close()
        
    @staticmethod
    def work_rate(work_rate_att):
        work_dict = {
            "Low": 0.0,
            "Medium": 1.0,
            "High": 2.0
        }   
        work_att_lst = work_rate_att.split('/ ')
        return (work_dict[work_att_lst[0]]+work_dict[work_att_lst[1]])/2
    
    @staticmethod
    def feet_to_m(height):
        height = height.split("\'")
        return round((float(height[0])+float(height[1])/10)*0.3048, 2)

    @staticmethod
    def convert_to_million(value_string):
        if "K" in value_string:
            value = float(value_string[1:-1])/1000
        elif "M" in value_string:
            value = float(value_string[1:-1])
        else:
            value = float(value_string[1:])
        return value

    @staticmethod
    def club_rank(club_name, club_overall_dict):
        # Rank 0: Small clubs (60-70)
        # Rank 1: Average Clubs (70-80)
        # Rank 3: Elite CLubs (>=80)    
        rank = 0
        if club_name not in club_overall_dict.keys():
            return rank
        else:
            if club_overall_dict[club_name] < 70:
                rank = 0
            elif club_overall_dict[club_name] < 80:
                rank = 1
            else:
                rank = 3
        return rank

    @staticmethod
    def random_attributes(attr):
        attr_split = attr.split("+")
        attr_value = [int(i) for i in attr_split]
        return np.random.randint(low=attr_split[0], high=sum(attr_value)+1)


if __name__ == '__main__':
    loader = Loader("/Users/mac/Projects/fifa19/data", 0.2, True)
    x, y = loader.prepare_train(True)