from numpy.lib.function_base import average
import pandas as pd
import numpy as np
# from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def group_pos(df, new_col_name, col_group):
    df[new_col_name] = 0
    for col in col_group:
        df[new_col_name] += round((df[col]/len(col_group)), 1)
    return df

def random_attributes(attr):
    attr_split = attr.split("+")
    attr_value = [int(i) for i in attr_split]
    return np.random.randint(low=attr_split[0], high=sum(attr_value)+1)

def work_rate(work_rate_att):
    work_dict = {
        "Low": 0.0,
        "Medium": 1.0,
        "High": 2.0
    }   
    work_att_lst = work_rate_att.split('/ ')
    return (work_dict[work_att_lst[0]]+work_dict[work_att_lst[1]])/2

def feet_to_m(height):
    height = height.split("\'")
    return round((float(height[0])+float(height[1])/10)*0.3048, 2)

def count_null(df):
    col_list = df.columns
    null_stats = {col: 0 for col in col_list}
    for i in null_stats.keys():
        if len(df[df[i].isnull()==True]) > 0:
            null_stats[i] = len(df[df[i].isnull()==True])
    return null_stats

def convert_to_million(value_string):
    if "K" in value_string:
        value = float(value_string[1:-1])/1000
    elif "M" in value_string:
        value = float(value_string[1:-1])
    else:
        value = float(value_string[1:])
    return value


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


def preprocess_release_clause(df):
    # This function replaced NAN value in release clause with the average release clause 
    # of players with the same rating.
    clause_fill = {'Release Clause': -1}
    df_rm_nan_clause = df.dropna(subset=['Release Clause'])
    df = df.fillna(clause_fill)

    df_rm_nan_clause['Release Clause'] = df_rm_nan_clause['Release Clause'].apply(lambda x : convert_to_million(x))

    average_clause_rating = {rating: 0 for rating in list(set(df['Overall'].to_list()))}
    for rating in average_clause_rating.keys():
        average_clause_rating[rating] = round(df_rm_nan_clause[df_rm_nan_clause["Overall"] == rating]["Release Clause"].mean().item(), 
                                              1)
    
    clause_list = df['Release Clause'].to_list()
    overall = df['Overall'].to_list()

    for i, (ovr, price) in enumerate(zip(overall, clause_list)):
        if price == -1:
            clause_list[i] = str("L"+str(average_clause_rating[ovr])+"M")
    df = df.drop(['Release Clause'], axis = 1)
    df['Release Clause'] = clause_list
    df['Release Clause'] = df['Release Clause'].apply(convert_to_million)

    return df


if __name__ == "__main__":
    # Read data
    label_df = pd.read_csv("/Users/mac/Projects/fifa19/data.csv")

    # Columns to drop due to irrelavance
    # We drop Potential since it is unconvincing to predict overall rating when given potential rating
    # We drop Loaned_from due to too many NULL value from the dataset
    position_list = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", 
                    "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", 
                    "RWB", "LB", "LCB", "CB", "RCB", "RB"]

    goalie = ['GK']
    defense_list = ['LWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
    mid_list = ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LDM', 'CDM', 'RDM', 'RW', 'LW']
    forward_list = ["LS", "ST", "RS", "LF", "CF", "RF"]

    # TODO: Save to json/yaml file for convenience
    attribute_group_dict = {
        "Pace": ["Acceleration", "SprintSpeed"],
        "Dribble": ["Agility", "Balance", "BallControl", "Composure", "Dribbling", "Reactions"],
        "Shoot": ["Finishing", "LongShots", "Penalties", "Positioning", "ShotPower", "Volleys", "FKAccuracy"],
        "Defense": ["HeadingAccuracy", "Interceptions", "Marking", "SlidingTackle", "StandingTackle"],
        "Pass": ["Crossing", "Curve", "FKAccuracy", "LongPassing", "ShortPassing", "Vision"],
        "Physical": ["Aggression", "Jumping", "Stamina", "Strength"],
        "Technical": ["Finishing", "LongShots", "Penalties", "ShotPower", "Volleys", 
                "Crossing", "Curve", "FKAccuracy", "LongPassing", "ShortPassing", "BallControl", 
                "Dribbling", "HeadingAccuracy", "Marking", "SlidingTackle", "StandingTackle"],
        "Mental": ["Positioning", "Vision", "Composure", "Interceptions", "Aggression"],
        "Physical": ["Acceleration", "SprintSpeed", "Agility", "Balance", "Reactions", "Jumping", "Stamina", "Strength"],
        "Goalkeeping_rating": ["GKDiving", "GKHandling", "GKKicking", "GKPositioning", "GKReflexes"],
        "Defensive_rating": ['LWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'],
        "Midfielder_rating": ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LDM', 'CDM', 'RDM', 'RW', 'LW'],
        "Offensive_rating": ["LS", "ST", "RS", "LF", "CF", "RF"]
    }

    # Fill NaN position for GK
    # We put NaN as O+0 for the sake of consistency
    position_fill = {pos: "0+0" for pos in position_list}
    # Fill NaN clubs
    position_fill["Club"] = "None"


    # Fill GK NaN attributes such as LS, ST, etc with 0
    label_df = label_df.fillna(position_fill)
    # Coach
    label_df = label_df.dropna(subset=['Position'])

    # Reformat Wage and Value
    label_df["Wage"] = label_df["Wage"].apply(convert_to_million)
    label_df["Value"] = label_df["Value"].apply(convert_to_million)

    # print(type(label_df[label_df['Release Clause'].isnull()==True]['Overall'].mean().item()))
    # print(count_null(label_df))

    # Find NaN club names
    club_rankings = pd.read_csv("/Users/mac/Projects/fifa19/clubs.csv")
    inval_club = []
    for club in label_df["Club"].to_list():
        if club not in club_rankings["Club"].to_list() and club not in inval_club:
            inval_club.append(club)

    invalid_name = {
        "Club": inval_club
    }

    invalid_clb = pd.DataFrame(invalid_name)
    invalid_clb.to_csv("inval.csv", index=False)

    # Handle Club and Categorize Club rankings
    club_ranking_dict = club_rankings.set_index('Club').to_dict()['Overall']
    label_df["Club_ranking"] = 0
    label_df["Club_ranking"] = label_df["Club"].apply(lambda x : club_rank(x, club_ranking_dict))

    # Random range of players' positions
    for pos in position_list:
        label_df[pos] = label_df[pos].apply(random_attributes)
    
    # Height
    label_df["Height"] = label_df["Height"].apply(feet_to_m)

    # Weight
    label_df["Weight"] = label_df["Weight"].apply(lambda x : int(x.split("lbs")[0]))

    # Foot
    label_df["Preferred Foot"] = label_df["Preferred Foot"].apply(lambda x : 0 if x == "Right" else 1)

    # Work Rate
    label_df["Work Rate"] = label_df["Work Rate"].apply(work_rate)

    # Group attributes
    for k, v in attribute_group_dict.items():
        label_df = group_pos(label_df, k, v)

    # TODO: Sort nationality based on ranking
    # Temporariy delete column nationality from table

    # Position Group:
    # 0: Defense, 1: Midfielder, 2: FW
    # label_df["Position Group"] = label_df["Position"].apply(position_group)

    # Temporarily factorize specific position
    position_df, pos_index = pd.factorize(label_df["Position"].to_list())
    label_df = label_df.drop(['Position'], axis = 1)
    label_df['Position'] = position_df

    position_index = {}
    for i, p in enumerate(pos_index):
        position_index[p] = i
    
    keep_cols = ["Age", "Overall", "Position", "Value", "Wage", "Special", "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                "Height", "Weight", "Release Clause", "Club_ranking", "Pace", "Dribble", "Shoot", "Defense", "Pass", "Physical",
                "Technical", "Mental", "Goalkeeping_rating", "Defensive_rating", "Midfielder_rating", "Offensive_rating"]
    label_df = label_df[keep_cols]

    # Fill NaN release clause based rating
    label_df = preprocess_release_clause(label_df)
    
    # Simple Correlation matrix
    corr = label_df.corr()
    corr.to_csv("./correlation.csv", index=True)
    label_df.to_csv("./data_fifa_19.csv", index=False)

    # Split train_val
    train, val = train_test_split(label_df, test_size=0.2, random_state=3042, shuffle=True)

    train.to_csv("./train.csv", index=False)
    val.to_csv("./val.csv", index=False)
