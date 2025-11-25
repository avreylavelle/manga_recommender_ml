# Dataset Operations
import pandas as pd
import os 
import json

from utils.lookup import get_all_unique
from utils.parsing import parse_list

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

USER_PATH = os.path.join(DATASET_DIR, "user_data.csv")
CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_manga_entries.csv")
INPUT_PATH = os.path.join(DATASET_DIR, "manga_entries.csv")
ML_PATH = os.path.join(DATASET_DIR, "ml_cleaned_manga_entries.csv")
TESTER_PATH = os.path.join(DATASET_DIR, "test_ml_dataset.csv")
FT_IMP_PATH = os.path.join(DATASET_DIR, "feature_importance.json")

# The three following functions are for cleaning / loading all of the csv files needed.
def clean_manga_dataset(input_path=INPUT_PATH, output_path=CLEANED_PATH):

    df = pd.read_csv(input_path)

    #Remove columns 13 (german name), 14 (french name), 15 (spanish name) since they had mixed data types and are not needed
    #Remove columns 26 (description) and 27 (background) since they contain large text data that is not needed for analysis 
    df.drop(df.columns[[12, 13, 14, 25, 26]], axis=1, inplace=True)

    #Save cleaned dataframe to new CSV file, rewrite existing cleaned file if it exists
    df.to_csv(output_path, index=False)

    return df

def clean_ml_manga_dataset(input_path=INPUT_PATH, output_path=ML_PATH):

    df = pd.read_csv(input_path)

    # Remove columns 2 and 3 (link and name) not needed for ml
    # Remove columns 10 (Synonyms), 11 (Japanese Name) 12 (English Name), 13 (german name), 14 (french name), 15 (spanish name) 
    # Remove columns 16, 17, 18, 19, 20(Chapters/status/publishing/Authors) - Cannot be used for determining
    # Remove column  25 (Demographic), not used for this (Can be filtered back in later)
    # Remove columns 26 (description) and 27 (background) since they contain large text data that is not needed for analysis 
    df.drop(df.columns[[1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26]], axis=1, inplace=True)

    # Expand columns 21, 22, 23, 24 (Authors, Serialization, Genres, Themes) into separate features
    #Save cleaned dataframe to new CSV file, rewrite existing cleaned file if it exists
    df.to_csv(output_path, index=False)

    return df

# turn lists into individual features
# Loads the cleaned data ^^^
def load_data(cleaned_path=CLEANED_PATH):
    if not os.path.exists(cleaned_path):
        clean_manga_dataset() # clean it
    df = pd.read_csv(cleaned_path) # load it

    return df 

# Load the user_data.csv
def load_user(user_path = USER_PATH):
    if not os.path.exists(user_path): # if it doesnt exist
        df = pd.DataFrame(columns=[ # create the headers for the file
            "username",
            "age",
            "gender",
            "preferred_genres",   
            "preferred_themes",   
            "read_manga" # todo At some point, id like to add a flag for if it was recommened by us       
        ])
        df.to_csv(user_path, index=False) # return it (will be blank if it was just made)
        return df
    # shouldnt be blank, but it could be if no user has been made yet
    return pd.read_csv(user_path)

def initialize_ml_dataset(ml_manga_df):

    df = ml_manga_df.copy()

    # authors = get_all_unique(df, "authors")
    serialization = get_all_unique(df, "serialization")
    genres = get_all_unique(df, "genres")
    themes = get_all_unique(df, "themes")
    demographic = get_all_unique(df, "demographic")

    feature_set = [
        # ("authors", authors, "author"),
        ("serialization", serialization, "serial"), 
        ("genres", genres, "genre"),
        ("themes", themes, "theme"), 
        ("demographic", demographic, "demo")
    ]

    new_cols = {}
    cols_to_drop = []

    for col_name, vocab, prefix in feature_set:
        print(f"Processing column: {col_name} with {len(vocab)} unique items.")
        cols_to_drop.append(col_name)

        parsed_lists = df[col_name].apply(parse_list)
        stripped = []
        for parsed_list in parsed_lists:
            for item in parsed_list:
                stripped.append(item.strip())

        
        for item in vocab:
            feature_col_name = f"{prefix}_{item.replace(' ', '_').replace('-', '_')}"

            col_values = []
            for parsed_list in parsed_lists:
                if item in parsed_list:
                    col_values.append(1)
                else:
                    col_values.append(0) 

            new_cols[feature_col_name] = col_values           
            
    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    df = df.drop(columns=cols_to_drop)
    return df

def load_ml_data(cleaned_path=ML_PATH):
    if not os.path.exists(cleaned_path):
        clean_ml_manga_dataset() # clean it

    df = pd.read_csv(cleaned_path) # load it

    return df

def load_ml_featureset(TESTER_PATH=TESTER_PATH, ML_PATH=ML_PATH):
    if not os.path.exists(TESTER_PATH):
        ml_df = load_ml_data(ML_PATH)
        df = initialize_ml_dataset(ml_df)
        df.to_csv(TESTER_PATH, index=False)
    
    df = pd.read_csv(TESTER_PATH)
    
    return df
    
def to_json_feature_importances(feature_dict, output_path=FT_IMP_PATH):

    # Save
    with open(output_path, 'w') as f:
        json.dump(feature_dict, f)

    with open(output_path, 'r') as f:
        tmp = json.load(f)

    return tmp

def json_load_feature_importances(input_path=FT_IMP_PATH):
    # Load
    with open(input_path, 'r') as f:
        loaded_dict = json.load(f)
    
    return loaded_dict

# if __name__ == "__main__":

#     ml_dataset = initialize_ml_dataset(load_ml_data(ML_PATH))
#     ml_dataset.to_csv(TESTER_PATH, index=False)
#     df = pd.read_csv(TESTER_PATH)