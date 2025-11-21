# Dataset Operations
import pandas as pd
import os 

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

USER_PATH = os.path.join(DATASET_DIR, "user_data.csv")
CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_manga_entries.csv")
INPUT_PATH = os.path.join(DATASET_DIR, "manga_entries.csv")

# The three following functions are for cleaning / loading all of the csv files needed.
def clean_manga_dataset(input_path=INPUT_PATH, output_path=CLEANED_PATH):

    df = pd.read_csv(input_path)

    #Remove columns 13 (german name), 14 (french name), 15 (spanish name) since they had mixed data types and are not needed
    #Remove columns 26 (description) and 27 (background) since they contain large text data that is not needed for analysis 
    # Added columns 25 and 26 back, for extra data for user
    df.drop(df.columns[[12, 13, 14]], axis=1, inplace=True)

    #Save cleaned dataframe to new CSV file, rewrite existing cleaned file if it exists
    df.to_csv(output_path, index=False)

    return df

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
            "read_manga"          
        ])
        df.to_csv(user_path, index=False) # return it (will be blank if it was just made)
        return df
    # shouldnt be blank, but it could be if no user has been made yet
    return pd.read_csv(user_path)
