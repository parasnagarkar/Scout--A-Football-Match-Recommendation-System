from flask import Flask, request, jsonify, render_template
import pandas as pd
import sqlite3
import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

app = Flask(__name__)

# Set up logging for better debugging in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Consolidated Data Loading, Preprocessing, and Model Setup ---
def load_and_prepare_model_components():
    logging.info("Starting data loading and preprocessing from local database...")
    
    # --- Data Acquisition from local database.sqlite ---
    try:
        conn = sqlite3.connect('database.sqlite')
        matches_df = pd.read_sql_query("SELECT * FROM Match;", conn)
        team_df = pd.read_sql_query("SELECT team_api_id, team_long_name FROM Team;", conn)
        league_df = pd.read_sql_query("SELECT id, name FROM League;", conn)
        conn.close()
        logging.info("Raw data loaded successfully from local database.sqlite.")
    except sqlite3.Error as e:
        logging.error(f"FATAL ERROR: Could not load data from database.sqlite. Ensure the file exists and is not corrupted. Error: {e}")
        raise RuntimeError("Failed to load data from local database.sqlite. Check file.") from e
    except Exception as e:
        logging.error(f"FATAL ERROR: An unexpected error occurred during data loading: {e}")
        raise RuntimeError("An unexpected error occurred during data loading.") from e


    # --- Preprocessing & Feature Engineering ---
    matches_df['date'] = pd.to_datetime(matches_df['date'])

    # Merge team names
    team_names_df = team_df[['team_api_id', 'team_long_name']].copy()
    matches_df = pd.merge(matches_df, team_names_df, left_on='home_team_api_id', right_on='team_api_id', how='left', suffixes=('', '_home_temp'))
    matches_df.rename(columns={'team_long_name': 'home_team_name'}, inplace=True)
    matches_df.drop(columns=['team_api_id_home_temp'], errors='ignore', inplace=True)

    matches_df = pd.merge(matches_df, team_names_df, left_on='away_team_api_id', right_on='team_api_id', how='left', suffixes=('', '_away_temp'))
    matches_df.rename(columns={'team_long_name': 'away_team_name'}, inplace=True)
    matches_df.drop(columns=['team_api_id_away_temp'], errors='ignore', inplace=True)
    logging.info("Team names merged.")

    # Merge league names
    league_names_df = league_df[['id', 'name']].copy()
    league_names_df.rename(columns={'id': 'league_id', 'name': 'league_name'}, inplace=True)
    matches_df = pd.merge(matches_df, league_names_df, left_on='league_id', right_on='league_id', how='left')
    logging.info("League names merged.")

    # Functions for XML parsing
    def extract_value_from_xml(xml_string, tag):
        if pd.isna(xml_string): return None
        match = re.search(fr'<{tag}>(\d+)</{tag}>', str(xml_string))
        if match: return int(match.group(1))
        return None

    def count_events_from_xml(xml_string):
        if pd.isna(xml_string): return 0
        return str(xml_string).count('<event>')

    # Extracting Possession
    matches_df['home_possession'] = matches_df['possession'].apply(lambda x: extract_value_from_xml(x, 'home'))
    matches_df['away_possession'] = matches_df['possession'].apply(lambda x: extract_value_from_xml(x, 'away'))

    # Extracting Total Events
    event_columns = ['shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner']
    for col in event_columns:
        matches_df[f'total_{col}_events'] = matches_df[col].apply(count_events_from_xml)
    logging.info("Match event statistics extracted.")

    # Fill NaN for extracted and goal columns
    columns_to_fill_zero = [
        'home_team_goal', 'away_team_goal',
        'home_possession', 'away_possession',
        'total_shoton_events', 'total_shotoff_events', 'total_foulcommit_events',
        'total_card_events', 'total_cross_events', 'total_corner_events'
    ]
    for col in columns_to_fill_zero:
        if col in matches_df.columns:
            matches_df[col] = matches_df[col].fillna(0).astype(int)
    logging.info("Initial NaNs filled.")

    # Drop irrelevant/sparse columns
    columns_to_drop = []
    player_xy_cols = [col for col in matches_df.columns if re.match(r'home_player_[XY]\d+|away_player_[XY]\d+', col)]
    columns_to_drop.extend(player_xy_cols)
    player_id_cols = [col for col in matches_df.columns if re.match(r'home_player_\d+|away_player_\d+', col)]
    columns_to_drop.extend(player_id_cols)
    betting_odds_prefixes = ['B365', 'Bb', 'IW', 'LAD', 'WH', 'SJ', 'VC', 'GB', 'BS']
    betting_odds_cols = [col for col in matches_df.columns if any(col.startswith(prefix) for prefix in betting_odds_prefixes)]
    columns_to_drop.extend(betting_odds_cols)
    original_xml_cols = ['goal', 'shoton', 'shotoff', 'foulcommit', 'card', 'cross', 'corner', 'possession']
    columns_to_drop.extend(original_xml_cols)
    columns_to_drop = list(set(columns_to_drop)) # Remove duplicates
    
    matches_df_cleaned = matches_df.drop(columns=columns_to_drop, errors='ignore')
    logging.info(f"Dropped {len(columns_to_drop)} unnecessary columns.")

    # Feature Engineering
    matches_df_cleaned['goal_difference'] = abs(matches_df_cleaned['home_team_goal'] - matches_df_cleaned['away_team_goal'])
    matches_df_cleaned['total_goals'] = matches_df_cleaned['home_team_goal'] + matches_df_cleaned['away_team_goal']
    def get_match_outcome(row):
        if row['home_team_goal'] > row['away_team_goal']: return 'Home Win'
        elif row['home_team_goal'] < row['away_team_goal']: return 'Away Win'
        else: return 'Draw'
    matches_df_cleaned['match_outcome'] = matches_df_cleaned.apply(get_match_outcome, axis=1)

    matches_df_cleaned['total_shots'] = matches_df_cleaned['total_shoton_events'] + matches_df_cleaned['total_shotoff_events']
    matches_df_cleaned['attack_rating'] = (matches_df_cleaned['total_goals'] * 3) + (matches_df_cleaned['total_shoton_events'] * 2) + matches_df_cleaned['total_cross_events'] + matches_df_cleaned['total_corner_events']
    matches_df_cleaned['defensive_rating'] = matches_df_cleaned['total_foulcommit_events'] + (matches_df_cleaned['total_card_events'] * 3)
    matches_df_cleaned['possession_balance'] = 100 - abs(matches_df_cleaned['home_possession'] - matches_df_cleaned['away_possession'])
    matches_df_cleaned['possession_balance'] = matches_df_cleaned.apply(
        lambda row: 0 if (row['home_possession'] == 0 and row['away_possession'] == 0) else row['possession_balance'], axis=1
    )
    matches_df_cleaned['excitement_score'] = (matches_df_cleaned['total_goals'] * 5) + \
                                              (matches_df_cleaned['goal_difference'].apply(lambda x: 10 if x <= 1 else 0)) + \
                                              (matches_df_cleaned['total_shots'] * 1) + \
                                              (matches_df_cleaned['total_card_events'] * 2) + \
                                              (matches_df_cleaned['possession_balance'] * 0.1)
    logging.info("Features engineered.")

    # Final feature selection for matches_processed_df
    final_feature_columns = [
        'match_api_id', 'date', 'league_id', 'league_name', 'season', # 'date' is here for general DataFrame use
        'home_team_api_id', 'home_team_name', 'away_team_api_id', 'away_team_name',
        'home_team_goal', 'away_team_goal',
        'home_possession', 'away_possession',
        'total_shoton_events', 'total_shotoff_events', 'total_foulcommit_events',
        'total_card_events', 'total_cross_events', 'total_corner_events',
        'goal_difference', 'total_goals', 'match_outcome', 'total_shots',
        'attack_rating', 'defensive_rating', 'possession_balance', 'excitement_score'
    ]
    # Filter for columns that actually exist in the DataFrame
    existing_final_cols = [col for col in final_feature_columns if col in matches_df_cleaned.columns]
    matches_processed_df = matches_df_cleaned[existing_final_cols].copy()
    logging.info(f"Final matches_processed_df prepared with {matches_processed_df.shape[1]} columns.")

    # Feature Vectorization (MinMaxScaler + OneHotEncoder)
    numerical_features = [
        'home_team_goal', 'away_team_goal', 'goal_difference', 'total_goals',
        'home_possession', 'away_possession', 'total_shoton_events', 'total_shotoff_events',
        'total_foulcommit_events', 'total_card_events', 'total_cross_events', 'total_corner_events',
        'total_shots', 'attack_rating', 'defensive_rating', 'possession_balance', 'excitement_score'
    ]
    categorical_features = [
        'league_name', 'home_team_name', 'away_team_name', 'match_outcome'
    ]
    
    # Filter features based on what's in matches_processed_df
    numerical_features = [f for f in numerical_features if f in matches_processed_df.columns]
    categorical_features = [f for f in categorical_features if f in matches_processed_df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # CRUCIAL FIX: Pass only the explicitly defined numerical and categorical features to fit_transform.
    # The 'date' column, and other non-feature columns, will remain in matches_processed_df
    # but won't be processed by the ColumnTransformer's internal array conversion.
    match_features_matrix = preprocessor.fit_transform(matches_processed_df[numerical_features + categorical_features])
    logging.info(f"Feature matrix created with shape: {match_features_matrix.shape}")
    logging.info("Model components ready.")

    return matches_processed_df, preprocessor, match_features_matrix

# Global variables to store our pre-processed data and model components
try:
    matches_processed_df, preprocessor, match_features_matrix = load_and_prepare_model_components()
except Exception as e:
    logging.critical(f"Application failed to initialize due to data loading/preprocessing error: {e}")
    raise

# --- 2. Recommendation Logic Function ---
def recommend_matches(seed_match_id, matches_df, features_matrix, top_n=10):
    """
    Recommends football matches similar to a given seed match ID using cosine similarity.
    """
    seed_match_row = matches_df[matches_df['match_api_id'] == seed_match_id]

    if seed_match_row.empty:
        logging.warning(f"Match with ID {seed_match_id} not found for recommendation.")
        return pd.DataFrame()

    seed_match_index = seed_match_row.index[0]
    seed_match_vector = features_matrix[seed_match_index]

    similarities = cosine_similarity(seed_match_vector.reshape(1, -1), features_matrix)
    similarities_1d = similarities.flatten()

    top_similar_indices = np.argsort(similarities_1d)[::-1]
    top_similar_indices = top_similar_indices[top_similar_indices != seed_match_index] # Exclude seed itself
    recommended_indices = top_similar_indices[:top_n]

    recommended_matches_df = matches_df.iloc[recommended_indices].copy()
    recommended_matches_df['similarity_score'] = similarities_1d[recommended_indices]

    # Define columns for display in the API response
    display_cols = [
        'similarity_score', 'date', 'league_name', 'home_team_name', 'away_team_name',
        'home_team_goal', 'away_team_goal', 'total_goals', 'match_outcome', 'excitement_score', 'match_api_id'
    ]
    # Filter to ensure only existing columns are selected for display
    display_cols = [col for col in display_cols if col in recommended_matches_df.columns]

    return recommended_matches_df[display_cols].sort_values(by='similarity_score', ascending=False)

# --- 3. Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    match_id_str = request.args.get('match_id')
    top_n_str = request.args.get('top_n', '10')

    if not match_id_str:
        logging.warning("Missing 'match_id' parameter in recommendation request.")
        return jsonify({"error": "match_id parameter is required."}), 400

    try:
        match_id = int(match_id_str)
        top_n = int(top_n_str)
    except ValueError:
        logging.error(f"Invalid 'match_id' or 'top_n' received: match_id={match_id_str}, top_n={top_n_str}")
        return jsonify({"error": "match_id and top_n must be integers."}), 400

    logging.info(f"Received request for recommendations for match_id: {match_id}, top_n: {top_n}")
    recommended_df = recommend_matches(
        seed_match_id=match_id,
        matches_df=matches_processed_df,
        features_matrix=match_features_matrix,
        top_n=top_n
    )

    if recommended_df.empty:
        logging.info(f"No recommendations found for match_id: {match_id}.")
        return jsonify({"message": "No recommendations found for this Match ID."}), 404

    # Convert date to string for JSON serialization
    recommended_df['date'] = recommended_df['date'].dt.strftime('%Y-%m-%d')
    recommendations = recommended_df.to_dict(orient='records')

    logging.info(f"Returning {len(recommendations)} recommendations for match_id: {match_id}.")
    return jsonify(recommendations)

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(debug=True)