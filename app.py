import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA Winner Predictor", page_icon="🏀")

@st.cache_resource
def load_bundle():
    return joblib.load("nba_elo_model.joblib")

bundle = load_bundle()
model = bundle["model"]
team_elo = bundle["team_elo"]
teams = sorted(team_elo)

st.title("🏀 NBA Winner Predictor")
st.caption("Based on Elo ratings. Ratings reflect games through December 2022.")

col1, col2 = st.columns(2)
home = col1.selectbox("Home team", teams, index=0)
away = col2.selectbox("Away team", teams, index=1)

if st.button("Predict"):
    if home == away:
        st.warning("Pick two different teams.")
    else:
        X = pd.DataFrame([[team_elo[home], team_elo[away]]],
                         columns=["ELO_h", "ELO_a"])
        p_home = model.predict_proba(X)[0][1]
        winner = home if p_home >= 0.5 else away

        st.subheader(f"Predicted winner: {winner}")
        st.progress(float(p_home))
        st.write(f"{home}: **{p_home:.0%}**  |  {away}: **{1 - p_home:.0%}**")
        st.write(f"Elo ratings: {home} {team_elo[home]:.0f}, {away} {team_elo[away]:.0f}")