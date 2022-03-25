from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

import numpy as np
import pandas as pd
import joblib
import pickle

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/welcome", response_class=HTMLResponse)
def welcome_page():
    return """
    <html>
            <head>
                <title>Some HTML in here</title>
            </head>
            <body>
                <form action='/predict' method='GET'>
  
                        <div className="form-group">
                            <label for="">Home Team</label>
                            <select name='home_team'  className="form-control" id="">
                            <option>Morocco</option>
                            <option>Egypt</option>
                            <option>Algeria</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label for="">Away Team</label>
                            <select name='away_team'  class="form-control" id="">
                            <option>Morocco</option>
                            <option>Egypt</option>
                            <option>Algeria</option>
                            </select>
                        </div>
                        <button type="submit" class="btn btn-success">Predicter</button>
                        </form>
            </body>
        </html>
    """
@app.get('/predict/')
async def predict_outcome(home_team: str, away_team: str):
    
    meta_model = joblib.load('model_rf__ODC_football_outcome_predictor_20220325__meta.model')
    
    model = meta_model['model']
    cols = meta_model['columns']

    a_country = home_team
    h_country = away_team

    df = pd.DataFrame(np.zeros((1,len(cols)), dtype=int), columns=cols)
    
    ranking = pd.read_csv(meta_model['ranking'])
    
    try:
        df.home_rank.iloc[0] = ranking[((ranking.rank_date == '2021-05-27') & (ranking.country_full == h_country))]['rank'].values[0]
    except:
        df.home_rank.iloc[0] = 155

    try:
        df.away_rank.iloc[0] = ranking[((ranking.rank_date == '2021-05-27') & (ranking.country_full == a_country))]['rank'].values[0]
    except:
        df.away_rank.iloc[0] = 155
        
    df['home_team_'+h_country].iloc[0] = 1
    df['away_team_'+a_country].iloc[0] = 1
    
    #outcome = model.predict(df)
    
    proba = model.predict_proba(df)
    outcome = model.predict(df)

    msg = ''

    if outcome == 'draw':
        msg = '{0} will draw with {1} with {2:.0%} chance'.format(home_team, away_team, proba[0][0])
    elif outcome == 'lose':
        msg = '{0} will lose to {1} with {2:.0%} chance'.format(home_team, away_team, proba[0][1])
    elif outcome == 'win':
        msg = '{0} will win versus {1} with {2:.0%} chance'.format(home_team, away_team, proba[0][2])
    else:
        msg = 'NA'
        
    return {
        'home team' : home_team,
        'away team' : away_team,
        'draw' : '{0:.0%}'.format(proba[0][0]),
        'lose' : '{0:.0%}'.format(proba[0][1]),
        'win' : '{0:.0%}'.format(proba[0][2]),
        'message' : msg
    }
        
    

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}