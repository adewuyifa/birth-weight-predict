import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from category_encoders import OneHotEncoder


# Import
def wrangle(filepath):
    """Read maternity data file into ``DataFrame``.
    Parameters
    ----------
    filepath : str
    Location of CSV file.
    """
    df = pd.read_excel(filepath)
    # Drop leaky column
    df.drop(columns="BWT", inplace=True)
    # Rename columns
    df.columns = ["Birth_weight", "Age", "Mother_weight", "Race",
                  "Smoking_status",
                  "History_of_premature_labor",
                  "History_of_Hypertension", "Presence_of_uterine_irritability",
                  "Physician_visits"]
    # Rename values in categorical variables columns
    df["Race"] = df["Race"].map({1: "White", 2: "Black", 3: "Others"})
    df["Smoking_status"] = df["Smoking_status"].map({0: "No", 1: "Yes"})
    df["History_of_Hypertension"] = df["History_of_Hypertension"].map({0: "No", 1: "Yes"})
    df["Presence_of_uterine_irritability"] = df["Presence_of_uterine_irritability"].map({0: "No", 1: "Yes"})
    return df

df = wrangle("data/low_birth_weight.xls")

# Split
target = "Birth_weight"
X = df.drop(columns=target)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline
acc_baseline = y_train.value_counts(normalize=True).max()

# Iterate
# Build model
model = make_pipeline(
OneHotEncoder(use_cat_names=True),
LogisticRegression(max_iter=1000)
)
# Fit model to training data
model.fit(X_train, y_train)

# App dev
navbar = dbc.NavbarSimple(
    brand="Birth Weight Predictor",
    brand_href="#",
    color="primary",
    dark=True,
    brand_style={'fontSize': 30, 'fontWeight': 'bold'}
)
question_one = html.Div(
    [
        dbc.Label("1. Age of mother (in years)"),
        dbc.Input(
            id='age',
            type='number',
            min=13,
            max=75,
            step=1,
            placeholder="Age"),
    ]
)
question_two = html.Div(
    [
        dbc.Label("2. Weight of mother at the last menstrual period (in pounds)"),
        dbc.Input(
            id='weight',
            type='number',
            min=50,
            max=600,
            step=1,
            placeholder="Weight in pounds"
        ),
    ],
    className="mb-3",
)
question_three = html.Div(
    [
        dbc.Label("3. Race of mother", html_for="dropdown"),
        dcc.Dropdown(
            id="race-id",
            options=[{'label': 'White', 'value': 'White'},
                     {'label': 'Black', 'value': 'Black'},
                     {'label': 'Others', 'value': 'Others'}],
            placeholder="Race"
        ),
    ],
    className="mb-3",
)
question_four = html.Div(
    [
        dbc.Label("4. Smoking status during pregnancy", html_for="dropdown"),
        dcc.Dropdown(
            id="smoke-id",
            options=[{'label': 'Yes', 'value': 'Yes'},
                     {'label': 'No', 'value': 'No'}],
            placeholder="Smoking status"
        ),
    ],
    className="mb-3",
)
question_five = html.Div(
    [
        dbc.Label("5. History of premature labor", html_for="dropdown"),
        dcc.Dropdown(
            id="prem-id",
            options=[{'label': 'None', 'value': 0},
                     {'label': 'One', 'value': 1},
                     {'label': 'Two', 'value': 2},
                     {'label': 'Three and above', 'value': 3}],
            placeholder="History of premature labor"
        ),
    ],
    className="mb-3",
)
question_six = html.Div(
    [
        dbc.Label("6. History of hypertension", html_for="dropdown"),
        dcc.Dropdown(
            id="hyper-id",
            options=[{'label': 'Yes', 'value': 'Yes'},
                     {'label': 'No', 'value': 'No'}],
            placeholder="History of hypertension"
        ),
    ],
    className="mb-3",
)
question_seven = html.Div(
    [
        dbc.Label("7. Presence of uterine irritability", html_for="dropdown"),
        dcc.Dropdown(
            id="uterine-id",
            options=[{'label': 'Yes', 'value': 'Yes'},
                     {'label': 'No', 'value': 'No'}],
            placeholder="Presence of uterine irritability"
        ),
    ],
    className="mb-3",
)
question_eight = html.Div(
    [
        dbc.Label("8. Number of physician visits during the first trimester"),
        dbc.Input(
            id='physician-visit',
            type='number',
            min=0,
            max=20,
            step=1,
            placeholder="No. of physician visits"
        ),
    ],
    className="mb-3",
)
form = dbc.Form([question_one,question_two,question_three,question_four,
                 question_five,question_six,question_seven,question_eight])
external_stylesheets = [dbc.themes.CERULEAN]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
server = app.server
app.config.suppress_callback_exceptions = True
app.title = "BWP"
app.layout = html.Div([
    html.Div(navbar),
    html.Br(),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("Information before use"),
                html.P("This app is based on a Logistic regression model that predicts the birth weight of "
                       "pregnant women based on the answers to eight questions on "
                       "their maternal behaviour. The app returns the predicted outcome as "
                       "low birth weight (birth weight <2500g) or normal birth weight "
                       "(birth weight >2500g).", style={"textAlign":"justify"}),
                html.P("All questions must be answered before clicking on the submit "
                       "button")
            ], width=5),
            dbc.Col([
                form,
                dbc.Button('Submit', id='submit-val', n_clicks=0, color='primary'),
                html.Br(),
                html.Hr(),
                dbc.Card([
                    dbc.CardHeader("Result:"),
                    dbc.CardBody(id="result-text")
                ]),
                html.Br()
            ])
        ])
    ]),
])


@app.callback(
    Output("result-text", "children"),
    Input("submit-val", "n_clicks"),
    State("age", "value"),
    State("weight", "value"),
    State("race-id", "value"),
    State("smoke-id", "value"),
    State("prem-id", "value"),
    State("hyper-id", "value"),
    State("uterine-id", "value"),
    State("physician-visit", "value")
)
def make_prediction(n_clicks, Age, Mother_weight, Race,Smoking_status, History_of_premature_labor,
                    History_of_Hypertension, Presence_of_uterine_irritability,Physician_visits):
    if Age is None:
        raise PreventUpdate
    if Mother_weight is None:
        raise PreventUpdate
    if Race is None:
        raise PreventUpdate
    if Smoking_status is None:
        raise PreventUpdate
    if History_of_premature_labor is None:
        raise PreventUpdate
    if History_of_Hypertension is None:
        raise PreventUpdate
    if Presence_of_uterine_irritability is None:
        raise PreventUpdate
    if Physician_visits is None:
        raise PreventUpdate
    data = {
        "Age": Age,
        "Mother_weight": Mother_weight,
        "Race": Race,
        "Smoking_status" : Smoking_status,
        "History_of_premature_labor":History_of_premature_labor,
        "History_of_Hypertension":History_of_Hypertension,
        "Presence_of_uterine_irritability": Presence_of_uterine_irritability,
        "Physician_visits": Physician_visits
    }
    df = pd.DataFrame(data, index=[0])
    predicted_value = {
        0:"Normal birth weight",
        1:"Low birth weight"
    }
    prediction = model.predict(df)
    my_dict = {
        0:"Normal Birth Weight (i.e birth weight  >= 2500 g)",
        1:"Low Birth Weight (i.e birth weight  < 2500g)"
    }
    y_train_pred_proba = model.predict_proba(X_train)
    maxElement = np.amax(y_train_pred_proba)
    return f"The predicted outcome is: {np.vectorize(my_dict.get)(prediction[0])}, OR: {round(maxElement, 2)}"


if __name__ == '__main__':
    app.run_server()
