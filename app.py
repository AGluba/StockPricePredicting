import io
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import arima
import lstm
import pandas as pd

app = dash.Dash(__name__)
app.title = "Stock Price Prediction"

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <title>Stock Price Prediction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        html, body {
            height: 100%;
            width: 100%;
            background-color: #1e1e1e;
            overflow: hidden;
        }
    </style>
</head>
<body>
    <div id="app">{%app_entry%}</div>
    <footer>
        {%config%}z
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''


app.layout = html.Div([

    html.Div([
        html.Div([
            html.H3("Opcje", style={'textAlign': 'center', 'font-size': '24px', 'color': 'white'}),

            html.Label("Wybór rynku:", style={'color': 'white', 'font-size': '16px',}),
            dcc.Dropdown(
                id='market-dropdown',
                options=[
                    {'label': 'S&P500', 'value': '^GSPC'},
                    {'label': 'NASDAQ', 'value': 'NQ=F'},
                    {'label': 'Dow Jones Industrial Average', 'value': '^DJI'}
                ],
                value='^GSPC',
                style={'backgroundColor': 'white', 'color': 'black', 'font-size': '14px',  'margin-bottom': '10px'}
            ),

            html.Label("Wybór algorytmu:", style={'color': 'white', 'font-size': '16px'}),
            dcc.RadioItems(
                id='algorithm-radio',
                options=[
                    {'label': 'ARIMA', 'value': 'ARIMA'},
                    {'label': 'LSTM', 'value': 'LSTM'}
                ],
                value='ARIMA',
                labelStyle={'display': 'block', 'color': 'white', 'font-size': '14px'},
                style={'margin-bottom': '10px'}
            ),

            html.Label("Parametry ARIMA:", style={'color': 'white', 'font-size': '16px'}),
            dcc.RadioItems(
                id='arima-params-radio',
                options=[
                    {'label': 'Automatyczne', 'value': 'automatic'},
                    {'label': 'Ręczne', 'value': 'manual'}
                ],
                value='automatic',
                labelStyle={'display': 'block', 'color': 'white', 'font-size': '14px'},
                style={'margin-bottom': '10px'}
            ),

            html.Div([
                html.Div([
                    html.Label("Parametr p:", style={'color': 'white', 'font-size': '14px', 'margin-right': '10px'}),
                    dcc.Input(
                        id='arima-p',
                        type='number',
                        value=1,
                        min=0,
                        disabled=True,
                        style={'color': 'black', 'font-size': '14px', 'backgroundColor': '#f4f4f4'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

                html.Div([
                    html.Label("Parametr d:", style={'color': 'white', 'font-size': '14px', 'margin-right': '10px'}),
                    dcc.Input(
                        id='arima-d',
                        type='number',
                        value=1,
                        min=0,
                        disabled=True,
                        style={'color': 'black', 'font-size': '14px', 'backgroundColor': '#f4f4f4'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

                html.Div([
                    html.Label("Parametr q:", style={'color': 'white', 'font-size': '14px', 'margin-right': '10px'}),
                    dcc.Input(
                        id='arima-q',
                        type='number',
                        value=1,
                        min=0,
                        disabled=True,
                        style={'color': 'black', 'font-size': '14px', 'backgroundColor': '#f4f4f4'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),
            ], id='manual-arima-params', style={'display': 'none'}),

            html.Label("Parametry LSTM:", style={'color': 'white', 'font-size': '16px'}),
            dcc.RadioItems(
                id='lstm-params-radio',
                options=[
                    {'label': 'Automatyczne', 'value': 'automatic'},
                    {'label': 'Ręczne', 'value': 'manual'}
                ],
                value='automatic',
                labelStyle={'display': 'block', 'color': 'white', 'font-size': '14px'},
                style={'margin-bottom': '10px'}
            ),

            html.Div([
                html.Div([
                    html.Label("Liczba warstw:", style={'color': 'white', 'font-size': '14px', 'margin-right': '10px'}),
                    dcc.Input(
                        id='lstm-layers',
                        type='number',
                        value=2,
                        min=1,
                        disabled=True,
                        style={'color': 'black', 'font-size': '14px', 'backgroundColor': '#f4f4f4'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

                html.Div([
                    html.Label("Neuronów na warstwę:",
                               style={'color': 'white', 'font-size': '14px', 'margin-right': '10px'}),
                    dcc.Input(
                        id='lstm-neurons',
                        type='number',
                        value=50,
                        min=1,
                        disabled=True,
                        style={'color': 'black', 'font-size': '14px', 'backgroundColor': '#f4f4f4'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

                html.Div([
                    html.Label("Współczynnik uczenia:",
                               style={'color': 'white', 'font-size': '14px', 'margin-right': '10px'}),
                    dcc.Input(
                        id='lstm-learning-rate',
                        type='number',
                        value=0.001,
                        min=0.0001,
                        step=0.0001,
                        disabled=True,
                        style={'color': 'black', 'font-size': '14px', 'backgroundColor': '#f4f4f4'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

                html.Div([
                    html.Label("Liczba epok:", style={'color': 'white', 'font-size': '14px', 'margin-right': '10px'}),
                    dcc.Input(
                        id='lstm-epochs',
                        type='number',
                        value=10,
                        min=1,
                        disabled=True,
                        style={'color': 'black', 'font-size': '14px', 'backgroundColor': '#f4f4f4'}
                    )
                ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),
            ], id='manual-lstm-params', style={'display': 'none'}),

            html.Label("Okres przewidywania:", style={'color': 'white', 'font-size': '16px'}),
            dcc.Dropdown(
                id='prediction-days-dropdown',
                options=[
                    {'label': '1 dzień', 'value': 1},
                    {'label': '3 dni', 'value': 3},
                    {'label': '7 dni', 'value': 7}
                ],
                value=1,
                style={'backgroundColor': 'white', 'color': 'black', 'font-size': '14px', 'margin-bottom': '20px'}
            ),

            dcc.Loading(
                id="loading-spinner",
                type="circle",
                color="green",
                children=[
                    html.Button(
                        "Generuj",
                        id='generate-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': 'green', 'color': 'white', 'padding': '10px',
                            'justify-content': 'center', 'display': 'flex',
                            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer',
                            'font-size': '16px', 'align-items': 'center', 'margin': 'auto'
                        }
                    )
                ]
            ),

            html.Div([
                html.Div(
                    id='download-icon',
                    children="Pobierz raport ⬇",
                    style={
                        'fontSize': '24px',
                        'color': 'green',
                        'cursor': 'pointer',
                        'display': 'none'
                    }
                ),
                dcc.Download(id='download-report')
            ], style={'textAlign': 'center', 'marginTop': '20px'})
        ], style={
            'width': '25%', 'backgroundColor': '#2e2e2e', 'padding': '20px', 'borderTopRightRadius': '15px',
            'borderBottomRightRadius': '15px', 'color': 'black', 'fontFamily': 'Arial, sans-serif',
            'height': '100vh', 'boxSizing': 'border-box'
        }),

        html.Div([
            dcc.Graph(id='stock-plot'),
            dcc.Store(id='stored-data'),
            dcc.Store(id='report-data'),
            dcc.Store(id='error-value', data=0),
            html.Div(id='value-change-stats', style={
                'backgroundColor': '#2e2e2e', 'color': 'white', 'padding': '10px',
                'borderRadius': '10px', 'marginTop': '20px', 'fontSize': '16px',
                'textAlign': 'center', 'width': '100%'
            }),
        ], style={
            'flexGrow': 1,
            'marginLeft': '20px',
            'height': '100%',
            'flexDirection': 'column',
            'display': 'flex'
        }),
    ], style={'flexDirection': 'row', 'width': '100%', 'height': '100%', 'display': 'flex'}),

], style={'background': '#1e1e1e', 'width': '100%', 'height': '100%'})

@app.callback(
    Output('manual-arima-params', 'style'),
    [Input('arima-params-radio', 'value')]
)
def toggle_arima_params(arima_params):
    if arima_params == 'manual':
        return {'display': 'block', 'color': 'white'}
    return {'display': 'none'}

@app.callback(
    Output('manual-lstm-params', 'style'),
    [Input('lstm-params-radio', 'value')]
)
def toggle_lstm_params(lstm_params):
    if lstm_params == 'manual':
        return {'display': 'block', 'color': 'white'}
    return {'display': 'none'}

@app.callback(
    Output('download-icon', 'style'),
    Input('generate-button', 'n_clicks')
)
def show_download_icon(n_clicks):
    if n_clicks and n_clicks > 0:
        return {'fontSize': '24px', 'color': 'green', 'cursor': 'pointer', 'display': 'block'}
    return {'display': 'none'}

@app.callback(
    [Output('arima-p', 'disabled'),
     Output('arima-d', 'disabled'),
     Output('arima-q', 'disabled')],
    [Input('arima-params-radio', 'value')]
)
def toggle_arima_inputs(arima_params):
    if arima_params == 'manual':
        return [False, False, False]
    return [True, True, True]

@app.callback(
    [Output('lstm-layers', 'disabled'),
     Output('lstm-neurons', 'disabled'),
     Output('lstm-learning-rate', 'disabled'),
     Output('lstm-epochs', 'disabled')],
    [Input('lstm-params-radio', 'value')]
)
def toggle_lstm_inputs(lstm_params):
    if lstm_params == 'manual':
        return [False, False, False, False]
    return [True, True, True, True]

@app.callback(
    Output('download-report', 'data'),
    [Input('download-icon', 'n_clicks')],
    [State('stored-data', 'data'),
     State('report-data', 'data'),
     State('error-value', 'data'),
     State('algorithm-radio', 'value'),
     State('arima-params-radio', 'value'),
     State('arima-p', 'value'),
     State('arima-d', 'value'),
     State('arima-q', 'value'),
     State('lstm-params-radio', 'value'),
     State('lstm-layers', 'value'),
     State('lstm-neurons', 'value'),
     State('lstm-learning-rate', 'value'),
     State('lstm-epochs', 'value')]
)
def generate_report(n_clicks, stored_data, report_df, error, algorithm, params_arima, p, d, q, params_lstm, layers,
                    neurons, learning_rate, epochs):
    if n_clicks is None or stored_data is None:
        return dash.no_update

    if params_arima == 'automatic':
        p, d, q = 6, 1, 8

    if params_lstm == 'automatic':
        layers, neurons, learning_rate, epochs = 2, 128, 0.01, 50

    df = pd.read_json(report_df)
    buffer = io.StringIO()

    buffer.write("Raport\n")
    buffer.write(f"Model: {algorithm}\n")
    if algorithm == "ARIMA":
        buffer.write(f"Parametry; P={p}; D={d}; Q={q}\n")
    elif algorithm == "LSTM":
        buffer.write(
            f"LSTM: Warstwy={layers}, Neurony={neurons}, Wspolczynnik uczenia={learning_rate}, Epoki={epochs}\n")
    buffer.write("\n\nData;Dane rzeczywiste;Dane przewidziane\n")

    for _, row in df.iterrows():
        buffer.write(f"{row['Date']};{row['Actual']};{row['Predicted']}\n")

    buffer.write(f"\n\nMSE={error}\n")

    csv_data = buffer.getvalue()

    return (
        {
            "filename": "report.csv",
            "content": csv_data,
            "type": "text/csv"
        }
    )


@app.callback(
    Output('stored-data', 'data'),
    Output('report-data', 'data'),
    Output('error-value', 'data'),
    Output('generate-button', 'n_clicks'),
    [Input('market-dropdown', 'value'),
     Input('generate-button', 'n_clicks')],
    [State('stored-data', 'data'),
     State('algorithm-radio', 'value'),
     State('arima-params-radio', 'value'),
     State('arima-p', 'value'),
     State('arima-d', 'value'),
     State('arima-q', 'value'),
     State('lstm-params-radio', 'value'),
     State('lstm-layers', 'value'),
     State('lstm-neurons', 'value'),
     State('lstm-learning-rate', 'value'),
     State('lstm-epochs', 'value'),
     State('prediction-days-dropdown', 'value')]
)
def update_stored_data(market, n_clicks, stored_data, algorithm, arima_params, p, d, q, lstm_params, layers, neurons,
                       learning_rate, epochs, days):
    if not market:
        return {} , {}, 0, 0

    if n_clicks is None or n_clicks == 0:
        df = arima.generate_data(market)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df.to_json(), {}, 0, 0

    if n_clicks:
        df = pd.read_json(stored_data)
        if algorithm == 'ARIMA':
            if arima_params == 'automatic':
                p, d, q = 6, 1, 8
                forecast_df = arima.generate_arima_forecast(df, days, p, d, q)
                combined_df = pd.concat([df, forecast_df])
                combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.tz_localize(None)
                report_df, error = arima.generate_report(combined_df, p, d, q, days)
            else:
                forecast_df = arima.generate_arima_forecast(df, days, p, d, q)
                combined_df = pd.concat([df, forecast_df])
                combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.tz_localize(None)
                report_df, error = arima.generate_report(combined_df, p, d, q, days)
        if algorithm == 'LSTM':
            if lstm_params == 'automatic':
                layers, neurons, learning_rate, epochs = 2, 128, 0.01, 50
                forecast_df, report_df, error = lstm.generate_LSTM_forecast(df, days, layers, neurons, learning_rate, epochs)
            else:
                forecast_df, report_df, error = lstm.generate_LSTM_forecast(df, days, layers, neurons, learning_rate, epochs)
            df = df.reset_index()
            combined_df = pd.concat([df, forecast_df])

        combined_df = combined_df.reset_index(drop=True)
        return combined_df.to_json(), report_df.to_json(), error, 1

@app.callback(
    Output('stock-plot', 'figure'),
    Output('value-change-stats', 'children'),
    [Input('stored-data', 'data')],
    [Input('stock-plot', 'relayoutData')],
    [State('market-dropdown', 'value')],
)
def update_forecast(stored_data, relayoutData, market):
    if stored_data is None or stored_data == {}:
        return {}, "Brak danych do wyświetlenia."
    df = pd.read_json(stored_data)

    markets = {'^GSPC': 'S&P500', 'NQ=F': 'NASDAQ', '^DJI': 'Dow Jones Industrial Average'}

    fig = px.line(df, x='Date', y='Close', title=f"{markets[market]}")
    fig.update_traces(line=dict(color='green'))

    start_value = df['Close'].iloc[0]
    end_value = df['Close'].iloc[-1]

    absolute_change = end_value - start_value
    percent_change = (absolute_change / start_value) * 100

    value_change_text = html.Div([
        html.Span(f"{absolute_change:.2f} ",
                  style={'marginRight': '10px', 'color': 'green' if absolute_change > 0 else 'red'}),
        html.Span(f"({percent_change:.2f}%)", style={'color': 'green' if percent_change > 0 else 'red'})
    ], style={'display': 'inline-block'})
    if relayoutData and 'xaxis.range' in relayoutData:
        x_range = relayoutData['xaxis.range']
        x_range_start = pd.to_datetime(x_range[0]).replace(tzinfo=None)
        x_range_end = pd.to_datetime(x_range[1]).replace(tzinfo=None)

        filtered_df = df[(df['Date'] >= x_range_start) & (df['Date'] <= x_range_end)]
        y_min, y_max = filtered_df['Close'].min()* 0.99, filtered_df['Close'].max()* 1.01
        start_value = filtered_df['Close'].iloc[0]
        end_value = filtered_df['Close'].iloc[-1]

        absolute_change = end_value - start_value
        percent_change = (absolute_change / start_value) * 100

        value_change_text = html.Div([
            html.Span(f"{absolute_change:.2f} ", style={'marginRight': '10px', 'color': 'green' if absolute_change > 0 else 'red'}),
            html.Span(f"({percent_change:.2f}%)", style={'color': 'green' if percent_change > 0 else 'red'})
        ], style={'display': 'inline-block'})

        fig.update_layout(
            xaxis=dict(
                range=[x_range_start, x_range_end],
                rangeslider=dict(visible=True),
                type='date',
                rangeslider_thickness=0.1
            ),
            yaxis=dict(
                autorange=False,
                range=[y_min, y_max]
            ),
            plot_bgcolor='black',
            xaxis_title=dict(text='Data', font=dict(size=16, color='#FFFFFF')),
            yaxis_title=dict(text='Cena', font=dict(size=16, color='#FFFFFF')),
            paper_bgcolor='black',
            font=dict(color='white')
        )
    else:
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date',
                rangeslider_thickness=0.1
            ),
            yaxis=dict(autorange=True),
            xaxis_title=dict(text='Data', font=dict(size=16, color='#FFFFFF')),
            yaxis_title=dict(text='Cena', font=dict(size=16, color='#FFFFFF')),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )

    return fig, value_change_text


if __name__ == '__main__':
    app.run_server(debug=False)
