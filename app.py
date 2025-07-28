import os
import glob
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

try:
    import talib
    TALIB_AVAILABLE = True

except ImportError:
    print("TA-Lib not found. Using mock implementation for demonstration purposes.")
    TALIB_AVAILABLE = False
    
    class MockTaLib:
        @staticmethod
        def CDLHAMMER(open_data, high_data, low_data, close_data):
            result = np.zeros_like(close_data)
            for i in range(1, len(close_data)):
                if (high_data[i] - low_data[i] > 3 * (open_data[i] - close_data[i]) and
                    close_data[i] > open_data[i] and
                    (high_data[i] - close_data[i]) < 0.3 * (high_data[i] - low_data[i])):
                    result[i] = 100
            return result
        
        @staticmethod
        def CDLDRAGONFLYDOJI(open_data, high_data, low_data, close_data):
            result = np.zeros_like(close_data)
            for i in range(1, len(close_data)):
                if (abs(open_data[i] - close_data[i]) < 0.1 * (high_data[i] - low_data[i]) and
                    (high_data[i] - max(open_data[i], close_data[i])) < 0.1 * (high_data[i] - low_data[i])):
                    result[i] = 100
            return result
        
        @staticmethod
        def CDLEVENINGSTAR(open_data, high_data, low_data, close_data):
            result = np.zeros_like(close_data)
            for i in range(2, len(close_data)):
                if (close_data[i-2] > open_data[i-2] and
                    abs(close_data[i-1] - open_data[i-1]) < 0.3 * (high_data[i-1] - low_data[i-1]) and
                    close_data[i] < open_data[i] and
                    close_data[i] < close_data[i-2]):
                    result[i] = 100
            return result
        
        @staticmethod
        def CDL3WHITESOLDIERS(open_data, high_data, low_data, close_data):
            result = np.zeros_like(close_data)
            for i in range(3, len(close_data)):
                bullish_candles = (
                    close_data[i-3] > open_data[i-3] and
                    close_data[i-2] > open_data[i-2] and
                    close_data[i-1] > open_data[i-1]
                )
                
                if not bullish_candles:
                    continue
                
                body1 = close_data[i-3] - open_data[i-3]
                body2 = close_data[i-2] - open_data[i-2]
                body3 = close_data[i-1] - open_data[i-1]
                
                higher_closes = (
                    close_data[i-3] < close_data[i-2] < close_data[i-1]
                )
                
                opens_within_body = (
                    open_data[i-2] > open_data[i-3] and
                    open_data[i-2] < close_data[i-3] and
                    open_data[i-1] > open_data[i-2] and
                    open_data[i-1] < close_data[i-2]
                )
                
                upper_shadow1 = high_data[i-3] - close_data[i-3]
                upper_shadow2 = high_data[i-2] - close_data[i-2]
                upper_shadow3 = high_data[i-1] - close_data[i-1]
                
                small_upper_shadows = (
                    upper_shadow1 <= 0.15 * body1 and
                    upper_shadow2 <= 0.15 * body2 and
                    upper_shadow3 <= 0.15 * body3
                )
                
                min_body_size = (high_data[i-1] - low_data[i-1]) * 0.3
                decent_body_size = (
                    body1 > min_body_size and
                    body2 > min_body_size and
                    body3 > min_body_size
                )
                
                if (bullish_candles and higher_closes and 
                    opens_within_body and small_upper_shadows and
                    decent_body_size):
                    result[i] = 100
            
            return result
    
    talib = MockTaLib()

SCRIPTS = ['BAJAJ-AUTO', 'BHARTIARTL', 'ICICIBANK', 'RELIANCE', 'TCS']
TIMEFRAMES = ['1Min', '5Min', '10Min', '15Min', '30Min', '1Hr']

def detect_rising_window(open_data, high_data, low_data, close_data):
    result = np.zeros_like(close_data)
    for i in range(1, len(close_data)):
        if low_data[i] > high_data[i-1]:
            result[i] = 100
    return result

PATTERNS = {
    'Hammer': talib.CDLHAMMER,
    'Dragonfly Doji': talib.CDLDRAGONFLYDOJI,
    'Rising Window': detect_rising_window,
    'Evening Star': talib.CDLEVENINGSTAR,
    'Three White Soldiers': talib.CDL3WHITESOLDIERS
}

def load_data(script):
    data_path = os.path.join('DU_Data', '5Scripts', script)
    all_files = glob.glob(os.path.join(data_path, '*.csv'))
    
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    if not df_list:
        return None
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    combined_df['datetime'] = pd.to_datetime(combined_df['date'] + ' ' + combined_df['time'], 
                                           format='%d-%m-%Y %H:%M:%S')
    
    combined_df = combined_df.sort_values('datetime')
    
    return combined_df

def resample_data(df, timeframe):
    if timeframe == '1Min':
        return df
    
    timeframe_map = {
        '5Min': '5min',
        '10Min': '10min',
        '15Min': '15min',
        '30Min': '30min',
        '1Hr': 'H'
    }
    
    resampled = df.set_index('datetime').resample(timeframe_map[timeframe]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()

def detect_patterns(df, pattern_func):
    if callable(pattern_func):
        return pattern_func(df['open'].values, df['high'].values, 
                           df['low'].values, df['close'].values)
    return np.zeros(len(df))

def get_pattern_occurrences(script, timeframe, pattern_name):
    df = load_data(script)
    if df is None:
        return pd.DataFrame()
    
    resampled_df = resample_data(df, timeframe)
    pattern_func = PATTERNS[pattern_name]
    
    pattern_results = detect_patterns(resampled_df, pattern_func)
    
    resampled_df['pattern'] = pattern_results
    pattern_df = resampled_df[resampled_df['pattern'] != 0].copy()
    
    if pattern_df.empty:
        return pd.DataFrame()
    
    pattern_df['script'] = script
    pattern_df['timeframe'] = timeframe
    pattern_df['pattern_name'] = pattern_name
    
    return pattern_df[['datetime', 'script', 'timeframe', 'pattern_name', 'open', 'high', 'low', 'close']]

def create_candlestick_chart(df, selected_row_index, theme="light"):
    try:
        if df.empty or selected_row_index is None:
            print("DataFrame is empty or row index is None")
            return go.Figure()

        selected_pattern = df.iloc[selected_row_index]
        script = selected_pattern['script']
        timeframe = selected_pattern['timeframe']
        pattern_datetime = selected_pattern['datetime']

        print(f"Creating chart for: {script}, {timeframe}, {pattern_datetime}")

        full_df = load_data(script)
        if full_df is None or full_df.empty:
            print(f"No data found for script: {script}")
            return go.Figure()

        resampled_df = resample_data(full_df, timeframe)
        if resampled_df.empty:
            print(f"No resampled data for timeframe: {timeframe}")
            return go.Figure()

        if isinstance(pattern_datetime, str):
            pattern_datetime = pd.to_datetime(pattern_datetime)

        matching_rows = resampled_df[resampled_df['datetime'] == pattern_datetime]
        if matching_rows.empty:
            print(f"No exact match found, finding closest datetime")
            time_diffs = abs(resampled_df['datetime'] - pattern_datetime)
            closest_idx = time_diffs.idxmin()
            pattern_index = resampled_df.index.get_loc(closest_idx)
            actual_pattern_datetime = resampled_df.iloc[pattern_index]['datetime']
            print(f"Using closest datetime: {actual_pattern_datetime}")
        else:
            pattern_index = resampled_df.index.get_loc(matching_rows.index[0])
            actual_pattern_datetime = pattern_datetime

        candles_before = 20
        candles_after = 5

        start_index = max(0, pattern_index - candles_before)
        end_index = pattern_index + 1 + candles_after
        chart_df = resampled_df.iloc[start_index:end_index].copy()

        timeframe_highlight_width = {
            '1Min': pd.Timedelta(minutes=1),
            '5Min': pd.Timedelta(minutes=5),
            '10Min': pd.Timedelta(minutes=10),
            '15Min': pd.Timedelta(minutes=15),
            '30Min': pd.Timedelta(minutes=30),
            '1Hr': pd.Timedelta(hours=1)
        }

        highlight_width = timeframe_highlight_width.get(timeframe, pd.Timedelta(minutes=1))

        if chart_df.empty:
            print("Chart DataFrame is empty")
            return go.Figure()

        print(f"Displaying {len(chart_df)} candles from {chart_df['datetime'].min()} to {chart_df['datetime'].max()}")

        fig = go.Figure(data=[go.Candlestick(
            x=chart_df['datetime'],
            open=chart_df['open'],
            high=chart_df['high'],
            low=chart_df['low'],
            close=chart_df['close'],
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            name=f"{script} - {timeframe}"
        )])

        try:
            y_range = chart_df['high'].max() - chart_df['low'].min()
            y_margin = y_range * 0.05

            fig.add_shape(
                type="rect",
                x0=actual_pattern_datetime - highlight_width/2,
                y0=chart_df['low'].min() - y_margin,
                x1=actual_pattern_datetime + highlight_width/2,
                y1=chart_df['high'].max() + y_margin,
                line=dict(color="rgba(255, 193, 7, 0.8)", width=2),
                fillcolor="rgba(255, 193, 7, 0.15)"
            )

            fig.add_annotation(
                x=actual_pattern_datetime,
                y=chart_df['high'].max() + y_margin * 2,
                text=f"{selected_pattern['pattern_name']}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#ff6b35",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#ff6b35",
                borderwidth=1,
                font=dict(size=12, color="#ff6b35")
            )

        except Exception as e:
            print(f"Error highlighting pattern candle: {e}")

        if theme == "dark":
            plot_bg = '#2c3e50'
            paper_bg = '#34495e'
            font_color = '#ecf0f1'
            grid_color = 'rgba(255, 255, 255, 0.1)'
            annotation_bg = 'rgba(52, 73, 94, 0.8)'
        else:
            plot_bg = 'white'
            paper_bg = 'white'
            font_color = '#2c3e50'
            grid_color = 'rgba(128, 128, 128, 0.2)'
            annotation_bg = 'rgba(255, 255, 255, 0.8)'

        fig.update_layout(
            title={
                'text': f"{script} - {selected_pattern['pattern_name']} Pattern ({timeframe})",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': font_color}
            },
            xaxis={
                'title': 'Time',
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': grid_color,
                'rangeslider': {'visible': False},
                'color': font_color
            },
            yaxis={
                'title': 'Price',
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': grid_color,
                'color': font_color
            },
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font_color=font_color,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified'
        )

        if timeframe in ['1Min', '5Min']:
            fig.update_xaxes(tickformat='%H:%M')
        elif timeframe in ['10Min', '15Min', '30Min']:
            fig.update_xaxes(tickformat='%m/%d %H:%M')
        else:  # 1Hr
            fig.update_xaxes(tickformat='%m/%d %H:%M')

        print("Chart created successfully")
        return fig

    except Exception as e:
        print(f"Error in create_candlestick_chart: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder='assets'
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Candlestick Pattern Detector", className="text-center my-4", id="main-title")
        ], width=10),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="fas fa-sun"), " Light"],
                    id="light-mode-btn",
                    color="outline-warning",
                    size="sm",
                    className="me-1"
                ),
                dbc.Button(
                    [html.I(className="fas fa-moon"), " Dark"],
                    id="dark-mode-btn", 
                    color="outline-secondary",
                    size="sm"
                )
            ], className="float-end mt-4")
        ], width=2)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Supported Candlestick Patterns"),
                dbc.CardBody([
                    html.Div([
                        html.Img(src="/assets/patterns.png", style={"width": "100%", "maxWidth": "1000px"}),
                    ], className="text-center")
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filter Options"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Script"),
                            dcc.Dropdown(
                                id="script-dropdown",
                                options=[{"label": script, "value": script} for script in SCRIPTS],
                                value=SCRIPTS[0],
                                clearable=False
                            )
                        ], width=4),
                        
                        dbc.Col([
                            html.Label("Pattern"),
                            dcc.Dropdown(
                                id="pattern-dropdown",
                                options=[{"label": pattern, "value": pattern} for pattern in PATTERNS.keys()],
                                value=list(PATTERNS.keys())[0],
                                clearable=False
                            )
                        ], width=4),
                        
                        dbc.Col([
                            html.Label("Timeframe"),
                            dcc.Dropdown(
                                id="timeframe-dropdown",
                                options=[{"label": tf, "value": tf} for tf in TIMEFRAMES],
                                value=TIMEFRAMES[0],
                                clearable=False
                            )
                        ], width=4)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Detect Patterns", id="detect-button", color="primary", className="mt-3")
                        ], width=12)
                    ])
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Pattern Occurrences"),
                dbc.CardBody([
                    html.Div(id="pattern-table-container")
                ])
            ], className="mt-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Candlestick Chart"),
                dbc.CardBody([
                    html.Div([
                        html.P("Click on any pattern row above to display the candlestick chart", 
                               id="chart-instruction", 
                               className="text-muted text-center"),
                        dcc.Loading(
                            id="loading-chart",
                            type="circle",
                            children=[dcc.Graph(id="candlestick-chart")]
                        )
                    ])
                ])
            ], className="mt-4")
        ], width=12)
    ]),
    
    dcc.Store(id="pattern-data"),
    dcc.Store(id="theme-store", data="light")
], fluid=True, id="main-container")

@app.callback(
    [Output("pattern-data", "data"),
     Output("pattern-table-container", "children")],
    [Input("detect-button", "n_clicks")],
    [State("script-dropdown", "value"),
     State("pattern-dropdown", "value"),
     State("timeframe-dropdown", "value")],
    prevent_initial_call=True
)
def update_pattern_data(n_clicks, script, pattern, timeframe):
    if n_clicks is None:
        return None, ""
    
    pattern_df = get_pattern_occurrences(script, timeframe, pattern)
    
    if pattern_df.empty:
        return None, html.Div("No patterns detected with the current filters.")
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Date & Time"),
                html.Th("Script"),
                html.Th("Timeframe"),
                html.Th("Pattern"),
                html.Th("Open"),
                html.Th("High"),
                html.Th("Low"),
                html.Th("Close")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(str(row["datetime"])),
                html.Td(row["script"]),
                html.Td(row["timeframe"]),
                html.Td(row["pattern_name"]),
                html.Td(f"{row['open']:.2f}"),
                html.Td(f"{row['high']:.2f}"),
                html.Td(f"{row['low']:.2f}"),
                html.Td(f"{row['close']:.2f}")
            ], id={"type": "pattern-row", "index": idx}, className="clickable-row", title="Click to view candlestick chart")
            for idx, (i, row) in enumerate(pattern_df.iterrows())
        ])
    ], bordered=True, hover=True, responsive=True)
    
    return pattern_df.to_dict('records'), table

@app.callback(
    [Output("theme-store", "data"),
     Output("light-mode-btn", "color"),
     Output("dark-mode-btn", "color")],
    [Input("light-mode-btn", "n_clicks"),
     Input("dark-mode-btn", "n_clicks")],
    prevent_initial_call=True
)

def switch_theme(light_clicks, dark_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "light", "warning", "outline-secondary"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "light-mode-btn":
        return "light", "warning", "outline-secondary"
    else:
        return "dark", "outline-warning", "secondary"

@app.callback(
    [Output("main-container", "style"),
     Output("main-container", "data-theme")],
    [Input("theme-store", "data")]
)
def update_theme_styles(theme):
    if theme == "dark":
        return {
            "backgroundColor": "#2c3e50",
            "color": "#ecf0f1",
            "minHeight": "100vh"
        }, "dark"
    else:
        return {
            "backgroundColor": "#f8f9fa",
            "color": "#212529",
            "minHeight": "100vh"
        }, "light"

@app.callback(
    [Output("candlestick-chart", "figure"),
     Output("chart-instruction", "style")],
    [Input({"type": "pattern-row", "index": dash.dependencies.ALL}, "n_clicks")],
    [State("pattern-data", "data"),
     State("theme-store", "data")],
    prevent_initial_call=True
)
def update_chart(n_clicks_list, pattern_data, theme):
    if not n_clicks_list or not any(n_clicks_list) or not pattern_data:
        return go.Figure(), {"display": "block"}
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure(), {"display": "block"}
    
    try:
        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
        row_index = json.loads(clicked_id)["index"]
        
        pattern_df = pd.DataFrame(pattern_data)
        
        if row_index >= len(pattern_df):
            print(f"Invalid row index: {row_index}, max index: {len(pattern_df)-1}")
            return go.Figure(), {"display": "block"}
        
        print(f"Row clicked: {row_index}, Total rows: {len(pattern_df)}")
        print(f"Pattern data for row {row_index}: {pattern_df.iloc[row_index].to_dict()}")
        
    except Exception as e:
        print(f"Error processing clicked row: {e}")
        return go.Figure(), {"display": "block"}
    
    fig = create_candlestick_chart(pattern_df, row_index, theme or "light")
    return fig, {"display": "none"}

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Candlestick Pattern Detector</title>
        {%favicon%}
        {%css%}
        <!-- Font Awesome for icons -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            .clickable-row {
                cursor: pointer;
                transition: all 0.2s ease-in-out;
            }
            .clickable-row:hover {
                background-color: #f8f9fa;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Theme-specific hover effects */
            [data-theme="dark"] .clickable-row:hover {
                background-color: #4a6741 !important;
                color: #ecf0f1 !important;
            }
            
            /* Button animations */
            .btn {
                transition: all 0.3s ease;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)