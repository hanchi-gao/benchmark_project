"""Dash web application for benchmark visualization."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import (
    get_experiment_folders,
    load_multiple_experiments,
    get_all_metadata_values,
    filter_folders_by_metadata,
)
from .config import CHART_TABS, X_AXIS_FIELD, X_AXIS_LABEL, OUTPUT_DIR


def create_app(output_dir: str = OUTPUT_DIR) -> dash.Dash:
    """
    Create and configure the Dash application.

    Args:
        output_dir: Directory containing benchmark results

    Returns:
        Configured Dash application
    """
    app = dash.Dash(__name__)
    app.title = "vLLM Benchmark Results"

    # Get initial data
    available_folders = get_experiment_folders(output_dir)
    metadata_values = get_all_metadata_values(output_dir)

    # Color palette for consistent experiment coloring
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ]

    def create_single_chart(data_dict, y_field, y_label, title):
        """Create a single metric chart."""
        fig = go.Figure()

        for i, (folder_name, df) in enumerate(data_dict.items()):
            if X_AXIS_FIELD in df.columns and y_field in df.columns:
                df_sorted = df.sort_values(X_AXIS_FIELD)
                color = color_palette[i % len(color_palette)]

                fig.add_trace(go.Scatter(
                    x=df_sorted[X_AXIS_FIELD],
                    y=df_sorted[y_field],
                    mode='lines+markers',
                    name=folder_name,
                    line=dict(color=color),
                    marker=dict(color=color),
                    hovertemplate=(
                        f'<b>{folder_name}</b><br>' +
                        f'{X_AXIS_LABEL}: %{{x}}<br>' +
                        f'{y_label}: %{{y:.2f}}<br>' +
                        '<extra></extra>'
                    )
                ))

        fig.update_layout(
            title=title,
            xaxis_title=X_AXIS_LABEL,
            yaxis_title=y_label,
            xaxis_type="linear",
            hovermode='x unified',
            template="plotly_white",
            height=500,
        )

        return fig

    def create_overview_chart(data_dict):
        """Create 2x3 overview chart with all key metrics."""
        overview_config = CHART_TABS["Overview"]
        charts = overview_config["charts"]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[chart["title"] for chart in charts],
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )

        # Assign fixed colors to each experiment
        folder_colors = {
            folder: color_palette[i % len(color_palette)]
            for i, folder in enumerate(data_dict.keys())
        }

        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]

        for idx, (chart_config, (row, col)) in enumerate(zip(charts, positions)):
            y_field = chart_config["y_field"]
            y_label = chart_config["y_label"]

            for folder_name, df in data_dict.items():
                if X_AXIS_FIELD in df.columns and y_field in df.columns:
                    df_sorted = df.sort_values(X_AXIS_FIELD)

                    fig.add_trace(
                        go.Scatter(
                            x=df_sorted[X_AXIS_FIELD],
                            y=df_sorted[y_field],
                            mode='lines+markers',
                            name=folder_name,
                            legendgroup=folder_name,
                            showlegend=(idx == 0),  # Only show legend for first subplot
                            line=dict(color=folder_colors[folder_name]),
                            marker=dict(color=folder_colors[folder_name]),
                            hovertemplate=(
                                f'<b>{folder_name}</b><br>' +
                                f'{X_AXIS_LABEL}: %{{x}}<br>' +
                                f'{y_label}: %{{y:.2f}}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        row=row, col=col
                    )

            fig.update_xaxes(title_text=X_AXIS_LABEL, type="linear", row=row, col=col)
            fig.update_yaxes(title_text=y_label, row=row, col=col)

        fig.update_layout(
            height=900,
            template="plotly_white",
            hovermode='x unified',
        )

        return fig

    # Application layout
    app.layout = html.Div([
        # Store for output directory
        dcc.Store(id='output-dir-store', data=output_dir),

        # Header
        html.H1(
            "vLLM Benchmark Results",
            style={'textAlign': 'center', 'marginTop': 20, 'color': '#333'}
        ),

        # Metadata filters
        html.Div([
            html.H3("Filters"),
            html.Div([
                html.Div([
                    html.Label("Software Version"),
                    dcc.Dropdown(
                        id='rocm-version-filter',
                        options=[{'label': str(v), 'value': v} for v in metadata_values.get('rocm_version', [])],
                        placeholder="All Versions",
                        clearable=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

                html.Div([
                    html.Label("TP Size"),
                    dcc.Dropdown(
                        id='tp-filter',
                        options=[{'label': f'TP={v}', 'value': v} for v in metadata_values.get('tensor_parallel_size', [])],
                        placeholder="All",
                        clearable=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

                html.Div([
                    html.Label("GPUs"),
                    dcc.Dropdown(
                        id='gpu-count-filter',
                        options=[{'label': f'{v}x GPU', 'value': v} for v in metadata_values.get('gpu_count', [])],
                        placeholder="All",
                        clearable=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),

                html.Div([
                    html.Label("Model"),
                    dcc.Dropdown(
                        id='model-filter',
                        options=[{'label': str(v), 'value': v} for v in metadata_values.get('model_name', [])],
                        placeholder="All",
                        clearable=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '24%', 'display': 'inline-block'}),
            ], style={'marginBottom': 15}),
        ], style={
            'margin': '20px',
            'padding': '15px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'border': '1px solid #dee2e6'
        }),

        # Experiment selection
        html.Div([
            html.H3("Select Experiments"),
            html.Div([
                html.Button('Select All', id='select-all-btn', n_clicks=0,
                           style={'marginRight': '10px', 'padding': '5px 15px'}),
                html.Button('Clear All', id='clear-all-btn', n_clicks=0,
                           style={'padding': '5px 15px'}),
            ], style={'marginBottom': '10px'}),
            dcc.Checklist(
                id='folder-checklist',
                options=[{'label': folder, 'value': folder} for folder in available_folders],
                value=[],
                style={'maxHeight': '300px', 'overflowY': 'auto'}
            ),
        ], style={
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#f0f0f0',
            'borderRadius': '5px'
        }),

        # Chart tabs
        html.Div([
            dcc.Tabs(
                id='chart-tabs',
                value='Overview',
                children=[
                    dcc.Tab(label=tab_name, value=tab_name)
                    for tab_name in CHART_TABS.keys()
                ]
            ),
        ], style={'margin': '20px'}),

        # Chart display
        html.Div([
            dcc.Graph(id='chart-display')
        ], style={'margin': '20px'}),

        # Footer
        html.Div([
            html.Hr(),
            html.P(
                "vLLM Benchmark Tool - Visualization Dashboard",
                style={'textAlign': 'center', 'color': '#666', 'fontSize': '14px'}
            )
        ], style={'margin': '20px'}),
    ])

    # Callback to filter available folders based on metadata
    @app.callback(
        Output('folder-checklist', 'options'),
        [Input('rocm-version-filter', 'value'),
         Input('tp-filter', 'value'),
         Input('gpu-count-filter', 'value'),
         Input('model-filter', 'value'),
         Input('output-dir-store', 'data')]
    )
    def update_folder_options(rocm_version, tp_size, gpu_count, model_name, out_dir):
        all_folders = get_experiment_folders(out_dir)
        filtered = filter_folders_by_metadata(
            all_folders,
            rocm_version=rocm_version,
            tensor_parallel_size=tp_size,
            gpu_count=gpu_count,
            model_name=model_name
        )
        return [{'label': folder, 'value': folder} for folder in filtered]

    # Callback for select all / clear all buttons
    @app.callback(
        Output('folder-checklist', 'value'),
        [Input('select-all-btn', 'n_clicks'),
         Input('clear-all-btn', 'n_clicks')],
        [State('folder-checklist', 'options'),
         State('folder-checklist', 'value')]
    )
    def handle_select_buttons(select_clicks, clear_clicks, options, current_value):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_value or []

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'select-all-btn':
            return [opt['value'] for opt in options]
        elif button_id == 'clear-all-btn':
            return []

        return current_value or []

    # Callback to update chart
    @app.callback(
        Output('chart-display', 'figure'),
        [Input('folder-checklist', 'value'),
         Input('chart-tabs', 'value'),
         Input('output-dir-store', 'data')]
    )
    def update_chart(selected_folders, selected_tab, out_dir):
        if not selected_folders:
            fig = go.Figure()
            fig.update_layout(
                title="Please select at least one experiment folder",
                template="plotly_white",
                height=500,
            )
            return fig

        # Load selected experiment data
        data_dict = load_multiple_experiments(selected_folders, out_dir)

        if not data_dict:
            fig = go.Figure()
            fig.update_layout(
                title="Unable to load data",
                template="plotly_white",
                height=500,
            )
            return fig

        # Create chart based on selected tab
        tab_config = CHART_TABS[selected_tab]

        if tab_config["type"] == "overview":
            return create_overview_chart(data_dict)
        else:
            return create_single_chart(
                data_dict,
                tab_config["y_field"],
                tab_config["y_label"],
                tab_config["title"]
            )

    return app


def run_app(output_dir: str = OUTPUT_DIR, host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    """
    Run the Dash application.

    Args:
        output_dir: Directory containing benchmark results
        host: Host address to bind
        port: Port number
        debug: Enable debug mode
    """
    app = create_app(output_dir)

    folders = get_experiment_folders(output_dir)
    print(f"Found {len(folders)} experiment folders")
    print(f"Starting web UI at http://{host}:{port}")
    print(f"Local access: http://localhost:{port}")

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
