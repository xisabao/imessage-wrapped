"""Generate Plotly visualizations for group chat report."""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLORS = px.colors.qualitative.Set2
BG_COLOR = '#1a1a2e'
PAPER_COLOR = '#16213e'
TEXT_COLOR = '#eaeaea'
GRID_COLOR = '#2d3a4f'


def style_fig(fig):
    """Apply consistent dark theme styling."""
    fig.update_layout(
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family='Inter, sans-serif'),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def create_group_stacked_area(monthly_df, title="Group Chat Volume Over Time"):
    """Create stacked area chart of message volume by group."""
    if monthly_df.empty:
        return None

    pivot = monthly_df.pivot_table(
        index='year_month', columns='group_name', values='count', aggfunc='sum'
    ).fillna(0)

    fig = go.Figure()
    x_values = [str(x) for x in pivot.index.tolist()]

    for i, group in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=x_values,
            y=pivot[group].tolist(),
            mode='lines',
            name=group,
            stackgroup='one',
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
            fillcolor=COLORS[i % len(COLORS)],
            hovertemplate=f'{group}<br>%{{x}}<br>Messages: %{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Messages',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return style_fig(fig)


def create_member_activity_bar(per_member_df, group_name, title=None):
    """Create horizontal bar chart showing member activity within a group."""
    if per_member_df.empty:
        return None

    grp = per_member_df[per_member_df['group_name'] == group_name].copy()
    if grp.empty:
        return None

    grp = grp.sort_values('message_count')
    title = title or f"Who Talks Most: {group_name}"

    fig = go.Figure(go.Bar(
        x=grp['message_count'].tolist(),
        y=grp['sender'].tolist(),
        orientation='h',
        marker_color='#4ecdc4',
        hovertemplate='%{y}<br>Messages: %{x}<br>%{customdata:.1f}% of group<extra></extra>',
        customdata=grp['pct_of_group'].tolist(),
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Messages',
        yaxis_title='',
        height=max(300, len(grp) * 30),
    )

    return style_fig(fig)


def create_member_activity_subplots(per_member_df, top_groups, title="Who Talks Most in Each Group"):
    """Create small multiples showing member activity across top groups."""
    if per_member_df.empty or not top_groups:
        return None

    groups = top_groups[:6]  # Max 6 subplots
    n = len(groups)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[str(g) for g in groups],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    for i, group_name in enumerate(groups):
        row = i // cols + 1
        col = i % cols + 1

        grp = per_member_df[per_member_df['group_name'] == group_name].copy()
        if grp.empty:
            continue

        grp = grp.sort_values('message_count').tail(8)  # Top 8 members

        fig.add_trace(
            go.Bar(
                x=grp['message_count'].tolist(),
                y=grp['sender'].tolist(),
                orientation='h',
                marker_color=COLORS[i % len(COLORS)],
                showlegend=False,
                hovertemplate='%{y}: %{x} msgs<extra></extra>',
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=title,
        height=300 * rows,
    )

    return style_fig(fig)


def create_overlap_network(overlaps, title="Shared Members Across Groups"):
    """Create a network-style visualization showing group overlap."""
    if not overlaps:
        return None

    top_overlaps = overlaps[:15]

    # Collect unique groups
    groups = set()
    for o in top_overlaps:
        groups.add(o['group_a_name'])
        groups.add(o['group_b_name'])
    groups = list(groups)

    # Position groups in a circle
    n = len(groups)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = {g: (np.cos(a), np.sin(a)) for g, a in zip(groups, angles)}

    fig = go.Figure()

    # Draw edges (connections)
    for o in top_overlaps:
        x0, y0 = positions[o['group_a_name']]
        x1, y1 = positions[o['group_b_name']]
        width = max(1, o['shared_count'])

        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=f'rgba(78,205,196,{min(0.8, o["jaccard"] + 0.2)})'),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Draw nodes (groups)
    node_x = [positions[g][0] for g in groups]
    node_y = [positions[g][1] for g in groups]

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=groups,
        textposition='top center',
        textfont=dict(size=10),
        marker=dict(size=20, color='#ff6b6b'),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>',
    ))

    # Add edge labels
    for o in top_overlaps:
        x0, y0 = positions[o['group_a_name']]
        x1, y1 = positions[o['group_b_name']]
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=f"{o['shared_count']} shared",
            font=dict(size=8, color=TEXT_COLOR),
            showarrow=False,
            bgcolor='rgba(0,0,0,0.5)',
        )

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
        height=600,
    )

    return style_fig(fig)


def create_concentration_chart(concentration_df, title="Message Concentration"):
    """Create bar chart showing how concentrated messages are per group."""
    if concentration_df.empty:
        return None

    df = concentration_df.head(15).copy()
    df = df.sort_values('gini')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['top1_pct'].tolist(),
        y=df['group_name'].tolist(),
        orientation='h',
        name='Top 1 person',
        marker_color='#ff6b6b',
        hovertemplate='%{y}<br>Top person: %{x:.1f}% of messages<extra></extra>',
    ))

    fig.add_trace(go.Bar(
        x=(df['top3_pct'] - df['top1_pct']).tolist(),
        y=df['group_name'].tolist(),
        orientation='h',
        name='Top 2-3 people',
        marker_color='#4ecdc4',
        hovertemplate='%{y}<br>Top 3 combined: %{customdata:.1f}%<extra></extra>',
        customdata=df['top3_pct'].tolist(),
    ))

    fig.update_layout(
        title=title,
        xaxis_title='% of Messages',
        yaxis_title='',
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=max(400, len(df) * 35),
    )

    return style_fig(fig)


def create_lifecycle_chart(lifecycle_df, title="Group Chat Lifecycles"):
    """Create visualization showing group lifecycle classifications."""
    if lifecycle_df.empty:
        return None

    color_map = {
        'consistent': '#4ecdc4',
        'burst': '#ff6b6b',
        'rising': '#ffd93d',
        'fading': '#c0c0c0',
        'steady': '#6c5ce7',
    }

    df = lifecycle_df.head(20).copy()
    df = df.sort_values('total_messages')

    fig = go.Figure(go.Bar(
        x=df['total_messages'].tolist(),
        y=df['group_name'].tolist(),
        orientation='h',
        marker_color=[color_map.get(c, '#6c5ce7') for c in df['classification']],
        hovertemplate='%{y}<br>Total: %{x:,} msgs<br>Type: %{customdata[0]}<br>Peak: %{customdata[1]} (%{customdata[2]:,} msgs)<extra></extra>',
        customdata=list(zip(
            df['classification'].tolist(),
            df['peak_year'].tolist(),
            df['peak_year_messages'].tolist(),
        )),
    ))

    # Add legend entries for classifications
    for cls, color in color_map.items():
        if cls in df['classification'].values:
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color=color,
                name=cls.capitalize(),
                showlegend=True,
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Total Messages',
        yaxis_title='',
        height=max(400, len(df) * 30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return style_fig(fig)


if __name__ == "__main__":
    print("Group visualization module loaded successfully")
