"""Generate Plotly visualizations for the report."""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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

def create_bump_chart(rankings_df, title="Contact Rankings Over Time"):
    """Create bump chart showing ranking changes over years."""
    fig = go.Figure()

    contacts = rankings_df['contact_name'].unique().tolist()

    for i, contact in enumerate(contacts):
        contact_data = rankings_df[rankings_df['contact_name'] == contact]
        fig.add_trace(go.Scatter(
            x=contact_data['year'].tolist(),
            y=contact_data['rank'].tolist(),
            mode='lines+markers',
            name=contact,
            line=dict(width=3, color=COLORS[i % len(COLORS)]),
            marker=dict(size=10),
            hovertemplate=f'{contact}<br>Year: %{{x}}<br>Rank: %{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Rank',
        yaxis=dict(autorange='reversed', dtick=1),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return style_fig(fig)

def create_stacked_area(monthly_df, title="Message Volume Over Time"):
    """Create stacked area chart of message volume by contact."""
    pivot = monthly_df.pivot(index='year_month', columns='contact_name', values='count').fillna(0)

    fig = go.Figure()

    # Convert index to string for proper JSON serialization
    x_values = [str(x) for x in pivot.index.tolist()]

    for i, contact in enumerate(pivot.columns):
        fig.add_trace(go.Scatter(
            x=x_values,
            y=pivot[contact].tolist(),
            mode='lines',
            name=contact,
            stackgroup='one',
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
            fillcolor=COLORS[i % len(COLORS)],
            hovertemplate=f'{contact}<br>%{{x}}<br>Messages: %{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Messages',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return style_fig(fig)

def create_lopsidedness_scatter(lopsidedness_df, title="Conversation Balance"):
    """Create scatter plot of lopsidedness vs total messages."""
    df = lopsidedness_df.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['total'].tolist(),
        y=df['lopsidedness'].tolist(),
        mode='markers+text',
        text=df['contact_name'].tolist(),
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(
            size=10,
            color=df['lopsidedness'].tolist(),
            colorscale='RdYlGn',
            cmid=1.0,
            showscale=True,
            colorbar=dict(title='Lopsidedness'),
        ),
        hovertemplate='%{text}<br>Total: %{x}<br>Lopsidedness: %{y:.2f}<extra></extra>'
    ))

    fig.add_hline(y=1.0, line_dash='dash', line_color='white', opacity=0.5,
                  annotation_text='Balanced', annotation_position='right')

    fig.update_layout(
        title=title,
        xaxis_title='Total Messages',
        yaxis_title='Lopsidedness Ratio (>1 = you send more)',
        xaxis_type='log',
        yaxis_type='log',
        showlegend=False,
    )

    return style_fig(fig)

def create_hour_day_heatmap(heatmap_df, title="When You Text"):
    """Create hour x day of week heatmap."""
    # Convert to lists for proper JSON serialization
    z_data = heatmap_df.values.tolist()
    y_labels = list(heatmap_df.index)
    x_labels = [f"{h}:00" for h in range(24)]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        hovertemplate='%{y} at %{x}<br>Messages: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        xaxis=dict(dtick=2),
    )

    return style_fig(fig)

def create_yearly_volume_bar(yearly_df, title="Messages Per Year"):
    """Create bar chart of sent vs received per year."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=yearly_df['year'].tolist(),
        y=yearly_df['sent'].tolist(),
        name='Sent',
        marker_color='#4ecdc4',
    ))

    fig.add_trace(go.Bar(
        x=yearly_df['year'].tolist(),
        y=yearly_df['received'].tolist(),
        name='Received',
        marker_color='#ff6b6b',
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Messages',
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )

    return style_fig(fig)

def create_peak_hours_small_multiples(hourly_by_year_df, title="Peak Hours By Year"):
    """Create small multiples showing hourly distribution per year."""
    years = sorted(hourly_by_year_df['year'].unique().tolist())
    n_years = len(years)
    cols = 3
    rows = (n_years + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[str(y) for y in years],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for i, year in enumerate(years):
        row = i // cols + 1
        col = i % cols + 1

        year_data = hourly_by_year_df[hourly_by_year_df['year'] == year]

        fig.add_trace(
            go.Bar(
                x=year_data['hour'].tolist(),
                y=year_data['count'].tolist(),
                marker_color='#4ecdc4',
                showlegend=False,
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=title,
        height=200 * rows,
    )

    return style_fig(fig)

def create_sentiment_bar(sentiment_df, title="Best Vibes", top_n=15, worst=False):
    """Create horizontal bar chart of sentiment by contact."""
    if sentiment_df is None or sentiment_df.empty:
        return None

    if worst:
        # Get worst (lowest) sentiment contacts
        df = sentiment_df.tail(top_n).copy()
        title = "Worst Vibes" if title == "Best Vibes" else title
    else:
        # Get best (highest) sentiment contacts
        df = sentiment_df.head(top_n).copy()

    if df.empty:
        return None

    df = df.sort_values('avg_sentiment')

    sentiments = df['avg_sentiment'].tolist()
    colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in sentiments]

    fig = go.Figure(go.Bar(
        x=sentiments,
        y=df['contact_name'].tolist(),
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}<br>Sentiment: %{x:.3f}<extra></extra>'
    ))

    fig.add_vline(x=0, line_color='white', opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title='Average Sentiment (negative ← → positive)',
        yaxis_title='',
        height=max(400, top_n * 30),
    )

    return style_fig(fig)

def create_emoji_grid(emoji_df, title="Your Emoji Story"):
    """Create emoji grid by year."""
    years = sorted(emoji_df['year'].unique().tolist())

    fig = go.Figure()

    for i, year in enumerate(years):
        year_emojis = emoji_df[emoji_df['year'] == year].sort_values('rank').head(5)
        for j, (_, row) in enumerate(year_emojis.iterrows()):
            fig.add_annotation(
                x=j, y=len(years) - i - 1,
                text=str(row['emojis']),
                font=dict(size=24),
                showarrow=False,
            )

    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 5.5]),
        yaxis=dict(
            showgrid=False, zeroline=False,
            tickmode='array',
            tickvals=list(range(len(years))),
            ticktext=[str(y) for y in reversed(years)],
        ),
        height=50 * len(years) + 100,
    )

    return style_fig(fig)

def create_question_ratio_line(question_df, title="Questions Over Time"):
    """Create line chart of question percentage by year."""
    if question_df is None or question_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=question_df['year'].tolist(),
        y=question_df['question_pct'].tolist(),
        mode='lines+markers',
        line=dict(color='#4ecdc4', width=3, shape='spline'),
        marker=dict(size=10),
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='% of Messages That Are Questions',
    )

    return style_fig(fig)

def create_monthly_top_contacts(monthly_data, title="Top Contact Each Month (2025)"):
    """Create a bar chart showing top contact per month."""
    months = monthly_data['month_name'].tolist()
    contacts = monthly_data['contact_name'].tolist()
    counts = monthly_data['count'].tolist()

    fig = go.Figure(go.Bar(
        x=months,
        y=counts,
        text=contacts,
        textposition='inside',
        marker_color='#4ecdc4',
        hovertemplate='%{x}<br>%{text}: %{y} messages<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Messages',
    )

    return style_fig(fig)

def create_initiators_bar(initiators_df, title="Who Initiates Conversations", top_n=15):
    """Create horizontal bar chart showing who initiates more conversations."""
    if initiators_df.empty:
        return None
    
    df = initiators_df.head(top_n).copy()
    if df.empty:
        return None
    
    df = df.sort_values('you_initiate_pct')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['you_initiate_pct'].tolist(),
        y=df['contact_name'].tolist(),
        orientation='h',
        marker_color='#4ecdc4',
        hovertemplate='%{y}<br>You initiate: %{x:.1f}%<br>Total conversations: %{customdata}<extra></extra>',
        customdata=df['total_conversations'].tolist(),
    ))
    
    fig.add_vline(x=50, line_dash='dash', line_color='white', opacity=0.5,
                  annotation_text='50% (balanced)', annotation_position='right')
    
    fig.update_layout(
        title=title,
        xaxis_title='% of Conversations You Initiated',
        yaxis_title='',
        height=max(400, top_n * 30),
    )
    
    return style_fig(fig)

def create_response_times_scatter(response_times_df, title="Response Times", top_n=20):
    """Create scatter plot comparing your vs their response times."""
    df = response_times_df.dropna(subset=['your_response_time_min', 'their_response_time_min']).head(top_n).copy()
    
    if len(df) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['your_response_time_min'].tolist(),
        y=df['their_response_time_min'].tolist(),
        mode='markers+text',
        text=df['contact_name'].tolist(),
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(size=12, color='#4ecdc4'),
        hovertemplate='%{text}<br>You: %{x:.0f} min<br>Them: %{y:.0f} min<extra></extra>'
    ))
    
    # Add diagonal line (equal response times)
    max_time = max(df['your_response_time_min'].max(), df['their_response_time_min'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_time],
        y=[0, max_time],
        mode='lines',
        line=dict(dash='dash', color='white'),
        opacity=0.5,
        showlegend=False,
        hovertemplate='Equal response times<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Your Response Time (minutes)',
        yaxis_title='Their Response Time (minutes)',
        xaxis_type='log',
        yaxis_type='log',
        showlegend=False,
    )
    
    return style_fig(fig)

def create_streaks_bar(streaks_df, title="Longest Texting Streaks", top_n=10):
    """Create horizontal bar chart of longest streaks."""
    if streaks_df.empty:
        return None
    
    df = streaks_df.head(top_n).copy()
    if df.empty:
        return None
    
    df = df.sort_values('streak_length')
    
    # Format date range for hover
    hover_text = [f"{row['contact_name']}<br>{row['start_date']} to {row['end_date']}<br>{row['streak_length']} days" 
                  for _, row in df.iterrows()]
    
    fig = go.Figure(go.Bar(
        x=df['streak_length'].tolist(),
        y=df['contact_name'].tolist(),
        orientation='h',
        marker_color='#ff6b6b',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text,
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Days in Streak',
        yaxis_title='',
        height=max(400, top_n * 30),
    )
    
    return style_fig(fig)

def create_emoji_by_contact_grid(emoji_df, title="Top Emojis by Contact", top_contacts=10):
    """Create grid showing top emojis for each contact."""
    if emoji_df.empty:
        return None
    
    contacts = emoji_df['contact_name'].unique()[:top_contacts]
    if len(contacts) == 0:
        return None
    
    fig = go.Figure()
    
    max_emojis = 5
    for i, contact in enumerate(contacts):
        contact_emojis = emoji_df[emoji_df['contact_name'] == contact].sort_values('rank').head(max_emojis)
        for j, (_, row) in enumerate(contact_emojis.iterrows()):
            fig.add_annotation(
                x=j,
                y=len(contacts) - i - 1,
                text=str(row['emojis']),
                font=dict(size=20),
                showarrow=False,
            )
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, max_emojis]),
        yaxis=dict(
            showgrid=False, zeroline=False,
            tickmode='array',
            tickvals=list(range(len(contacts))),
            ticktext=list(reversed(contacts)),
        ),
        height=50 * len(contacts) + 100,
    )
    
    return style_fig(fig)

def create_question_by_contact_bar(question_df, title="Question Rate by Contact", top_n=15):
    """Create horizontal bar chart of question percentage by contact."""
    if question_df is None or question_df.empty:
        return None
    
    df = question_df.head(top_n).copy()
    if df.empty:
        return None
    
    df = df.sort_values('question_pct')
    
    fig = go.Figure(go.Bar(
        x=df['question_pct'].tolist(),
        y=df['contact_name'].tolist(),
        orientation='h',
        marker_color='#ff6b6b',
        hovertemplate='%{y}<br>Questions: %{x:.1f}%<br>Total messages: %{customdata}<extra></extra>',
        customdata=df['total'].tolist(),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='% of Messages That Are Questions',
        yaxis_title='',
        height=max(400, top_n * 30),
    )
    
    return style_fig(fig)

def create_unique_words_timeline(unique_words_df, title="Unique Words by Year", words_per_year=5):
    """Create timeline visualization of unique words that spiked each year."""
    if unique_words_df.empty:
        return None
    
    years = sorted(unique_words_df['year'].unique())
    if len(years) == 0:
        return None
    
    fig = go.Figure()
    
    # Get top words per year
    for year in years:
        year_words = unique_words_df[unique_words_df['year'] == year].sort_values('tfidf_score', ascending=False).head(words_per_year)
        for i, (_, row) in enumerate(year_words.iterrows()):
            fig.add_annotation(
                x=year,
                y=i,
                text=row['word'],
                font=dict(size=12, color=COLORS[i % len(COLORS)]),
                showarrow=False,
                bgcolor='rgba(255,255,255,0.1)',
                bordercolor=COLORS[i % len(COLORS)],
                borderwidth=1,
            )
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, words_per_year]),
        height=200,
    )
    
    return style_fig(fig)

def create_topics_by_contact_display(topics_df, title="Topics by Contact"):
    """Create text-based display of topics for each contact."""
    if topics_df.empty:
        return None
    
    contacts = topics_df['contact_name'].unique()
    if len(contacts) == 0:
        return None
    
    # This is more of a text display, so we'll create a simple visualization
    # showing topics as annotations
    fig = go.Figure()
    
    y_pos = 0
    for contact in contacts[:10]:  # Top 10 contacts
        contact_topics = topics_df[topics_df['contact_name'] == contact]
        topics_text = []
        for _, row in contact_topics.iterrows():
            topics_text.append(row['top_words'])
        
        # Add contact name
        fig.add_annotation(
            x=0,
            y=y_pos,
            text=f"<b>{contact}</b>",
            font=dict(size=14, color='black'),
            showarrow=False,
            xanchor='left',
        )
        
        # Add topics
        for i, topic_text in enumerate(topics_text[:3]):  # Top 3 topics
            fig.add_annotation(
                x=0.3,
                y=y_pos - i * 0.15,
                text=topic_text,
                font=dict(size=10, color='black'),
                showarrow=False,
                xanchor='left',
            )
        
        y_pos -= 0.6
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[y_pos - 0.5, 0.5]),
        height=max(400, abs(y_pos) * 100),
    )
    
    return style_fig(fig)

if __name__ == "__main__":
    print("Visualization module loaded successfully")
