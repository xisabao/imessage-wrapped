"""Generate the HTML report for group chat analytics."""
import json
from pathlib import Path
from config import OUTPUT_DIR, START_YEAR, END_YEAR
from report import embed_plotly_chart, COLORS

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iMessage Wrapped: Group Chats {start_year}-{end_year}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', sans-serif;
            background: #F2F2F7;
            color: #1C1C1E;
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}

        header {{
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(180deg, #34C759 0%, #007AFF 100%);
            color: white;
        }}

        .logo-icon {{ font-size: 4rem; margin-bottom: 1rem; opacity: 0.9; }}

        h1 {{ font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: -1px; }}
        .subtitle {{ font-size: 1.2rem; opacity: 0.9; margin-bottom: 2rem; }}

        .hero-stats {{
            display: flex;
            justify-content: center;
            gap: 3rem;
            flex-wrap: wrap;
        }}

        .hero-stat {{
            text-align: center;
            padding: 1.5rem 2rem;
            background: rgba(255,255,255,0.15);
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }}

        .hero-number {{ font-size: 2.8rem; font-weight: 700; }}
        .hero-label {{ font-size: 0.85rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }}

        section {{
            margin: 2rem 0;
            padding: 2rem;
            background: white;
            border-radius: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }}

        .section-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .section-icon {{
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            color: white;
        }}

        .section-icon.blue {{ background: #007AFF; }}
        .section-icon.purple {{ background: #5856D6; }}
        .section-icon.green {{ background: #34C759; }}
        .section-icon.pink {{ background: #FF2D55; }}
        .section-icon.orange {{ background: #FF9500; }}
        .section-icon.teal {{ background: #5AC8FA; }}
        .section-icon.red {{ background: #FF3B30; }}

        h2 {{ font-size: 1.8rem; font-weight: 600; color: #1C1C1E; }}
        h3 {{ font-size: 1.2rem; font-weight: 600; color: #1C1C1E; margin: 1.5rem 0 1rem; }}

        .section-subtitle {{ color: #8E8E93; margin-bottom: 1.5rem; font-size: 1rem; }}

        .podium {{
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 1.5rem;
            margin: 2rem 0;
            padding: 2rem 0;
        }}

        .podium-item {{
            text-align: center;
            padding: 2rem 1.5rem;
            border-radius: 20px;
            min-width: 200px;
            transition: transform 0.3s ease;
        }}

        .podium-item:hover {{ transform: translateY(-5px); }}

        .podium-item.gold {{
            order: 2;
            background: linear-gradient(135deg, #FFD60A, #FFCC00);
            transform: scale(1.1);
            box-shadow: 0 8px 30px rgba(255, 214, 10, 0.4);
        }}

        .podium-item.silver {{ order: 1; background: #E5E5EA; }}

        .podium-item.bronze {{
            order: 3;
            background: linear-gradient(135deg, #FF9500, #FF9F0A);
            color: white;
        }}

        .podium-medal {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
        .podium-name {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem; }}
        .podium-count {{ font-size: 0.9rem; opacity: 0.8; }}
        .podium-detail {{ font-size: 0.8rem; opacity: 0.6; margin-top: 0.25rem; }}

        .contact-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }}

        .contact-card {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.25rem;
            background: #F2F2F7;
            border-radius: 14px;
            transition: all 0.2s ease;
        }}

        .contact-card:hover {{ background: #E5E5EA; transform: translateX(5px); }}

        .contact-rank {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #34C759;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.1rem;
        }}

        .contact-name {{ font-weight: 600; font-size: 1rem; }}
        .contact-stats {{ font-size: 0.85rem; color: #8E8E93; }}

        .chart-container {{
            margin: 1.5rem 0;
            background: #F9F9F9;
            border-radius: 16px;
            padding: 1rem;
        }}

        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.25rem;
        }}

        .insight-card {{
            padding: 1.5rem;
            background: #F2F2F7;
            border-radius: 16px;
            border-left: 4px solid #34C759;
        }}

        .insight-title {{ font-weight: 600; color: #34C759; margin-bottom: 0.75rem; font-size: 1rem; }}
        .insight-content {{ font-size: 0.95rem; color: #3C3C43; line-height: 1.7; }}

        .lifecycle-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}

        .lifecycle-card {{
            padding: 1.25rem;
            border-radius: 14px;
            background: #F2F2F7;
        }}

        .lifecycle-card.consistent {{ border-left: 4px solid #4ecdc4; }}
        .lifecycle-card.burst {{ border-left: 4px solid #ff6b6b; }}
        .lifecycle-card.rising {{ border-left: 4px solid #ffd93d; }}
        .lifecycle-card.fading {{ border-left: 4px solid #c0c0c0; }}
        .lifecycle-card.steady {{ border-left: 4px solid #6c5ce7; }}

        .lifecycle-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }}

        .lifecycle-card.consistent .lifecycle-label {{ color: #4ecdc4; }}
        .lifecycle-card.burst .lifecycle-label {{ color: #ff6b6b; }}
        .lifecycle-card.rising .lifecycle-label {{ color: #ffd93d; }}
        .lifecycle-card.fading .lifecycle-label {{ color: #c0c0c0; }}
        .lifecycle-card.steady .lifecycle-label {{ color: #6c5ce7; }}

        .lifecycle-name {{ font-weight: 600; font-size: 1rem; margin-bottom: 0.25rem; }}
        .lifecycle-stats {{ font-size: 0.85rem; color: #8E8E93; }}

        .concentration-highlight {{
            background: linear-gradient(135deg, rgba(52, 199, 89, 0.1), rgba(0, 122, 255, 0.1));
            border: 1px solid rgba(52, 199, 89, 0.3);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            margin: 1.5rem 0;
        }}

        .concentration-stat {{ font-size: 1.5rem; font-weight: 700; color: #34C759; }}
        .concentration-label {{ font-size: 0.9rem; color: #8E8E93; margin-top: 0.25rem; }}

        .year-section {{
            margin: 1.5rem 0;
            padding: 1.5rem;
            background: #F2F2F7;
            border-radius: 14px;
        }}

        .year-title {{ font-size: 1.2rem; font-weight: 600; color: #34C759; margin-bottom: 0.75rem; }}
        .year-list {{ list-style: none; }}
        .year-list li {{ padding: 0.5rem 0; border-bottom: 1px solid #E5E5EA; font-size: 0.95rem; }}
        .year-list li:last-child {{ border-bottom: none; }}

        footer {{
            text-align: center;
            padding: 3rem 2rem;
            color: #8E8E93;
            font-size: 0.9rem;
        }}

        footer .heart {{ color: #FF2D55; }}

        @media (max-width: 768px) {{
            .podium {{ flex-direction: column; align-items: center; }}
            .podium-item {{ order: unset !important; transform: none !important; }}
            .podium-item.gold {{ transform: none; }}
            h1 {{ font-size: 2.5rem; }}
            .hero-stats {{ gap: 1rem; }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo-icon"><i class="fas fa-users"></i></div>
        <h1>Group Chats Wrapped</h1>
        <p class="subtitle">{start_year} - {end_year}</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-number">{total_messages:,}</div>
                <div class="hero-label">Group Messages</div>
            </div>
            <div class="hero-stat">
                <div class="hero-number">{total_groups:,}</div>
                <div class="hero-label">Group Chats</div>
            </div>
            <div class="hero-stat">
                <div class="hero-number">{total_participants:,}</div>
                <div class="hero-label">People</div>
            </div>
        </div>
    </header>

    <div class="container">
        {sections}
    </div>

    <footer>
        <p>{total_messages:,} group messages. {total_groups:,} group chats. {total_participants:,} people. The chaos, captured.</p>
        <p style="margin-top: 0.5rem;">Generated with <span class="heart">&#9829;</span> by Claude Code</p>
    </footer>
</body>
</html>
"""


def create_group_podium_html(top_groups):
    """Create podium HTML for top 3 group chats."""
    if len(top_groups) < 3:
        return ""
    top3 = top_groups.head(3).to_dict('records')

    def _detail(g):
        senders = g.get('unique_senders', '?')
        years = g.get('years_active', '?')
        return f"{senders} members, {years} years active"

    return f"""
    <div class="podium">
        <div class="podium-item silver">
            <div class="podium-medal">&#129352;</div>
            <div class="podium-name">{top3[1]['group_name']}</div>
            <div class="podium-count">{top3[1]['total_messages']:,} messages</div>
            <div class="podium-detail">{_detail(top3[1])}</div>
        </div>
        <div class="podium-item gold">
            <div class="podium-medal">&#129351;</div>
            <div class="podium-name">{top3[0]['group_name']}</div>
            <div class="podium-count">{top3[0]['total_messages']:,} messages</div>
            <div class="podium-detail">{_detail(top3[0])}</div>
        </div>
        <div class="podium-item bronze">
            <div class="podium-medal">&#129353;</div>
            <div class="podium-name">{top3[2]['group_name']}</div>
            <div class="podium-count">{top3[2]['total_messages']:,} messages</div>
            <div class="podium-detail">{_detail(top3[2])}</div>
        </div>
    </div>
    """


def create_group_grid_html(top_groups, start_rank=4, max_groups=6):
    """Create group grid HTML."""
    cards = []
    group_list = top_groups.iloc[start_rank-1:start_rank-1+max_groups].to_dict('records')
    for i, g in enumerate(group_list, start=start_rank):
        senders = g.get('unique_senders', '?')
        cards.append(f"""
        <div class="contact-card">
            <div class="contact-rank">{i}</div>
            <div class="contact-info">
                <div class="contact-name">{g['group_name']}</div>
                <div class="contact-stats">{g['total_messages']:,} messages &middot; {senders} members</div>
            </div>
        </div>
        """)
    return '<div class="contact-grid">' + ''.join(cards) + '</div>'


def create_top_groups_by_year_html(top_by_year):
    """Create yearly top groups sections."""
    if top_by_year is None or top_by_year.empty:
        return ""

    html_parts = []
    years = sorted(top_by_year['year'].unique())

    for year in years:
        year_data = top_by_year[top_by_year['year'] == year].sort_values('rank').head(5)
        items = [f"<li>{row['group_name']} ({row['total_messages']:,} msgs)</li>"
                 for _, row in year_data.iterrows()]
        html_parts.append(f"""
        <div class="year-section">
            <div class="year-title">{year}</div>
            <ol class="year-list">{''.join(items)}</ol>
        </div>
        """)

    return ''.join(html_parts)


def create_lifecycle_html(lifecycle_df):
    """Create lifecycle classification cards."""
    if lifecycle_df.empty:
        return ""

    label_map = {
        'consistent': 'The Regulars',
        'burst': 'Event / Trip Groups',
        'rising': 'Rising',
        'fading': 'Fading Away',
        'steady': 'Steady',
    }

    cards = []
    for _, row in lifecycle_df.head(15).iterrows():
        cls = row['classification']
        label = label_map.get(cls, cls.capitalize())
        cards.append(f"""
        <div class="lifecycle-card {cls}">
            <div class="lifecycle-label">{label}</div>
            <div class="lifecycle-name">{row['group_name']}</div>
            <div class="lifecycle-stats">
                {row['total_messages']:,} messages &middot; Peak: {row['peak_year']} ({row['peak_year_messages']:,} msgs) &middot; {row['years_active']} years active
            </div>
        </div>
        """)

    return '<div class="lifecycle-grid">' + ''.join(cards) + '</div>'


def create_concentration_summary_html(concentration_df):
    """Create concentration summary highlights."""
    if concentration_df.empty:
        return ""

    avg_gini = concentration_df['gini'].mean()
    avg_top1 = concentration_df['top1_pct'].mean()
    avg_top3 = concentration_df['top3_pct'].mean()

    most_equal = concentration_df.nsmallest(1, 'gini')
    most_dominated = concentration_df.nlargest(1, 'gini')

    html = f"""
    <div class="concentration-highlight">
        <div class="concentration-stat">On average, the top person sends {avg_top1:.0f}% of messages</div>
        <div class="concentration-label">Top 3 people account for {avg_top3:.0f}% of all group messages</div>
    </div>
    """

    if not most_equal.empty:
        g = most_equal.iloc[0]
        html += f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
            <div class="lifecycle-card consistent">
                <div class="lifecycle-label">Most Democratic</div>
                <div class="lifecycle-name">{g['group_name']}</div>
                <div class="lifecycle-stats">Top person: {g['top1_pct']:.0f}% &middot; {g['total_members']} members</div>
            </div>
    """

    if not most_dominated.empty:
        g = most_dominated.iloc[0]
        html += f"""
            <div class="lifecycle-card burst">
                <div class="lifecycle-label">Most Dominated</div>
                <div class="lifecycle-name">{g['group_name']}</div>
                <div class="lifecycle-stats">Top person: {g['top1_pct']:.0f}% &middot; {g['total_members']} members</div>
            </div>
        </div>
        """

    return html


def create_insight_cards_html(insights):
    """Create insight cards from generated insights."""
    if not insights:
        return ""

    cards = []
    for i, (title, content) in enumerate(insights[:20], 1):
        cards.append(f"""
        <div class="insight-card">
            <div class="insight-title">{i}. {title}</div>
            <div class="insight-content">{content}</div>
        </div>
        """)
    return '<div class="insights-grid">' + ''.join(cards) + '</div>'


def generate_group_report(total_messages, total_groups, total_participants,
                          top_groups, charts, top_by_year=None,
                          lifecycle_df=None, concentration_df=None,
                          insights=None):
    """Generate the complete group chat HTML report."""
    sections = []
    num_years = END_YEAR - START_YEAR

    # Section 1: Top Groups
    section1 = f"""
    <section>
        <div class="section-header">
            <div class="section-icon green"><i class="fas fa-trophy"></i></div>
            <h2>Your Top Group Chats</h2>
        </div>
        {create_group_podium_html(top_groups)}
        {create_group_grid_html(top_groups, start_rank=4, max_groups=6)}
    </section>
    """
    sections.append(section1)

    # Section 2: Groups Over Time
    stacked_chart = ""
    if 'group_stacked_area' in charts and charts['group_stacked_area'] is not None:
        stacked_chart = embed_plotly_chart(charts['group_stacked_area'], 'group-stacked-chart', height=500)

    section2 = f"""
    <section>
        <div class="section-header">
            <div class="section-icon blue"><i class="fas fa-chart-area"></i></div>
            <h2>Groups Over Time</h2>
        </div>
        <p class="section-subtitle">How your group chats have evolved year by year.</p>
        {stacked_chart}
        <h3>Top Groups Each Year</h3>
        {create_top_groups_by_year_html(top_by_year)}
    </section>
    """
    sections.append(section2)

    # Section 3: Who Talks Most
    member_chart = ""
    if 'member_activity' in charts and charts['member_activity'] is not None:
        member_chart = embed_plotly_chart(charts['member_activity'], 'member-activity-chart', height=800)

    section3 = f"""
    <section>
        <div class="section-header">
            <div class="section-icon orange"><i class="fas fa-bullhorn"></i></div>
            <h2>Who Talks Most</h2>
        </div>
        <p class="section-subtitle">The loudest voices in each group chat.</p>
        {member_chart}
    </section>
    """
    sections.append(section3)

    # Section 4: Social Circles (Overlap)
    overlap_chart = ""
    if 'overlap_network' in charts and charts['overlap_network'] is not None:
        overlap_chart = embed_plotly_chart(charts['overlap_network'], 'overlap-chart', height=600)

    if overlap_chart:
        section4 = f"""
        <section>
            <div class="section-header">
                <div class="section-icon purple"><i class="fas fa-project-diagram"></i></div>
                <h2>Your Social Circles</h2>
            </div>
            <p class="section-subtitle">People who appear in multiple group chats and how your circles intersect.</p>
            {overlap_chart}
        </section>
        """
        sections.append(section4)

    # Section 5: Conversation Concentration
    concentration_chart = ""
    if 'concentration' in charts and charts['concentration'] is not None:
        concentration_chart = embed_plotly_chart(charts['concentration'], 'concentration-chart', height=500)

    if concentration_df is not None and not concentration_df.empty:
        section5 = f"""
        <section>
            <div class="section-header">
                <div class="section-icon pink"><i class="fas fa-chart-pie"></i></div>
                <h2>Conversation Concentration</h2>
            </div>
            <p class="section-subtitle">How evenly do people participate in each group?</p>
            {create_concentration_summary_html(concentration_df)}
            {concentration_chart}
        </section>
        """
        sections.append(section5)

    # Section 6: Consistent vs Burst
    lifecycle_chart = ""
    if 'lifecycle' in charts and charts['lifecycle'] is not None:
        lifecycle_chart = embed_plotly_chart(charts['lifecycle'], 'lifecycle-chart', height=600)

    if lifecycle_df is not None and not lifecycle_df.empty:
        section6 = f"""
        <section>
            <div class="section-header">
                <div class="section-icon teal"><i class="fas fa-clock"></i></div>
                <h2>The Regulars vs. The Events</h2>
            </div>
            <p class="section-subtitle">Some group chats are forever. Others burn bright for a trip or an event.</p>
            {create_lifecycle_html(lifecycle_df)}
            {lifecycle_chart}
        </section>
        """
        sections.append(section6)

    # Section 7: Group Insights
    if insights:
        insights_html = create_insight_cards_html(insights)
        if insights_html:
            section7 = f"""
            <section>
                <div class="section-header">
                    <div class="section-icon teal"><i class="fas fa-lightbulb"></i></div>
                    <h2>Group Chat Insights</h2>
                </div>
                {insights_html}
            </section>
            """
            sections.append(section7)

    # Generate final HTML
    html = HTML_TEMPLATE.format(
        start_year=START_YEAR,
        end_year=END_YEAR,
        total_messages=total_messages,
        total_groups=total_groups,
        total_participants=total_participants,
        sections=''.join(sections),
    )

    return html


def save_group_report(html, filename="wrapped_groups.html"):
    """Save HTML report to file."""
    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Group report saved to: {output_path}")
    return output_path
