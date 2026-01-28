"""Main orchestration script for iMessage Wrapped."""
import pandas as pd
from pathlib import Path
import json

from config import DATA_DIR, START_YEAR, END_YEAR, EXCLUDED_CONTACTS, MIN_TWO_WAY_RATIO, MIN_MESSAGES_FOR_SENTIMENT, BORING_WORDS
import re
from extract import extract_messages
from contacts import (
    get_contacts_from_macos,
    create_contact_mappings,
    save_contact_mappings,
    prompt_for_unresolved,
)
from analysis.people import (
    get_top_contacts_alltime,
    get_top_contacts_by_year,
    calculate_lopsidedness,
    get_conversation_initiator_stats,
    calculate_response_times,
    find_rising_stars,
    find_faded_connections,
    get_message_volume_over_time,
)
from analysis.temporal import (
    get_hour_day_heatmap,
    get_peak_hours_by_year,
    get_yearly_volume,
    get_longest_streaks,
)
from analysis.content import (
    get_top_emojis_by_year,
    get_emoji_by_contact,
    get_question_ratio_by_year,
    get_question_ratio_by_contact,
    get_sentiment_by_contact,
    get_top_phrases_by_year,
    get_unique_words_by_year,
    get_topics_by_year,
    get_topics_by_contact,
    add_sentiment_to_df,
)
from visualize import (
    create_bump_chart,
    create_stacked_area,
    create_lopsidedness_scatter,
    create_hour_day_heatmap,
    create_yearly_volume_bar,
    create_peak_hours_small_multiples,
    create_sentiment_bar,
    create_emoji_grid,
    create_question_ratio_line,
    create_monthly_top_contacts,
    create_initiators_bar,
    create_response_times_scatter,
    create_streaks_bar,
    create_emoji_by_contact_grid,
    create_question_by_contact_bar,
    create_unique_words_timeline,
    create_topics_by_contact_display,
)
from report import generate_report, save_report


def main():
    print("=" * 60)
    print("iMessage Wrapped Generator")
    print("=" * 60)

    # Step 1: Extract messages
    print("\n[1/8] Extracting messages...")
    df = extract_messages()

    # Step 2: Resolve contacts
    print("\n[2/8] Resolving contacts...")
    contacts_map = get_contacts_from_macos()
    mappings = create_contact_mappings(df, contacts_map)
    save_contact_mappings(mappings)
    df['contact_name'] = df['contact_id'].astype(str).map(mappings)

    unresolved = prompt_for_unresolved(df, mappings, top_n=20)
    if unresolved:
        print("\nNote: Some top contacts couldn't be resolved to names:")
        for contact_id, count in unresolved[:5]:
            print(f"  {contact_id}: {count:,} messages")
        print("  (You can manually edit output/data/contacts.json to add names)")

    # Filter out excluded contacts (self, businesses, etc.)
    before_filter = len(df)
    df = df[~df['contact_name'].str.lower().isin([c.lower() for c in EXCLUDED_CONTACTS])]
    print(f"\nFiltered out excluded contacts: {before_filter - len(df):,} messages removed")

    # Filter out contacts that look like phone numbers or short codes (keep only named contacts)
    def is_phone_or_code(name):
        if not name:
            return True
        name_str = str(name).strip()
        # If it's all digits, it's a phone number or short code
        if name_str.isdigit():
            return True
        # If it starts with +, it's a phone number
        if name_str.startswith('+'):
            return True
        # If it's mostly digits (more than 50% digits), treat as number
        digits = re.sub(r'\D', '', name_str)
        if len(digits) > 0 and len(digits) / len(name_str) > 0.5:
            return True
        return False

    before_filter = len(df)
    df = df[~df['contact_name'].apply(is_phone_or_code)]
    print(f"Filtered out unnamed contacts (numbers/codes): {before_filter - len(df):,} messages removed")

    # Filter out one-sided contacts (notifications, etc.)
    contact_stats = df.groupby('contact_name').agg(
        total=('message_id', 'count'),
        sent=('is_from_me', 'sum')
    )
    contact_stats['received'] = contact_stats['total'] - contact_stats['sent']
    contact_stats['sent_ratio'] = contact_stats['sent'] / contact_stats['total']

    # Keep contacts where both sent and received are at least MIN_TWO_WAY_RATIO of total
    two_way_contacts = contact_stats[
        (contact_stats['sent_ratio'] >= MIN_TWO_WAY_RATIO) &
        (contact_stats['sent_ratio'] <= (1 - MIN_TWO_WAY_RATIO))
    ].index.tolist()

    before_filter = len(df)
    df = df[df['contact_name'].isin(two_way_contacts)]
    print(f"Filtered out one-sided contacts: {before_filter - len(df):,} messages removed")
    print(f"Remaining: {len(df):,} messages with {df['contact_name'].nunique()} contacts")

    # Step 3: People analysis
    print("\n[3/8] Analyzing relationships...")
    top_contacts = get_top_contacts_alltime(df)
    top_by_year = get_top_contacts_by_year(df)
    lopsidedness = calculate_lopsidedness(df)
    initiators = get_conversation_initiator_stats(df)
    response_times = calculate_response_times(df)

    top_names = top_contacts['contact_name'].head(10).tolist()
    monthly_volume = get_message_volume_over_time(df, contacts=top_names)

    # Step 4: Temporal analysis
    print("\n[4/8] Analyzing temporal patterns...")
    heatmap_data = get_hour_day_heatmap(df)
    peak_hours, hourly_by_year = get_peak_hours_by_year(df)
    yearly_volume = get_yearly_volume(df)
    streaks = get_longest_streaks(df)

    # Step 5: Content analysis
    print("\n[5/8] Analyzing content (this may take a while)...")
    emojis_by_year = get_top_emojis_by_year(df)
    emojis_by_contact = get_emoji_by_contact(df)
    question_ratio = get_question_ratio_by_year(df)
    question_by_contact = get_question_ratio_by_contact(df)

    print("  - Computing sentiment...")
    sentiment_by_contact = get_sentiment_by_contact(df, min_messages=MIN_MESSAGES_FOR_SENTIMENT)

    print("  - Extracting phrases...")
    phrases = get_top_phrases_by_year(df)
    unique_words = get_unique_words_by_year(df)

    print("  - Topic modeling...")
    topics_by_year = get_topics_by_year(df)
    topics_by_contact = get_topics_by_contact(df, contacts=top_names[:10])

    # Step 6: Save data for follow-up queries
    print("\n[6/8] Saving data for follow-up queries...")
    df_with_sentiment = add_sentiment_to_df(df)
    df_with_sentiment.to_parquet(DATA_DIR / "messages.parquet")
    sentiment_by_contact.to_parquet(DATA_DIR / "sentiment_scores.parquet")
    topics_by_year.to_parquet(DATA_DIR / "topics.parquet")

    yearly_stats = {
        'yearly_volume': yearly_volume.to_dict('records'),
        'top_by_year': top_by_year.to_dict('records'),
        'peak_hours': peak_hours.to_dict('records'),
    }
    with open(DATA_DIR / "yearly_stats.json", 'w') as f:
        json.dump(yearly_stats, f, indent=2, default=str)

    # 2025 Deep Dive Analysis
    print("\n[6.5/8] Generating 2025 deep dive...")
    df_2025 = df[df['year'] == 2025].copy()

    # Top 10 people in 2025
    top_2025 = df_2025.groupby('contact_name').agg(
        total_messages=('message_id', 'count'),
        sent=('is_from_me', 'sum'),
    ).reset_index()
    top_2025['received'] = top_2025['total_messages'] - top_2025['sent']
    top_2025 = top_2025.sort_values('total_messages', ascending=False).head(10)

    # Monthly breakdown for 2025
    df_2025['month'] = df_2025['datetime'].dt.month
    df_2025['month_name'] = df_2025['datetime'].dt.strftime('%b')

    monthly_top = df_2025.groupby(['month', 'month_name', 'contact_name']).size().reset_index(name='count')
    monthly_top['rank'] = monthly_top.groupby('month')['count'].rank(ascending=False, method='first')
    monthly_top_1 = monthly_top[monthly_top['rank'] == 1].sort_values('month')

    # Calculate fastest response times for 2025 (who you respond to fastest)
    response_times_2025 = calculate_response_times(df_2025)
    fastest_responses_2025 = []
    if not response_times_2025.empty:
        valid = response_times_2025.dropna(subset=['your_response_time_min'])
        if not valid.empty:
            fastest = valid.sort_values('your_response_time_min').head(3)
            fastest_responses_2025 = [
                {'contact_name': row['contact_name'], 'response_time_min': row['your_response_time_min']}
                for _, row in fastest.iterrows()
            ]

    # Step 7: Generate visualizations
    print("\n[7/8] Generating visualizations...")
    
    # Get top contact names for filtering
    top_contact_names = top_contacts['contact_name'].head(20).tolist()
    
    # Filter data to only top contacts
    initiators_filtered = initiators[initiators['contact_name'].isin(top_contact_names)]
    response_times_filtered = response_times[response_times['contact_name'].isin(top_contact_names)]
    emojis_by_contact_filtered = emojis_by_contact[emojis_by_contact['contact_name'].isin(top_contact_names)]
    question_by_contact_filtered = question_by_contact[question_by_contact['contact_name'].isin(top_contact_names)]
    
    # Load sentiment scores for top contacts
    sentiment_scores_path = DATA_DIR / "sentiment_scores.parquet"
    sentiment_by_contact_filtered = None
    if sentiment_scores_path.exists():
        try:
            sentiment_all = pd.read_parquet(sentiment_scores_path)
            sentiment_by_contact_filtered = sentiment_all[sentiment_all['contact_name'].isin(top_contact_names)]
        except Exception as e:
            print(f"  Warning: Could not load sentiment scores: {e}")
            sentiment_by_contact_filtered = sentiment_by_contact[sentiment_by_contact['contact_name'].isin(top_contact_names)]
    else:
        sentiment_by_contact_filtered = sentiment_by_contact[sentiment_by_contact['contact_name'].isin(top_contact_names)]
    
    charts = {
        'stacked_area': create_stacked_area(monthly_volume),
        'lopsidedness': create_lopsidedness_scatter(lopsidedness.head(30)),
        'heatmap': create_hour_day_heatmap(heatmap_data),
        'yearly_volume': create_yearly_volume_bar(yearly_volume),
        'peak_hours': create_peak_hours_small_multiples(hourly_by_year),
        'sentiment_best': create_sentiment_bar(sentiment_by_contact, title="Best Vibes", top_n=15),
        'sentiment_worst': create_sentiment_bar(sentiment_by_contact, title="Worst Vibes", top_n=15, worst=True),
        'emoji_grid': create_emoji_grid(emojis_by_year),
        'question_ratio': create_question_ratio_line(question_ratio),
        'fastest_responses_2025': fastest_responses_2025,
        'initiators': create_initiators_bar(initiators_filtered, top_n=15),
        'response_times': create_response_times_scatter(response_times_filtered, top_n=20),
        'streaks': create_streaks_bar(streaks, top_n=10),
        'emoji_by_contact': create_emoji_by_contact_grid(emojis_by_contact_filtered, top_contacts=10),
        'question_by_contact': create_question_by_contact_bar(question_by_contact_filtered, top_n=15),
        'unique_words': create_unique_words_timeline(unique_words, words_per_year=5),
        'topics_by_contact': create_topics_by_contact_display(topics_by_contact),
        'sentiment_top_contacts': create_sentiment_bar(sentiment_by_contact_filtered, title="Sentiment: Your Top People", top_n=20),
    }

    # Build AI-generated insights about surprising relationship dynamics (2023-2025)
    print("  - Generating relationship insights...")
    insights = {'ai_insights': []}

    df_recent = df[df['year'].isin([2023, 2024, 2025])].copy()

    # 1. Find dramatic relationship changes - people who exploded in 2025
    yearly_counts = df_recent.groupby(['year', 'contact_name']).size().unstack(fill_value=0)
    if 2024 in yearly_counts.columns and 2025 in yearly_counts.columns:
        yearly_counts['change_2024_2025'] = yearly_counts[2025] - yearly_counts[2024]
        yearly_counts['ratio_2024_2025'] = (yearly_counts[2025] + 1) / (yearly_counts[2024] + 1)

        # Biggest explosions (absolute) - people with >1000 increase
        explosions = yearly_counts[yearly_counts['change_2024_2025'] > 1000].sort_values('change_2024_2025', ascending=False)
        if not explosions.empty:
            for name in explosions.head(3).index:
                old = int(yearly_counts.loc[name, 2024])
                new = int(yearly_counts.loc[name, 2025])
                if old < 100:  # Truly new relationship
                    insights['ai_insights'].append((
                        f"New Major Relationship: {name}",
                        f"From {old} messages in 2024 to {new:,} in 2025. This person went from barely on your radar to one of your most frequent contacts."
                    ))
                else:
                    insights['ai_insights'].append((
                        f"Relationship Intensified: {name}",
                        f"Jumped from {old:,} to {new:,} messages (+{new-old:,}). Something changed significantly in this relationship."
                    ))

    # 2. Find the "comeback" pattern - dropped then came back
    if 2023 in yearly_counts.columns and 2024 in yearly_counts.columns and 2025 in yearly_counts.columns:
        comebacks = yearly_counts[
            (yearly_counts[2023] > 1000) &
            (yearly_counts[2024] < yearly_counts[2023] * 0.3) &
            (yearly_counts[2025] > yearly_counts[2023] * 0.8)
        ]
        for name in comebacks.index:
            y23 = int(yearly_counts.loc[name, 2023])
            y24 = int(yearly_counts.loc[name, 2024])
            y25 = int(yearly_counts.loc[name, 2025])
            insights['ai_insights'].append((
                f"The Comeback: {name}",
                f"A friendship that nearly went silent. {y23:,} msgs in 2023, dropped to just {y24} in 2024, then roared back to {y25:,} in 2025."
            ))

    # 3. Find complete fadeouts - top 10 in 2023, nearly zero now
    if 2023 in yearly_counts.columns and 2025 in yearly_counts.columns:
        top_2023 = yearly_counts[2023].sort_values(ascending=False).head(15).index
        fadeouts = yearly_counts.loc[top_2023]
        fadeouts = fadeouts[(fadeouts[2023] > 500) & (fadeouts[2025] < 50)]
        for name in fadeouts.index[:2]:
            y23 = int(yearly_counts.loc[name, 2023])
            y25 = int(yearly_counts.loc[name, 2025])
            insights['ai_insights'].append((
                f"Faded Connection: {name}",
                f"Once a top contact with {y23:,} messages in 2023, now down to just {y25} in 2025. This relationship has gone quiet."
            ))

    # 4. Late night confidant pattern
    df_recent['hour'] = df_recent['datetime'].dt.hour
    late_night = df_recent[(df_recent['hour'] >= 0) & (df_recent['hour'] < 4)]
    late_counts = late_night.groupby('contact_name').size()
    total_counts = df_recent.groupby('contact_name').size()
    late_pct = (late_counts / total_counts).dropna()

    # Find people with high late-night % AND significant volume
    late_heavy = late_pct[(late_pct > 0.15) & (total_counts > 200)]
    if not late_heavy.empty:
        name = late_heavy.sort_values(ascending=False).index[0]
        pct = late_heavy[name] * 100
        count = int(late_counts.get(name, 0))
        insights['ai_insights'].append((
            f"Late Night Confidant: {name}",
            f"{pct:.0f}% of your messages ({count} total) are between midnight and 4am. This is your go-to person for those sleepless nights."
        ))

    # 5. Work hours vs personal relationship patterns
    df_recent['is_weekend'] = df_recent['datetime'].dt.dayofweek >= 5
    df_recent['is_work_hours'] = (
        (df_recent['hour'] >= 9) &
        (df_recent['hour'] < 18) &
        (~df_recent['is_weekend'])
    )
    work_msgs = df_recent[df_recent['is_work_hours']].groupby('contact_name').size()
    total = df_recent.groupby('contact_name').size()
    work_ratio = (work_msgs / total).dropna()
    
    # Weekend-heavy relationships
    weekend_msgs = df_recent[df_recent['is_weekend']].groupby('contact_name').size()
    weekend_ratio = (weekend_msgs / total).dropna()
    weekend_heavy = weekend_ratio[weekend_ratio > 0.5]
    if not weekend_heavy.empty and 2025 in yearly_counts.columns:
        weekend_contacts = weekend_heavy[weekend_heavy.index.isin(yearly_counts[yearly_counts[2025] > 300].index)]
        if not weekend_contacts.empty:
            top_weekend = weekend_contacts.sort_values(ascending=False).head(1)
            name = top_weekend.index[0]
            pct = weekend_ratio[name] * 100
            msgs_2025 = int(yearly_counts.loc[name, 2025])
            insights['ai_insights'].append((
                f"Weekend Friend: {name}",
                f"{pct:.0f}% of your {msgs_2025:,} messages are on weekends. This is your personal time relationship, separate from work life."
            ))

    # Find people who are almost exclusively work hours AND became big in 2025
    if 2025 in yearly_counts.columns:
        work_heavy = work_ratio[work_ratio > 0.75]
        new_work_contacts = work_heavy[work_heavy.index.isin(yearly_counts[yearly_counts[2025] > 500].index)]
        if not new_work_contacts.empty:
            top_work = new_work_contacts.sort_values(ascending=False)
            for name in top_work.head(2).index:
                pct = work_ratio[name] * 100
                msgs_2025 = int(yearly_counts.loc[name, 2025])
                insights['ai_insights'].append((
                    f"Professional Relationship: {name}",
                    f"{pct:.0f}% of your {msgs_2025:,} messages are during work hours (9-6 weekdays). This relationship lives in your professional life."
                ))

    # 6. Response time asymmetries
    df_sorted = df_recent.sort_values(['contact_name', 'datetime']).copy()
    df_sorted['prev_time'] = df_sorted.groupby('contact_name')['datetime'].shift(1)
    df_sorted['prev_from_me'] = df_sorted.groupby('contact_name')['is_from_me'].shift(1)
    df_sorted['is_response'] = df_sorted['is_from_me'] != df_sorted['prev_from_me']
    df_sorted['response_min'] = (df_sorted['datetime'] - df_sorted['prev_time']).dt.total_seconds() / 60

    responses = df_sorted[df_sorted['is_response'] & (df_sorted['response_min'] > 0) & (df_sorted['response_min'] < 1440)]
    your_resp = responses[responses['is_from_me'] == 1].groupby('contact_name')['response_min'].median()
    their_resp = responses[responses['is_from_me'] == 0].groupby('contact_name')['response_min'].median()

    combined = pd.DataFrame({'you': your_resp, 'them': their_resp}).dropna()
    combined = combined[combined.index.isin(total[total > 300].index)]
    combined['ratio'] = combined['them'] / combined['you']

    # You respond faster
    eager = combined[combined['ratio'] > 3].sort_values('ratio', ascending=False)
    if not eager.empty:
        name = eager.index[0]
        your_time = combined.loc[name, 'you']
        their_time = combined.loc[name, 'them']
        insights['ai_insights'].append((
            f"You're Eager: {name}",
            f"You respond in {your_time:.0f} min on average, but they take {their_time:.0f} min. You're {their_time/your_time:.1f}x faster to respond."
        ))

    # They respond faster
    patient = combined[combined['ratio'] < 0.3].sort_values('ratio')
    if not patient.empty:
        name = patient.index[0]
        your_time = combined.loc[name, 'you']
        their_time = combined.loc[name, 'them']
        insights['ai_insights'].append((
            f"They're Waiting: {name}",
            f"They respond in {their_time:.0f} min, but you take {your_time:.0f} min. They're {your_time/their_time:.1f}x faster than you."
        ))
    
    # Balanced response times
    balanced_responses = combined[(combined['ratio'] >= 0.5) & (combined['ratio'] <= 2.0)]
    if not balanced_responses.empty and len(balanced_responses) >= 2:
        fastest_balanced = balanced_responses.nsmallest(1, 'you')
        if not fastest_balanced.empty:
            name = fastest_balanced.index[0]
            your_time = fastest_balanced.loc[name, 'you']
            their_time = fastest_balanced.loc[name, 'them']
            insights['ai_insights'].append((
                f"Perfectly Synced: {name}",
                f"You both respond in similar timeframes (you: {your_time:.0f} min, them: {their_time:.0f} min). This relationship has matching energy levels."
            ))

    # 7. Your texting volume is exploding
    yearly_totals = df.groupby('year').size()
    if 2023 in yearly_totals.index and 2025 in yearly_totals.index:
        y23_total = yearly_totals[2023]
        y25_total = yearly_totals[2025]
        if y25_total > y23_total * 1.3:
            insights['ai_insights'].append((
                "Your Texting Has Exploded",
                f"You sent {y25_total:,} messages in 2025 vs {y23_total:,} in 2023. That's a {(y25_total/y23_total - 1)*100:.0f}% increase in how much you text."
            ))

    # 8. Burst vs Consistent relationships (all-time analysis 2017-2025)
    print("  - Analyzing burst vs consistent relationships...")
    all_years = list(range(2017, 2026))
    yearly_by_contact = df.groupby(['contact_name', 'year']).size().unstack(fill_value=0)

    # For each contact, calculate consistency metrics
    consistency_data = []
    for contact in yearly_by_contact.index:
        yearly_msgs = yearly_by_contact.loc[contact]
        years_active = (yearly_msgs > 50).sum()  # Years with meaningful contact
        total_msgs = yearly_msgs.sum()
        max_year_msgs = yearly_msgs.max()
        max_year = yearly_msgs.idxmax()

        if total_msgs >= 500:  # Only analyze people with significant history
            # Coefficient of variation - lower = more consistent
            mean_msgs = yearly_msgs[yearly_msgs > 0].mean()
            std_msgs = yearly_msgs[yearly_msgs > 0].std()
            cv = std_msgs / mean_msgs if mean_msgs > 0 else 0

            # Concentration - what % of all messages were in the peak year
            concentration = max_year_msgs / total_msgs

            consistency_data.append({
                'contact': contact,
                'total_msgs': total_msgs,
                'years_active': years_active,
                'max_year': max_year,
                'max_year_msgs': max_year_msgs,
                'concentration': concentration,
                'cv': cv
            })

    if consistency_data:
        consistency_df = pd.DataFrame(consistency_data)

        # Most consistent long-term friendships: many years active, low concentration
        consistent = consistency_df[
            (consistency_df['years_active'] >= 5) &
            (consistency_df['concentration'] < 0.4)
        ].sort_values('total_msgs', ascending=False)

        if not consistent.empty:
            insights['ai_insights'].append((
                "Your Ride-or-Dies",
                "These friendships have stayed consistent across 5+ years without major peaks or valleys:"
            ))
            for _, row in consistent.head(4).iterrows():
                insights['ai_insights'].append((
                    f"Consistent: {row['contact']}",
                    f"{int(row['total_msgs']):,} total messages across {int(row['years_active'])} years. Peak year ({int(row['max_year'])}) was only {row['concentration']*100:.0f}% of total - steady throughout."
                ))

        # Intense burst relationships: high concentration in one year
        bursts = consistency_df[
            (consistency_df['concentration'] > 0.7) &
            (consistency_df['total_msgs'] >= 500)
        ].sort_values('concentration', ascending=False)

        if not bursts.empty:
            insights['ai_insights'].append((
                "Intense But Brief",
                "These relationships burned bright for a moment then faded:"
            ))
            for _, row in bursts.head(4).iterrows():
                insights['ai_insights'].append((
                    f"Burst: {row['contact']}",
                    f"{row['concentration']*100:.0f}% of your {int(row['total_msgs']):,} messages were in {int(row['max_year'])} ({int(row['max_year_msgs']):,} that year). A concentrated chapter of your life."
                ))

    # 9. Content-based psychoanalysis
    print("  - Analyzing message content patterns...")

    def count_pattern(text, pattern):
        if not text:
            return 0
        return len(re.findall(pattern, str(text).lower()))

    sent_msgs = df[df['is_from_me'] == 1].copy()
    contact_msg_counts = df.groupby('contact_name').size()
    valid_contacts = contact_msg_counts[contact_msg_counts >= 100].index
    sent_msgs = sent_msgs[sent_msgs['contact_name'].isin(valid_contacts)]

    # Question rate per contact (ASCII ? and fullwidth ？ U+FF1F)
    sent_msgs['has_question'] = sent_msgs['text'].fillna('').astype(str).str.contains(r'\?|？', regex=True)
    question_rate = sent_msgs.groupby('contact_name')['has_question'].mean()

    # Message length
    sent_msgs['msg_length'] = sent_msgs['text'].fillna('').str.len()
    avg_length = sent_msgs.groupby('contact_name')['msg_length'].mean()

    # I vs you ratio (self vs other focused)
    sent_msgs['i_count'] = sent_msgs['text'].apply(lambda x: count_pattern(x, r'\bi\b|\bi\'|\bmy\b|\bme\b'))
    sent_msgs['you_count'] = sent_msgs['text'].apply(lambda x: count_pattern(x, r'\byou\b|\byour\b'))
    i_rate = sent_msgs.groupby('contact_name')['i_count'].mean()
    you_rate = sent_msgs.groupby('contact_name')['you_count'].mean()
    i_you_ratio = (i_rate / (you_rate + 0.1))

    # Vulnerable words
    vulnerable_pattern = r'\bfeel\b|\bfeeling\b|\bworried\b|\bscared\b|\banxious\b|\bstressed\b|\bsad\b|\bupset\b|\bhurt\b|\bafraid\b|\blonely\b'
    sent_msgs['vulnerable'] = sent_msgs['text'].apply(lambda x: count_pattern(x, vulnerable_pattern))
    vulnerable_rate = sent_msgs.groupby('contact_name')['vulnerable'].mean()

    # Advice seeking
    advice_pattern = r'\bshould i\b|\bwhat do you think\b|\badvice\b|\bwhat should\b|\bdo you think\b'
    sent_msgs['advice'] = sent_msgs['text'].apply(lambda x: count_pattern(x, advice_pattern))
    advice_rate = sent_msgs.groupby('contact_name')['advice'].mean()

    # Intellectual words
    intellectual_pattern = r'\bthink\b|\binteresting\b|\bwonder\b|\bidea\b|\bargument\b|\breason\b|\btheory\b|\bconcept\b'
    sent_msgs['intellectual'] = sent_msgs['text'].apply(lambda x: count_pattern(x, intellectual_pattern))
    intellectual_rate = sent_msgs.groupby('contact_name')['intellectual'].mean()

    # "we" vs "I" - collaboration
    sent_msgs['we_count'] = sent_msgs['text'].apply(lambda x: count_pattern(x, r'\bwe\b|\bour\b|\bus\b'))
    we_rate = sent_msgs.groupby('contact_name')['we_count'].mean()
    we_i_ratio = (we_rate / (i_rate + 0.1))

    # Get top contacts for context
    top_500 = contact_msg_counts[contact_msg_counts >= 500].index

    # Insight: Who you go to for advice vs who you just talk at
    advice_seekers = advice_rate[advice_rate.index.isin(top_500)].nlargest(3)
    declarative = question_rate[question_rate.index.isin(top_500)].nsmallest(3)

    if not advice_seekers.empty:
        names = ', '.join(advice_seekers.index[:3].tolist())
        insights['ai_insights'].append((
            "Your Advisors",
            f"You seek advice and opinions most from: {names}. These are relationships where you're uncertain and looking for guidance."
        ))

    if not declarative.empty:
        names = list(declarative.index[:3])
        rates = [f"{question_rate[n]*100:.0f}%" for n in names]
        insights['ai_insights'].append((
            "No Questions Asked",
            f"With {names[0]} ({rates[0]} questions), {names[1]} ({rates[1]}), and {names[2]} ({rates[2]}), you rarely ask questions. You're declarative, certain, just sharing rather than seeking."
        ))

    # Insight: Self-focused vs other-focused conversations
    most_self = i_you_ratio[i_you_ratio.index.isin(top_500)].nlargest(3)
    most_other = i_you_ratio[i_you_ratio.index.isin(top_500)].nsmallest(3)

    if not most_self.empty and most_self.iloc[0] > 1.5:
        name = most_self.index[0]
        ratio = most_self.iloc[0]
        insights['ai_insights'].append((
            f"It's About You: {name}",
            f"You use 'I/my/me' {ratio:.1f}x more than 'you/your' in this relationship. This might be a space where you process your own life out loud."
        ))

    if not most_other.empty and most_other.iloc[0] < 0.7:
        name = most_other.index[0]
        ratio = 1/most_other.iloc[0]
        insights['ai_insights'].append((
            f"Focused on Them: {name}",
            f"You use 'you/your' {ratio:.1f}x more than 'I/my' here. This relationship is oriented around them - their life, their needs, their updates."
        ))

    # Insight: Emotional vulnerability patterns
    most_vulnerable = vulnerable_rate[vulnerable_rate.index.isin(valid_contacts)].nlargest(3)
    if not most_vulnerable.empty and most_vulnerable.iloc[0] > 0.02:
        names = ', '.join(most_vulnerable.index[:3].tolist())
        insights['ai_insights'].append((
            "Where You're Vulnerable",
            f"Your most emotionally open conversations (feeling words like worried, scared, stressed) are with: {names}. These people see a side of you others don't."
        ))

    # Insight: Intellectual sparring partners
    most_intellectual = intellectual_rate[intellectual_rate.index.isin(top_500)].nlargest(3)
    if not most_intellectual.empty:
        names = ', '.join(most_intellectual.index[:3].tolist())
        insights['ai_insights'].append((
            "Your Intellectual Sparring Partners",
            f"Highest density of 'think', 'interesting', 'idea', 'argument': {names}. These are the people you do your thinking with."
        ))

    # Insight: Collaborative "we" relationships
    most_collaborative = we_i_ratio[we_i_ratio.index.isin(top_500)].nlargest(3)
    if not most_collaborative.empty and most_collaborative.iloc[0] > 0.15:
        names = ', '.join(most_collaborative.index[:3].tolist())
        insights['ai_insights'].append((
            "Partnership Mode",
            f"You use 'we/us/our' most with: {names}. These feel like true partnerships - you're building something together, not just talking."
        ))

    # Insight: Short message rapid-fire relationships
    shortest_msgs = avg_length.nsmallest(10)
    shortest_with_volume = shortest_msgs[shortest_msgs.index.isin(top_500)]
    if not shortest_with_volume.empty:
        name = shortest_with_volume.index[0]
        length = shortest_with_volume.iloc[0]
        count = int(contact_msg_counts[name])
        insights['ai_insights'].append((
            f"Rapid Fire: {name}",
            f"Average message length: {length:.0f} characters across {count:,} messages. This is staccato conversation - quick exchanges, not essays."
        ))
    
    # Long message relationships
    longest_msgs = avg_length.nlargest(5)
    longest_with_volume = longest_msgs[longest_msgs.index.isin(top_500)]
    if not longest_with_volume.empty:
        name = longest_with_volume.index[0]
        length = longest_with_volume.iloc[0]
        count = int(contact_msg_counts[name])
        if length > 100:  # Only if significantly longer
            insights['ai_insights'].append((
                f"Deep Conversations: {name}",
                f"Average message length: {length:.0f} characters across {count:,} messages. You write longer, more thoughtful messages here - this is where you go deep."
            ))

    # Insight: Late night confidant
    late_night_sent = df_recent[(df_recent['hour'] >= 0) & (df_recent['hour'] < 4) & (df_recent['is_from_me'] == 1)]
    late_counts = late_night_sent.groupby('contact_name').size()
    total_sent_counts = df_recent[df_recent['is_from_me'] == 1].groupby('contact_name').size()
    late_pct = (late_counts / total_sent_counts).dropna()

    high_late = late_pct[(late_pct > 0.1) & (late_counts > 50)]
    if not high_late.empty:
        name = high_late.sort_values(ascending=False).index[0]
        pct = high_late[name] * 100
        count = int(late_counts[name])
        insights['ai_insights'].append((
            f"3am Thoughts: {name}",
            f"{pct:.0f}% of your messages to them are between midnight-4am ({count} messages). When you can't sleep, this is who you reach for."
        ))

    # 10. Initiator patterns
    print("  - Analyzing conversation initiation patterns...")
    if not initiators.empty:
        # You always initiate
        you_initiate_heavy = initiators[initiators['you_initiate_pct'] > 75].head(3)
        if not you_initiate_heavy.empty:
            names = ', '.join(you_initiate_heavy['contact_name'].tolist())
            insights['ai_insights'].append((
                "You Always Reach Out First",
                f"With {names}, you initiate 75%+ of conversations. These are relationships where you're the one keeping the connection alive."
            ))
        
        # They always initiate
        they_initiate_heavy = initiators[initiators['you_initiate_pct'] < 25].head(3)
        if not they_initiate_heavy.empty:
            names = ', '.join(they_initiate_heavy['contact_name'].tolist())
            insights['ai_insights'].append((
                "They Always Reach Out First",
                f"With {names}, they initiate 75%+ of conversations. These people are the ones keeping you in their lives."
            ))
        
        # Balanced initiators
        balanced = initiators[(initiators['you_initiate_pct'] >= 40) & (initiators['you_initiate_pct'] <= 60)].head(3)
        if not balanced.empty and len(balanced) >= 2:
            names = ', '.join(balanced['contact_name'].head(3).tolist())
            insights['ai_insights'].append((
                "Mutual Initiators",
                f"With {names}, initiation is balanced (40-60% each way). These feel like true two-way friendships where both people put in effort."
            ))

    # 11. Streak patterns
    print("  - Analyzing streak patterns...")
    if not streaks.empty:
        longest_streak = streaks.iloc[0]
        insights['ai_insights'].append((
            f"Your Longest Streak: {longest_streak['contact_name']}",
            f"{int(longest_streak['streak_length'])} consecutive days from {longest_streak['start_date']} to {longest_streak['end_date']}. You texted every single day - that's commitment."
        ))
        
        # Multiple long streaks with same person
        if len(streaks) > 1:
            top_contact = streaks['contact_name'].iloc[0]
            streaks_with_top = streaks[streaks['contact_name'] == top_contact]
            if len(streaks_with_top) > 1:
                total_streak_days = streaks_with_top['streak_length'].sum()
                insights['ai_insights'].append((
                    f"Streak Master: {top_contact}",
                    f"You have multiple long streaks totaling {int(total_streak_days)} days. This person is your most consistent daily contact."
                ))

    # 12. Emoji patterns
    print("  - Analyzing emoji usage patterns...")
    if not emojis_by_contact.empty:
        # Most emoji-heavy contacts
        emoji_counts = emojis_by_contact.groupby('contact_name').size()
        emoji_heavy = emoji_counts.nlargest(3)
        if not emoji_heavy.empty:
            names = ', '.join(emoji_heavy.index.tolist())
            insights['ai_insights'].append((
                "Your Emoji People",
                f"You use the most diverse emojis with: {names}. These conversations are colorful and expressive."
            ))
        
        # Emoji evolution by year
        if not emojis_by_year.empty:
            recent_emojis = emojis_by_year[emojis_by_year['year'] >= 2023]
            if not recent_emojis.empty:
                top_recent = recent_emojis.groupby('emojis')['count'].sum().nlargest(1)
                if not top_recent.empty:
                    top_emoji = top_recent.index[0]
                    count = int(top_recent.iloc[0])
                    insights['ai_insights'].append((
                        "Your Signature Emoji",
                        f"Since 2023, you've used '{top_emoji}' {count:,} times. This emoji has become your go-to expression."
                    ))

    # 13. Question patterns
    print("  - Analyzing question patterns...")
    if not question_by_contact.empty:
        # Most questions
        most_questions = question_by_contact.nlargest(3, 'question_pct')
        if not most_questions.empty:
            names = ', '.join([f"{name} ({row['question_pct']:.0f}%)" 
                              for name, row in most_questions.iterrows()])
            insights['ai_insights'].append((
                "Your Question People",
                f"You ask questions most with: {names}. These are relationships where you're curious, seeking information, or checking in."
            ))
        
        # Fewest questions
        fewest_questions = question_by_contact.nsmallest(3, 'question_pct')
        if not fewest_questions.empty and fewest_questions.iloc[0]['question_pct'] < 5:
            names = ', '.join([f"{name} ({row['question_pct']:.0f}%)" 
                              for name, row in fewest_questions.iterrows()])
            insights['ai_insights'].append((
                "No Questions Needed",
                f"With {names}, you rarely ask questions. You're declarative, sharing information rather than seeking it."
            ))

    # 14. Sentiment patterns
    print("  - Analyzing sentiment patterns...")
    if not sentiment_by_contact.empty:
        # Best vibes
        best_vibes = sentiment_by_contact.head(3)
        if not best_vibes.empty and best_vibes.iloc[0]['avg_sentiment'] > 0.1:
            names = ', '.join(best_vibes['contact_name'].tolist())
            avg_sent = best_vibes.iloc[0]['avg_sentiment']
            insights['ai_insights'].append((
                "Your Positive Energy People",
                f"Conversations with {names} have the highest positive sentiment (avg {avg_sent:.2f}). These people bring out your best vibes."
            ))
        
        # Worst vibes (but not too negative, just less positive)
        worst_vibes = sentiment_by_contact.tail(3)
        if not worst_vibes.empty and worst_vibes.iloc[-1]['avg_sentiment'] < 0:
            names = ', '.join(worst_vibes['contact_name'].tolist())
            avg_sent = worst_vibes.iloc[-1]['avg_sentiment']
            insights['ai_insights'].append((
                "More Serious Conversations",
                f"With {names}, your conversations have lower sentiment (avg {avg_sent:.2f}). These might be where you discuss problems, stress, or difficult topics."
            ))

    # 15. Unique words evolution
    print("  - Analyzing vocabulary evolution...")
    if not unique_words.empty:
        # Words that appeared in recent years
        recent_words = unique_words[unique_words['year'] >= 2023]
        if not recent_words.empty:
            top_recent_word = recent_words.nlargest(1, 'tfidf_score')
            if not top_recent_word.empty:
                word = top_recent_word.iloc[0]['word']
                year = int(top_recent_word.iloc[0]['year'])
                insights['ai_insights'].append((
                    "Your New Vocabulary",
                    f"The word '{word}' spiked in {year} - it became uniquely important to how you communicate that year."
                ))
        
        # Words that disappeared
        old_words = unique_words[unique_words['year'] <= 2019]
        recent_years = unique_words[unique_words['year'] >= 2023]
        if not old_words.empty and not recent_years.empty:
            old_word_list = set(old_words['word'].unique())
            recent_word_list = set(recent_years['word'].unique())
            disappeared = old_word_list - recent_word_list
            if len(disappeared) > 0:
                top_disappeared = old_words[old_words['word'].isin(list(disappeared))].nlargest(1, 'tfidf_score')
                if not top_disappeared.empty:
                    word = top_disappeared.iloc[0]['word']
                    insights['ai_insights'].append((
                        "Words You Left Behind",
                        f"'{word}' was important in your early years but disappeared from your vocabulary. Your language evolved."
                    ))

    # 16. Topics by contact patterns
    print("  - Analyzing topic patterns...")
    if not topics_by_contact.empty:
        # Most topic-diverse contacts
        topic_counts = topics_by_contact.groupby('contact_name').size()
        diverse_topics = topic_counts.nlargest(3)
        if not diverse_topics.empty:
            names = ', '.join(diverse_topics.index.tolist())
            insights['ai_insights'].append((
                "Your Multi-Topic People",
                f"With {names}, you discuss the widest range of topics. These are your most versatile conversation partners."
            ))
        
        # Unique topics per contact
        all_topics = topics_by_contact.groupby('top_words').size()
        unique_topics = all_topics[all_topics == 1]  # Topics only discussed with one person
        if not unique_topics.empty:
            unique_contacts = topics_by_contact[topics_by_contact['top_words'].isin(unique_topics.index)]
            if not unique_contacts.empty:
                contact_with_unique = unique_contacts['contact_name'].iloc[0]
                topic = unique_contacts['top_words'].iloc[0]
                insights['ai_insights'].append((
                    f"Unique Topics: {contact_with_unique}",
                    f"You discuss '{topic}' almost exclusively with {contact_with_unique}. This topic defines your relationship with them."
                ))

    # 17. Grammar patterns (formal vs casual)
    print("  - Analyzing grammar patterns...")
    # This uses the formal_contacts and casual_contacts from section 7.5
    # We'll add this after section 7.5 is calculated

    # 18. Agreement vs Debate patterns
    print("  - Analyzing agreement patterns...")
    # This uses agreers and debaters from section 7.5
    # We'll add this after section 7.5 is calculated

    # 19. Social churn insights
    print("  - Analyzing social churn patterns...")
    # This uses fadeouts and new_friends from section 7.5
    # We'll add this after section 7.5 is calculated

    # Step 7.5: Calculate data for missing sections 4-7
    print("\n[7.5/8] Calculating additional sections...")
    
    # Section 4: Word Cloud Comparison
    print("  - Word cloud comparison...")
    wordcloud_old = None
    wordcloud_new = None
    if START_YEAR in df['year'].values and (END_YEAR - 1) in df['year'].values:
        from collections import Counter
        from analysis.content import clean_text_for_phrases
        
        sent_old = df[(df['year'] == START_YEAR) & (df['is_from_me'] == 1)].copy()
        sent_new = df[(df['year'] == END_YEAR - 1) & (df['is_from_me'] == 1)].copy()
        
        if len(sent_old) > 0:
            old_text = ' '.join(sent_old['text'].fillna('').astype(str).tolist())
            old_clean = clean_text_for_phrases(old_text)
            old_words = [w for w in old_clean.split() if w not in BORING_WORDS and len(w) > 2]
            wordcloud_old = list(Counter(old_words).most_common(20))
        
        if len(sent_new) > 0:
            new_text = ' '.join(sent_new['text'].fillna('').astype(str).tolist())
            new_clean = clean_text_for_phrases(new_text)
            new_words = [w for w in new_clean.split() if w not in BORING_WORDS and len(w) > 2]
            wordcloud_new = list(Counter(new_words).most_common(20))
    
    # Section 5: Grammar (Formal vs Casual)
    print("  - Grammar analysis...")
    formal_contacts = None
    casual_contacts = None
    
    sent_msgs_grammar = df[df['is_from_me'] == 1].copy()
    contact_msg_counts_grammar = sent_msgs_grammar.groupby('contact_name').size()
    valid_contacts_grammar = contact_msg_counts_grammar[contact_msg_counts_grammar >= 50].index
    sent_msgs_grammar = sent_msgs_grammar[sent_msgs_grammar['contact_name'].isin(valid_contacts_grammar)]
    
    if len(sent_msgs_grammar) > 0:
        # Formal score: proper punctuation, capitalization, longer sentences
        sent_msgs_grammar['has_period'] = sent_msgs_grammar['text'].fillna('').str.contains(r'\.', regex=True)
        sent_msgs_grammar['has_capital'] = sent_msgs_grammar['text'].fillna('').str.contains(r'[A-Z]')
        sent_msgs_grammar['word_count'] = sent_msgs_grammar['text'].fillna('').str.split().str.len()
        sent_msgs_grammar['is_all_lower'] = sent_msgs_grammar['text'].fillna('').str.islower()
        
        formal_scores = sent_msgs_grammar.groupby('contact_name').agg(
            punctuation_rate=('has_period', 'mean'),
            capital_rate=('has_capital', 'mean'),
            avg_length=('word_count', 'mean'),
        )
        formal_scores['formal_score'] = (
            formal_scores['punctuation_rate'] * 0.4 +
            formal_scores['capital_rate'] * 0.3 +
            (formal_scores['avg_length'] / 20).clip(0, 1) * 0.3
        )
        formal_contacts = [(name, score) for name, score in 
                          formal_scores.nlargest(10, 'formal_score')['formal_score'].items()]
        
        # Casual: all lowercase percentage
        lowercase_pct = sent_msgs_grammar.groupby('contact_name')['is_all_lower'].mean()
        casual_contacts = list(lowercase_pct.nlargest(10).items())
    
    # Section 6: Agreement vs Debate
    print("  - Agreement vs debate analysis...")
    agreers = None
    debaters = None
    
    sent_msgs_debate = df[df['is_from_me'] == 1].copy()
    contact_msg_counts_debate = sent_msgs_debate.groupby('contact_name').size()
    valid_contacts_debate = contact_msg_counts_debate[contact_msg_counts_debate >= 50].index
    sent_msgs_debate = sent_msgs_debate[sent_msgs_debate['contact_name'].isin(valid_contacts_debate)]
    
    if len(sent_msgs_debate) > 0:
        # Agreement patterns
        agreement_patterns = [
            r'\btotally\b', r'\bexactly\b', r'\bso true\b', r'\b100%\b', r'\babsolutely\b',
            r'\bcompletely\b', r'\bdefinitely\b', r'\bfor sure\b', r'\byeah\b', r'\byep\b'
        ]
        sent_msgs_debate['has_agreement'] = sent_msgs_debate['text'].fillna('').str.lower().str.contains(
            '|'.join(agreement_patterns), regex=True, na=False
        )
        agreement_rates = sent_msgs_debate.groupby('contact_name')['has_agreement'].mean() * 100
        agreers = list(agreement_rates.nlargest(10).items())
        
        # Debate patterns
        debate_patterns = [
            r'\bactually\b', r'\bbut\b', r'\bi disagree\b', r'\bnot sure\b', r'\bhowever\b',
            r'\bthough\b', r'\balthough\b', r'\bdisagree\b', r'\bdont think\b', r'\bnot really\b'
        ]
        sent_msgs_debate['has_debate'] = sent_msgs_debate['text'].fillna('').str.lower().str.contains(
            '|'.join(debate_patterns), regex=True, na=False
        )
        debate_rates = sent_msgs_debate.groupby('contact_name')['has_debate'].mean() * 100
        debaters = list(debate_rates.nlargest(10).items())
    
    # Section 7: Social Churn (Fadeouts and New Friends)
    print("  - Social churn analysis...")
    fadeouts = None
    new_friends = None
    
    # Compare START_YEAR vs END_YEAR-1
    if START_YEAR in df['year'].values and (END_YEAR - 1) in df['year'].values:
        yearly_counts_churn = df.groupby(['year', 'contact_name']).size().unstack(fill_value=0)
        
        if START_YEAR in yearly_counts_churn.columns and (END_YEAR - 1) in yearly_counts_churn.columns:
            old_year = START_YEAR
            new_year = END_YEAR - 1
            
            # Fadeouts: were active in old year, much less in new year
            fadeout_data = []
            for contact in yearly_counts_churn.index:
                old_count = yearly_counts_churn.loc[contact, old_year]
                new_count = yearly_counts_churn.loc[contact, new_year]
                if old_count >= 100 and new_count < old_count * 0.3:  # Dropped by 70%+
                    fadeout_data.append((contact, int(old_count), int(new_count)))
            fadeouts = sorted(fadeout_data, key=lambda x: x[1] - x[2], reverse=True)[:10]
            
            # New friends: barely existed in old year, significant in new year
            new_friend_data = []
            for contact in yearly_counts_churn.index:
                old_count = yearly_counts_churn.loc[contact, old_year]
                new_count = yearly_counts_churn.loc[contact, new_year]
                if old_count < 50 and new_count >= 200:  # Barely existed, now significant
                    new_friend_data.append((contact, int(old_count), int(new_count)))
            new_friends = sorted(new_friend_data, key=lambda x: x[2], reverse=True)[:10]

    # Add insights for patterns calculated in section 7.5
    # 17. Grammar patterns
    if formal_contacts and len(formal_contacts) > 0:
        most_formal = formal_contacts[0]
        insights['ai_insights'].append((
            f"Most Formal: {most_formal[0]}",
            f"You use proper punctuation, capitalization, and longer sentences most with {most_formal[0]} (score: {most_formal[1]:.2f}). They get your best English."
        ))
    
    if casual_contacts and len(casual_contacts) > 0:
        most_casual = casual_contacts[0]
        insights['ai_insights'].append((
            f"Most Casual: {most_casual[0]}",
            f"You're most relaxed with {most_casual[0]} - {most_casual[1]:.0f}% of messages are all lowercase. Zero pretense, pure comfort."
        ))

    # 18. Agreement vs Debate patterns
    if agreers and len(agreers) > 0:
        top_agreer = agreers[0]
        insights['ai_insights'].append((
            f"You Agree Most: {top_agreer[0]}",
            f"You use agreement phrases ('totally', 'exactly', 'so true') {top_agreer[1]:.1f}% of the time with {top_agreer[0]}. This person gets your enthusiastic agreement."
        ))
    
    if debaters and len(debaters) > 0:
        top_debater = debaters[0]
        insights['ai_insights'].append((
            f"You Debate Most: {top_debater[0]}",
            f"You use debate phrases ('actually', 'but', 'I disagree') {top_debater[1]:.1f}% of the time with {top_debater[0]}. This is your intellectual sparring partner."
        ))

    # 19. Social churn insights
    if fadeouts and len(fadeouts) > 0:
        biggest_fadeout = fadeouts[0]
        if biggest_fadeout[1] > 0:
            drop_pct = int((1 - biggest_fadeout[2] / biggest_fadeout[1]) * 100)
            insights['ai_insights'].append((
                f"Biggest Fadeout: {biggest_fadeout[0]}",
                f"Went from {biggest_fadeout[1]:,} messages in {START_YEAR} to {biggest_fadeout[2]:,} in {END_YEAR - 1} ({drop_pct}% drop). This relationship has significantly quieted."
            ))
    
    if new_friends and len(new_friends) > 0:
        biggest_new_friend = new_friends[0]
        growth = biggest_new_friend[2] - biggest_new_friend[1]
        insights['ai_insights'].append((
            f"New Major Friend: {biggest_new_friend[0]}",
            f"Exploded from {biggest_new_friend[1]:,} messages in {START_YEAR} to {biggest_new_friend[2]:,} in {END_YEAR - 1} (+{growth:,}). This person became central to your life."
        ))

    # Step 8: Generate report
    print("\n[8/8] Generating HTML report...")

    total_messages = len(df)
    total_sent = df['is_from_me'].sum()
    total_received = total_messages - total_sent
    total_contacts = df['contact_name'].nunique()

    html = generate_report(
        total_messages=total_messages,
        total_sent=int(total_sent),
        total_received=int(total_received),
        total_contacts=total_contacts,
        top_contacts=top_contacts,
        charts=charts,
        phrases_df=phrases,
        emojis_df=emojis_by_year,
        topics_df=topics_by_year,
        insights=insights,
        top_2025=top_2025,
        df_2025=df_2025,
        top_by_year=top_by_year,
        monthly_top_2025=monthly_top_1,
        wordcloud_old=wordcloud_old,
        wordcloud_new=wordcloud_new,
        formal_contacts=formal_contacts,
        casual_contacts=casual_contacts,
        agreers=agreers,
        debaters=debaters,
        fadeouts=fadeouts,
        new_friends=new_friends,
    )

    output_path = save_report(html)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Report saved to: {output_path}")
    print(f"Data saved to: {DATA_DIR}")
    print("\nOpen the report with:")
    print(f"  open {output_path}")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    main()
