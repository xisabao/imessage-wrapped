"""Group chat analysis functions."""
import pandas as pd
import numpy as np
from config import (
    TOP_GROUPS_COUNT,
    MIN_MESSAGES_FOR_TOP_GROUP,
    MIN_GROUP_MEMBERS,
    BURST_CONCENTRATION_THRESHOLD,
    CONSISTENT_MIN_YEARS,
)


def get_top_groups_alltime(df, n=TOP_GROUPS_COUNT):
    """Get top N group chats by total message count."""
    counts = df.groupby(['chat_id', 'group_name']).agg(
        total_messages=('message_id', 'count'),
        sent=('is_from_me', 'sum'),
        years_active=('year', 'nunique'),
        first_message=('datetime', 'min'),
        last_message=('datetime', 'max'),
        unique_senders=('contact_id', 'nunique'),
    ).reset_index()

    counts['received'] = counts['total_messages'] - counts['sent']
    counts = counts[counts['total_messages'] >= MIN_MESSAGES_FOR_TOP_GROUP]
    counts = counts.sort_values('total_messages', ascending=False)

    return counts.head(n)


def get_top_groups_by_year(df, n=10):
    """Get top N group chats for each year."""
    yearly = df.groupby(['year', 'chat_id', 'group_name']).agg(
        total_messages=('message_id', 'count'),
        sent=('is_from_me', 'sum'),
    ).reset_index()

    yearly['received'] = yearly['total_messages'] - yearly['sent']
    yearly['rank'] = yearly.groupby('year')['total_messages'].rank(
        ascending=False, method='min'
    )

    top_per_year = yearly[yearly['rank'] <= n].sort_values(['year', 'rank'])
    return top_per_year


def get_group_activity_per_member(df, top_n_groups=10):
    """Get per-member message counts within each group.

    Returns a DataFrame with columns: chat_id, group_name, contact_name,
    message_count, pct_of_group.
    """
    # Only analyze top groups by volume
    top_groups = df.groupby('chat_id').size().nlargest(top_n_groups).index

    df_top = df[df['chat_id'].isin(top_groups)].copy()

    # Map is_from_me to "You" for sender identification
    df_top['sender'] = df_top['contact_name'].copy()
    df_top.loc[df_top['is_from_me'] == 1, 'sender'] = 'You'

    per_member = df_top.groupby(['chat_id', 'group_name', 'sender']).agg(
        message_count=('message_id', 'count'),
    ).reset_index()

    # Calculate percentage of group total
    group_totals = per_member.groupby('chat_id')['message_count'].transform('sum')
    per_member['pct_of_group'] = (per_member['message_count'] / group_totals * 100).round(1)

    per_member = per_member.sort_values(['chat_id', 'message_count'], ascending=[True, False])

    return per_member


def get_member_overlap(members_df, min_group_messages=None, active_groups=None):
    """Find shared members across active group chats.

    Returns a list of dicts: {group_a, group_b, group_a_name, group_b_name,
    shared_members, jaccard, shared_names}.
    """
    # Filter to active groups if provided
    if active_groups is not None:
        members_df = members_df[members_df['chat_id'].isin(active_groups)]

    # Build sets of members per group
    group_members = {}
    group_names = {}
    for chat_id, grp in members_df.groupby('chat_id'):
        group_members[chat_id] = set(grp['handle_id'].dropna().tolist())
        # Use first non-null group_name
        names = grp['group_name'].dropna().unique()
        group_names[chat_id] = names[0] if len(names) > 0 else f"Group {chat_id}"

    # Compute pairwise overlap
    overlaps = []
    chat_ids = list(group_members.keys())
    for i in range(len(chat_ids)):
        for j in range(i + 1, len(chat_ids)):
            a, b = chat_ids[i], chat_ids[j]
            set_a, set_b = group_members[a], group_members[b]
            shared = set_a & set_b
            if len(shared) >= 2:  # At least 2 shared members
                union = set_a | set_b
                jaccard = len(shared) / len(union) if len(union) > 0 else 0
                overlaps.append({
                    'group_a': a,
                    'group_b': b,
                    'group_a_name': group_names[a],
                    'group_b_name': group_names[b],
                    'shared_count': len(shared),
                    'jaccard': round(jaccard, 3),
                    'shared_handles': list(shared),
                })

    overlaps.sort(key=lambda x: x['shared_count'], reverse=True)
    return overlaps


def get_conversation_concentration(df, top_n_groups=15):
    """Calculate message concentration per group.

    Returns a DataFrame with columns: chat_id, group_name, total_members,
    total_messages, top1_pct, top3_pct, top50pct_members_pct, gini.
    """
    top_groups = df.groupby('chat_id').size().nlargest(top_n_groups).index
    df_top = df[df['chat_id'].isin(top_groups)].copy()

    df_top['sender'] = df_top['contact_name'].copy()
    df_top.loc[df_top['is_from_me'] == 1, 'sender'] = 'You'

    results = []
    for chat_id, grp in df_top.groupby('chat_id'):
        group_name = grp['group_name'].iloc[0]
        member_counts = grp.groupby('sender').size().sort_values(ascending=False)
        total = member_counts.sum()
        n_members = len(member_counts)

        if n_members < 2:
            continue

        # Top 1 person's share
        top1_pct = (member_counts.iloc[0] / total * 100) if total > 0 else 0

        # Top 3 people's share
        top3_pct = (member_counts.head(3).sum() / total * 100) if total > 0 else 0

        # What % of members produce 50% of messages
        cumsum = member_counts.cumsum()
        halfway = total / 2
        n_for_half = (cumsum <= halfway).sum() + 1
        top50pct_members_pct = (n_for_half / n_members * 100)

        # Gini coefficient
        values = member_counts.values.astype(float)
        if len(values) > 1 and values.sum() > 0:
            sorted_vals = np.sort(values)
            n = len(sorted_vals)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals))
        else:
            gini = 0

        results.append({
            'chat_id': chat_id,
            'group_name': group_name,
            'total_members': n_members,
            'total_messages': total,
            'top1_pct': round(top1_pct, 1),
            'top3_pct': round(top3_pct, 1),
            'top50pct_members_pct': round(top50pct_members_pct, 1),
            'gini': round(gini, 3),
        })

    return pd.DataFrame(results).sort_values('total_messages', ascending=False)


def classify_group_lifecycle(df):
    """Classify group chats as consistent, burst, rising, or fading.

    Returns a DataFrame with columns: chat_id, group_name, classification,
    total_messages, years_active, peak_year, peak_year_messages, concentration.
    """
    yearly = df.groupby(['chat_id', 'group_name', 'year']).size().reset_index(name='count')

    results = []
    for (chat_id, group_name), grp in yearly.groupby(['chat_id', 'group_name']):
        total = grp['count'].sum()
        if total < MIN_MESSAGES_FOR_TOP_GROUP:
            continue

        years_active = len(grp[grp['count'] > 10])  # Years with meaningful activity
        peak_row = grp.loc[grp['count'].idxmax()]
        peak_year = int(peak_row['year'])
        peak_count = int(peak_row['count'])
        concentration = peak_count / total if total > 0 else 0

        # Classification logic
        if concentration >= BURST_CONCENTRATION_THRESHOLD:
            classification = 'burst'
        elif years_active >= CONSISTENT_MIN_YEARS and concentration < 0.4:
            classification = 'consistent'
        else:
            # Check trend: compare first half vs second half
            years = sorted(grp['year'].unique())
            if len(years) >= 2:
                mid = len(years) // 2
                first_half = grp[grp['year'].isin(years[:mid])]['count'].sum()
                second_half = grp[grp['year'].isin(years[mid:])]['count'].sum()
                if second_half > first_half * 2:
                    classification = 'rising'
                elif first_half > second_half * 2:
                    classification = 'fading'
                else:
                    classification = 'steady'
            else:
                classification = 'steady'

        results.append({
            'chat_id': chat_id,
            'group_name': group_name,
            'classification': classification,
            'total_messages': total,
            'years_active': years_active,
            'peak_year': peak_year,
            'peak_year_messages': peak_count,
            'concentration': round(concentration, 3),
        })

    return pd.DataFrame(results).sort_values('total_messages', ascending=False)


def get_group_volume_over_time(df, groups=None, top_n=10):
    """Get monthly message volume for top group chats."""
    if groups is None:
        top = df.groupby('chat_id').size().nlargest(top_n).index
        groups = top

    df_filtered = df[df['chat_id'].isin(groups)].copy()
    df_filtered['year_month'] = df_filtered['datetime'].dt.to_period('M')

    monthly = df_filtered.groupby(['year_month', 'chat_id', 'group_name']).size().reset_index(name='count')
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()

    return monthly
