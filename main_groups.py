"""Main orchestration script for iMessage Wrapped - Group Chats."""
import pandas as pd
import re
from pathlib import Path

from config import (
    DATA_DIR, START_YEAR, END_YEAR,
    EXCLUDED_CONTACTS, MIN_TWO_WAY_RATIO,
    MIN_GROUP_MEMBERS, TOP_GROUPS_COUNT,
)
from extract import extract_group_messages, extract_group_members
from contacts import (
    get_contacts_from_macos,
    create_contact_mappings,
    load_contact_mappings,
    resolve_contact_id,
)
from analysis.groups import (
    get_top_groups_alltime,
    get_top_groups_by_year,
    get_group_activity_per_member,
    get_member_overlap,
    get_conversation_concentration,
    classify_group_lifecycle,
    get_group_volume_over_time,
)
from visualize_groups import (
    create_group_stacked_area,
    create_member_activity_subplots,
    create_overlap_network,
    create_concentration_chart,
    create_lifecycle_chart,
)
from report_groups import generate_group_report, save_group_report


def generate_group_name(chat_id, members_df, contact_mappings):
    """Generate a display name for an unnamed group from its members."""
    members = members_df[members_df['chat_id'] == chat_id]['handle_id'].tolist()
    names = []
    for handle in members:
        name = contact_mappings.get(str(handle), 'Unknown')
        # Skip unresolved (phone numbers)
        if name != str(handle) and name != 'Unknown':
            names.append(name.split()[0])  # First name only
        elif len(names) < 3:
            # Use a short version of the handle as fallback
            names.append(str(handle)[-4:])

    if len(names) == 0:
        return f"Group {chat_id}"
    elif len(names) <= 3:
        return ', '.join(names)
    else:
        return f"{', '.join(names[:3])} & {len(names) - 3} more"


def main():
    print("=" * 60)
    print("iMessage Wrapped - Group Chats")
    print("=" * 60)

    # Step 1: Extract group messages
    print("\n[1/7] Extracting group messages...")
    df = extract_group_messages()

    print(f"  Total group chats found: {df['chat_id'].nunique()}")

    if df.empty:
        print("No group messages found. Exiting.")
        return None

    # Step 2: Extract group membership
    print("\n[2/7] Extracting group membership...")
    members_df = extract_group_members()

    # Step 3: Resolve contacts
    print("\n[3/7] Resolving contacts...")
    # Try to load existing mappings first, then supplement from macOS Contacts
    contacts_map = load_contact_mappings()
    if not contacts_map:
        macos_contacts = get_contacts_from_macos()
        contacts_map = macos_contacts
    else:
        # Merge with macOS contacts for any new handle_ids
        macos_contacts = get_contacts_from_macos()
        for k, v in macos_contacts.items():
            if k not in contacts_map:
                contacts_map[k] = v

    mappings = create_contact_mappings(df, contacts_map)
    # Resolve sender names for group messages
    # Use contact_id as fallback so unresolved handles stay unique (not NaN)
    df['contact_name'] = df['contact_id'].astype(str).map(mappings).fillna(df['contact_id'].astype(str))
    # is_from_me messages have no handle_id, mark as "You"
    df.loc[df['is_from_me'] == 1, 'contact_name'] = 'You'

    # Resolve member names in membership table
    members_df['contact_name'] = members_df['handle_id'].astype(str).map(mappings).fillna(members_df['handle_id'].astype(str))

    # Step 4: Generate names for unnamed groups
    print("\n[4/7] Generating group names...")
    unnamed_chats = df[df['group_name'].isna() | (df['group_name'] == '')]['chat_id'].unique()
    generated_names = {}
    for chat_id in unnamed_chats:
        generated_names[chat_id] = generate_group_name(chat_id, members_df, contacts_map)

    # Apply generated names
    def resolve_group_name(row):
        if pd.notna(row['group_name']) and str(row['group_name']).strip():
            return str(row['group_name'])
        return generated_names.get(row['chat_id'], f"Group {row['chat_id']}")

    df['group_name'] = df.apply(resolve_group_name, axis=1)
    members_df['group_name'] = members_df.apply(
        lambda row: str(row['group_name']) if pd.notna(row['group_name']) and str(row['group_name']).strip()
        else generated_names.get(row['chat_id'], f"Group {row['chat_id']}"),
        axis=1,
    )

    n_unnamed = len(unnamed_chats)
    n_named = df['chat_id'].nunique() - n_unnamed
    print(f"  {n_named} groups with names, {n_unnamed} auto-named from members")

    # Filter out groups with too few participants
    # Use actual message senders rather than chat_handle_join (which can be incomplete
    # if members left the group or for older chats)
    sender_counts = df.groupby('chat_id')['contact_name'].nunique()
    valid_groups = sender_counts[sender_counts >= MIN_GROUP_MEMBERS].index
    before = len(df)
    df = df[df['chat_id'].isin(valid_groups)]
    print(f"  Filtered to groups with {MIN_GROUP_MEMBERS}+ participants: {before - len(df):,} messages removed")
    print(f"  Remaining: {len(df):,} messages in {df['chat_id'].nunique()} groups")

    if df.empty:
        print("No qualifying group chats. Exiting.")
        return None

    # Step 5: Run group analysis
    print("\n[5/7] Analyzing group chats...")
    top_groups = get_top_groups_alltime(df)
    top_by_year = get_top_groups_by_year(df)
    per_member = get_group_activity_per_member(df)

    print("  - Computing member overlap...")
    active_groups = df.groupby('chat_id').size()
    active_groups = active_groups[active_groups >= 50].index.tolist()
    overlaps = get_member_overlap(members_df, active_groups=active_groups)

    print("  - Computing conversation concentration...")
    concentration = get_conversation_concentration(df)

    print("  - Classifying group lifecycles...")
    lifecycle = classify_group_lifecycle(df)

    print("  - Computing volume over time...")
    top_group_ids = top_groups['chat_id'].head(10).tolist()
    monthly_volume = get_group_volume_over_time(df, groups=top_group_ids)

    # Generate insights
    print("  - Generating group insights...")
    insights = _generate_group_insights(df, top_groups, lifecycle, concentration, overlaps, members_df)

    # Step 6: Generate visualizations
    print("\n[6/7] Generating visualizations...")
    top_group_names = top_groups['group_name'].head(6).tolist()

    charts = {
        'group_stacked_area': create_group_stacked_area(monthly_volume),
        'member_activity': create_member_activity_subplots(per_member, top_group_names),
        'overlap_network': create_overlap_network(overlaps),
        'concentration': create_concentration_chart(concentration),
        'lifecycle': create_lifecycle_chart(lifecycle),
    }

    # Step 7: Generate report
    print("\n[7/7] Generating HTML report...")
    total_messages = len(df)
    total_groups = df['chat_id'].nunique()
    # Count unique participants (unique handle_ids + "You")
    total_participants = df['contact_name'].nunique()

    html = generate_group_report(
        total_messages=total_messages,
        total_groups=total_groups,
        total_participants=total_participants,
        top_groups=top_groups,
        charts=charts,
        top_by_year=top_by_year,
        lifecycle_df=lifecycle,
        concentration_df=concentration,
        insights=insights,
    )

    output_path = save_group_report(html)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Group report saved to: {output_path}")
    print(f"\nOpen the report with:")
    print(f"  open {output_path}")
    print("=" * 60)

    return output_path


def _generate_group_insights(df, top_groups, lifecycle, concentration, overlaps, members_df):
    """Generate AI-style insight cards for group chat patterns."""
    insights = []

    # 1. Biggest group chat
    if not top_groups.empty:
        top = top_groups.iloc[0]
        insights.append((
            f"Your #1 Group: {top['group_name']}",
            f"{top['total_messages']:,} messages across {top['years_active']} years with {top['unique_senders']} participants. This is your most active group chat."
        ))

    # 2. Burst groups (events/trips)
    if not lifecycle.empty:
        bursts = lifecycle[lifecycle['classification'] == 'burst']
        if not bursts.empty:
            for _, row in bursts.head(3).iterrows():
                insights.append((
                    f"Event Group: {row['group_name']}",
                    f"{row['concentration']*100:.0f}% of its {row['total_messages']:,} messages were in {row['peak_year']}. Looks like this group was for a specific event or trip."
                ))

    # 3. Consistent groups
    if not lifecycle.empty:
        consistent = lifecycle[lifecycle['classification'] == 'consistent']
        if not consistent.empty:
            for _, row in consistent.head(3).iterrows():
                insights.append((
                    f"The Regulars: {row['group_name']}",
                    f"Active for {row['years_active']} years with {row['total_messages']:,} messages spread evenly. This group has staying power."
                ))

    # 4. Most dominated group
    if not concentration.empty:
        most_dominated = concentration.nlargest(1, 'gini')
        if not most_dominated.empty:
            g = most_dominated.iloc[0]
            insights.append((
                f"One Person's Show: {g['group_name']}",
                f"The top person sends {g['top1_pct']:.0f}% of messages in this {g['total_members']}-person group. Top 3 account for {g['top3_pct']:.0f}%."
            ))

    # 5. Most democratic group
    if not concentration.empty:
        most_equal = concentration.nsmallest(1, 'gini')
        if not most_equal.empty:
            g = most_equal.iloc[0]
            if g['total_members'] >= 4:
                insights.append((
                    f"True Democracy: {g['group_name']}",
                    f"In this {g['total_members']}-person group, the top person only has {g['top1_pct']:.0f}% of messages. Everyone participates."
                ))

    # 6. Biggest overlap
    if overlaps:
        top_overlap = overlaps[0]
        insights.append((
            "Most Overlap",
            f'"{top_overlap["group_a_name"]}" and "{top_overlap["group_b_name"]}" share {top_overlap["shared_count"]} members (Jaccard: {top_overlap["jaccard"]:.2f}). These are likely the same social circle.'
        ))

    # 7. Fading groups
    if not lifecycle.empty:
        fading = lifecycle[lifecycle['classification'] == 'fading']
        if not fading.empty:
            row = fading.iloc[0]
            insights.append((
                f"Fading Away: {row['group_name']}",
                f"This group peaked in {row['peak_year']} with {row['peak_year_messages']:,} messages but has gone quiet. {row['total_messages']:,} total across its lifetime."
            ))

    # 8. Rising groups
    if not lifecycle.empty:
        rising = lifecycle[lifecycle['classification'] == 'rising']
        if not rising.empty:
            row = rising.iloc[0]
            insights.append((
                f"On the Rise: {row['group_name']}",
                f"This group is gaining momentum. {row['total_messages']:,} total messages and growing, active for {row['years_active']} years."
            ))

    # 9. Year-over-year group chat growth
    yearly_totals = df.groupby('year').size()
    if len(yearly_totals) >= 2:
        recent_year = yearly_totals.index.max()
        prev_year = recent_year - 1
        if prev_year in yearly_totals.index:
            recent = yearly_totals[recent_year]
            prev = yearly_totals[prev_year]
            if recent > prev * 1.3:
                insights.append((
                    "Group Chats Are Exploding",
                    f"You sent {recent:,} group messages in {recent_year} vs {prev:,} in {prev_year}. That's a {(recent/prev - 1)*100:.0f}% increase."
                ))
            elif prev > recent * 1.3:
                insights.append((
                    "Group Chats Slowing Down",
                    f"You sent {recent:,} group messages in {recent_year} vs {prev:,} in {prev_year}. Group chat activity is declining."
                ))

    # 10. Your activity across groups
    your_msgs = df[df['is_from_me'] == 1]
    your_pct = len(your_msgs) / len(df) * 100 if len(df) > 0 else 0
    n_groups = df['chat_id'].nunique()
    insights.append((
        "Your Group Presence",
        f"You account for {your_pct:.1f}% of all group messages across {n_groups} groups. {'You are a major contributor!' if your_pct > 20 else 'You mostly listen.' if your_pct < 10 else 'A balanced participant.'}"
    ))

    # 11. People in the most groups
    member_group_counts = members_df.groupby('contact_name')['chat_id'].nunique().sort_values(ascending=False)
    # Filter out unresolved names (phone numbers)
    resolved_members = member_group_counts[~member_group_counts.index.str.match(r'^[\+\d]')]
    if not resolved_members.empty and resolved_members.iloc[0] >= 3:
        top_member = resolved_members.index[0]
        n_groups_in = resolved_members.iloc[0]
        insights.append((
            f"Social Connector: {top_member}",
            f"{top_member} appears in {n_groups_in} of your group chats. They are a bridge across your social circles."
        ))

    return insights


if __name__ == "__main__":
    main()
