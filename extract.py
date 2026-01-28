"""Extract iMessage data from chat.db."""
import sqlite3
import pandas as pd
import re
from datetime import datetime
from config import CHAT_DB_PATH, START_YEAR

# Apple's Cocoa epoch: 2001-01-01 00:00:00 UTC
COCOA_EPOCH_OFFSET = 978307200


def extract_text_from_attributed_body(blob):
    """Extract plain text from NSAttributedString blob."""
    if blob is None:
        return None

    try:
        # The text is stored after 'NSString' marker in the blob
        # Look for the pattern: NSString followed by length byte(s) and then text

        # Method 1: Find text after NSString marker
        text_marker = b'NSString'
        idx = blob.find(text_marker)
        if idx != -1:
            # Skip past NSString and some header bytes
            start = idx + len(text_marker)
            # Look for the actual text content
            # Usually format is: NSString + some bytes + length + text
            remaining = blob[start:start+500]

            # Find printable text sequences
            decoded = remaining.decode('utf-8', errors='ignore')
            # Remove control characters but keep the text
            # Text usually starts after a few bytes
            for i in range(min(20, len(decoded))):
                substring = decoded[i:]
                # Check if this looks like the start of real text
                if substring and substring[0].isprintable() and not substring[0].isspace():
                    # Find where the text ends (usually at a control character)
                    end = 0
                    for j, c in enumerate(substring):
                        if ord(c) < 32 and c not in '\n\r\t':
                            end = j
                            break
                    else:
                        end = len(substring)

                    if end > 0:
                        text = substring[:end].strip()
                        if len(text) > 0:
                            return text

        # Method 2: Try to find text between common delimiters
        decoded = blob.decode('utf-8', errors='ignore')
        # Look for substantial printable sequences
        matches = re.findall(r'[\x20-\x7e\n]{5,}', decoded)
        # Filter out known non-text patterns
        for match in matches:
            if not any(skip in match.lower() for skip in ['nsstring', 'nsattributed', 'nsdictionary', 'streamtyped', 'nsmutable', '__kim']):
                cleaned = match.strip()
                if cleaned and len(cleaned) > 1:
                    return cleaned

        return None

    except Exception as e:
        return None


def connect_db():
    """Connect to iMessage database (read-only)."""
    return sqlite3.connect(f"file:{CHAT_DB_PATH}?mode=ro", uri=True)


def _coerce_from_me(x):
    """Coerce is_from_me to 0/1."""
    if pd.isna(x):
        return 0
    try:
        v = int(x) if isinstance(x, (int, float, bool)) else int(float(str(x).strip() or 0))
    except (ValueError, TypeError):
        return 0
    return 1 if v else 0


def _process_message_df(df, label="messages"):
    """Common post-processing for extracted message DataFrames.

    Handles: is_from_me coercion, text extraction from attributedBody,
    attachment sentinel text, reaction filtering, timestamp conversion,
    date range filtering, and contact_id generation.
    """
    print(f"Raw {label} fetched: {len(df):,}")

    # Filter out reactions (associated_message_type != 0)
    if 'associated_message_type' in df.columns:
        n_before = len(df)
        df = df[df['associated_message_type'].fillna(0).astype(int) == 0].copy()
        n_reactions = n_before - len(df)
        if n_reactions > 0:
            print(f"  Filtered out {n_reactions:,} reactions/tapbacks")
        df = df.drop(columns=['associated_message_type'])

    # Coerce is_from_me
    df['is_from_me'] = df['is_from_me'].apply(_coerce_from_me).astype(int)
    n_from_me = df['is_from_me'].sum()
    n_from_them = len(df) - n_from_me
    print(f"  Messages from you: {n_from_me:,} | from them: {n_from_them:,}")

    # Extract text from attributedBody where text is NULL
    def get_message_text(row):
        if row['text'] and str(row['text']).strip():
            return str(row['text'])
        if row['attributedBody']:
            extracted = extract_text_from_attributed_body(row['attributedBody'])
            if extracted:
                return extracted
        return None

    df['text'] = df.apply(get_message_text, axis=1)

    # Track attachments before dropping rows with no text
    has_attachment = df['cache_has_attachments'].fillna(0).astype(int) == 1 if 'cache_has_attachments' in df.columns else pd.Series(False, index=df.index)
    df['has_attachment'] = has_attachment

    # For attachment-only messages (no text), set sentinel text
    attachment_no_text = df['text'].isna() & df['has_attachment']
    df.loc[attachment_no_text, 'text'] = '[attachment]'
    n_attachment_only = attachment_no_text.sum()
    if n_attachment_only > 0:
        print(f"  Attachment-only messages preserved: {n_attachment_only:,}")

    # Drop the blob column and filter out messages with no text
    df = df.drop(columns=['attributedBody'], errors='ignore')
    if 'cache_has_attachments' in df.columns:
        df = df.drop(columns=['cache_has_attachments'])
    df = df[df['text'].notna() & (df['text'] != '')]

    print(f"  {label.capitalize()} with content: {len(df):,}")

    # Convert timestamps to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date

    # Filter to our date range
    df = df[df['year'] >= START_YEAR]

    return df


def extract_messages():
    """Extract all 1:1 messages with metadata."""
    query = """
    SELECT
        m.ROWID as message_id,
        m.text,
        m.attributedBody,
        m.is_from_me,
        m.date / 1000000000 + ? as timestamp,
        m.date_read / 1000000000 + ? as timestamp_read,
        m.date_delivered / 1000000000 + ? as timestamp_delivered,
        m.associated_message_type,
        m.cache_has_attachments,
        h.id as handle_id,
        h.service,
        c.ROWID as chat_id,
        c.chat_identifier
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    JOIN chat c ON cmj.chat_id = c.ROWID
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE
        c.chat_identifier NOT LIKE 'chat%'
        AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL OR m.cache_has_attachments = 1)
    ORDER BY m.date
    """

    with connect_db() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(COCOA_EPOCH_OFFSET, COCOA_EPOCH_OFFSET, COCOA_EPOCH_OFFSET)
        )

    df = _process_message_df(df, label="1:1 messages")

    # Clean handle_id (use chat_identifier as fallback)
    df['contact_id'] = df['handle_id'].fillna(df['chat_identifier'])

    print(f"Extracted {len(df):,} messages from {df['year'].min()} to {df['year'].max()}")
    print(f"Unique contacts: {df['contact_id'].nunique()}")

    return df


def extract_group_messages():
    """Extract all group chat messages with metadata."""
    query = """
    SELECT
        m.ROWID as message_id,
        m.text,
        m.attributedBody,
        m.is_from_me,
        m.date / 1000000000 + ? as timestamp,
        m.date_read / 1000000000 + ? as timestamp_read,
        m.date_delivered / 1000000000 + ? as timestamp_delivered,
        m.associated_message_type,
        m.cache_has_attachments,
        h.id as handle_id,
        h.service,
        c.ROWID as chat_id,
        c.chat_identifier,
        c.display_name as group_name
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    JOIN chat c ON cmj.chat_id = c.ROWID
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE
        c.chat_identifier LIKE 'chat%'
        AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL OR m.cache_has_attachments = 1)
    ORDER BY m.date
    """

    with connect_db() as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(COCOA_EPOCH_OFFSET, COCOA_EPOCH_OFFSET, COCOA_EPOCH_OFFSET)
        )

    df = _process_message_df(df, label="group messages")

    # For group chats, contact_id is the sender's handle_id (NULL for is_from_me=1)
    df['contact_id'] = df['handle_id']

    print(f"Extracted {len(df):,} group messages from {df['year'].min()} to {df['year'].max()}")
    print(f"Unique group chats: {df['chat_id'].nunique()}")

    return df


def extract_group_members():
    """Extract membership of each group chat."""
    query = """
    SELECT
        c.ROWID as chat_id,
        c.chat_identifier,
        c.display_name as group_name,
        h.id as handle_id
    FROM chat c
    JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
    JOIN handle h ON chj.handle_id = h.ROWID
    WHERE c.chat_identifier LIKE 'chat%'
    """

    with connect_db() as conn:
        df = pd.read_sql_query(query, conn)

    print(f"Group membership: {len(df):,} entries across {df['chat_id'].nunique()} groups")
    return df


if __name__ == "__main__":
    df = extract_messages()
    print(df.head())
    print(f"\nMessages per year:")
    print(df.groupby('year').size())

    print("\n\n=== GROUP MESSAGES ===")
    gdf = extract_group_messages()
    print(gdf.head())
    print(f"\nGroup messages per year:")
    print(gdf.groupby('year').size())

    print("\n\n=== GROUP MEMBERS ===")
    members = extract_group_members()
    print(members.head(20))
