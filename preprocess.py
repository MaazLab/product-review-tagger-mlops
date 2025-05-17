import pandas as pd
import re

# 1. Load Dataset
df = pd.read_csv('data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')
df = df[['reviews.text']].dropna().rename(columns={'reviews.text': 'review_text'})

# 2. Define tag keywords
TAG_KEYWORDS = {
    'price': ['cheap', 'expensive', 'price', 'cost', 'worth', 'value'],
    'quality': ['quality', 'durable', 'sturdy', 'fragile', 'broken', 'defect', 'material'],
    'delivery': ['delivery', 'shipping', 'arrived', 'late', 'delayed', 'on time'],
    'customer_service': ['support', 'return', 'refund', 'service', 'helpful', 'agent'],
    'packaging': ['package', 'packaging', 'box', 'wrap', 'sealed', 'damaged'],
    'battery': ['battery', 'charge', 'charging', 'power', 'long-lasting', 'drain']
}

# 3. Tagging function
def generate_tags(text):
    text = text.lower()
    tags = []
    for label, keywords in TAG_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                tags.append(label)
                break
    return tags if tags else ['misc']  # Add 'misc' if no tag is matched

# 4. Apply tagging
df['tags'] = df['review_text'].apply(generate_tags)
df['tags'] = df['tags'].apply(lambda tag_list: ','.join(tag_list))

# 5. Drop empty or too-short reviews (optional)
df = df[df['review_text'].str.len() > 20]

# 6. Save processed data
df.to_csv('data/processed_reviews.csv', index=False)

print(f"✅ Preprocessing complete. {len(df)} reviews saved to data/processed_reviews.csv")
