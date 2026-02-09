from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from openai import OpenAI

app = Flask(__name__)
CORS(app)

nltk.download('vader_lexicon', quiet=True)

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'PH99oWZjM43GimMtYigFvA')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '3tJsXQKEtFFYInxzLEDqRZ0s_w5z0g')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set in environment variables")

print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent='Sentivity_PR_Crisis_Monitor',
    check_for_async=False
)
print("Reddit client ready")

client = None

def get_openai_client():
    global client
    if client is None:
        print("Initializing OpenAI client...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client ready")
    return client

sia = SentimentIntensityAnalyzer()

CRISIS_SUBREDDITS = [
    "news", "worldnews", "politics", "business",
    "technology", "PublicFreakout", "OutOfTheLoop"
]

def reddit_influence(row) -> float:
    score = max(row.get("score", 0) or 0, 0)
    comments = max(row.get("num_comments", 0) or 0, 0)
    return np.log1p(score) + 0.5 * np.log1p(comments)

def compute_negativity(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    compound = sia.polarity_scores(text)["compound"]
    return max(0.0, -compound)

def fetch_reddit_posts(subreddits: List[str], days: int = 7, limit: int = 150):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    all_posts = []
    
    for sub in subreddits:
        try:
            print(f"  Fetching from r/{sub}...")
            subreddit = reddit.subreddit(sub)
            
            for post in subreddit.hot(limit=limit):
                created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                
                if created < cutoff:
                    break
                
                all_posts.append({
                    'source': f'r/{sub}',
                    'title': post.title,
                    'url': f"https://reddit.com{post.permalink}",
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'datetime': created.isoformat()
                })
        except Exception as e:
            print(f"Error fetching r/{sub}: {e}")
            continue
    
    df = pd.DataFrame(all_posts)
    
    if df.empty:
        return df
    
    initial = len(df)
    df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"Collected {initial} posts, {len(df)} after dedup")
    
    return df

def analyze_crisis(days: int = 7, limit: int = 150):
    print("Starting crisis analysis...")
    
    df = fetch_reddit_posts(CRISIS_SUBREDDITS, days, limit)
    
    if df.empty:
        return {"error": "No posts collected"}
    
    print("Calculating scores...")
    
    df['influence'] = df.apply(reddit_influence, axis=1)
    df['negativity'] = df['title'].apply(compute_negativity)
    df['negative_impact_score'] = df['negativity'] * df['influence']
    
    df_crisis = df[
        (df['negative_impact_score'] >= 1) &
        (df['influence'] > 0)
    ].sort_values('negative_impact_score', ascending=False).reset_index(drop=True)
    
    if df_crisis.empty:
        return {
            "companies_in_crisis": [],
            "crisis_report": "No significant crisis signals detected in the monitored timeframe."
        }
    
    print(f"Found {len(df_crisis)} crisis signals")
    
    print("Extracting entities with OpenAI...")
    titles = df_crisis['title'].tolist()[:150]
    text_blob = "\n".join(f"- {t}" for t in titles)
    
    extraction_prompt = f"""
Extract the names of companies, organizations, groups. NO INDIVIDUAL PEOPLE
that are being criticized or facing backlash in the text below.

Return ONLY a comma-separated list of unique names.
No explanations. No bullet points. No extra words.

TEXT:
{text_blob}
""".strip()
    
    try:
        openai_client = get_openai_client()
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
            max_tokens=200
        )
        
        raw = resp.choices[0].message.content.replace("\n", " ")
        raw = re.sub(r"\s*,\s*", ", ", raw)
        raw = re.sub(r"\s{2,}", " ", raw).strip(" ,")
        
        entities = []
        seen = set()
        for e in raw.split(","):
            e = e.strip()
            if e and e.lower() not in seen:
                seen.add(e.lower())
                entities.append(e)
        
        print(f"Extracted {len(entities)} entities: {', '.join(entities[:10])}")
    except Exception as e:
        print(f"Error extracting entities: {e}")
        entities = []
    
    if not entities:
        return {
            "companies_in_crisis": [],
            "crisis_report": "Analysis completed but no clear company/organization entities were identified in crisis."
        }
    
    def match_entities(title):
        title_l = title.lower()
        return [e for e in entities if e.lower() in title_l]
    
    df_crisis['entities'] = df_crisis['title'].apply(match_entities)
    
    df_exploded = (
        df_crisis
        .explode('entities')
        .dropna(subset=['entities'])
        .rename(columns={'entities': 'entity'})
    )
    
    if df_exploded.empty:
        return {
            "companies_in_crisis": [],
            "crisis_report": "Entities extracted but none matched to crisis posts."
        }
    
    df_entity_grouped = (
        df_exploded
        .groupby('entity', as_index=False)
        .agg(
            avg_negative_impact=('negative_impact_score', 'mean'),
            mentions=('entity', 'count'),
            titles=('title', list),
            urls=('url', list)
        )
        .sort_values('avg_negative_impact', ascending=False)
        .reset_index(drop=True)
    )
    
    print("Generating crisis summaries with OpenAI...")
    crisis_summaries = []
    openai_client = get_openai_client()
    
    for _, row in df_entity_grouped.head(20).iterrows():
        try:
            summary_resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarize the potential PR or reputational crisis in ONE sentence, "
                        "based ONLY on the following headlines. Be concrete and neutral.\n\n"
                        f"ENTITY: {row['entity']}\n"
                        "HEADLINES:\n" + "\n".join(f"- {t}" for t in row['titles'][:6])
                    )
                }],
                temperature=0,
                max_tokens=60
            )
            row['crisis_summary'] = summary_resp.choices[0].message.content.strip()
            crisis_summaries.append(row['crisis_summary'])
        except Exception as e:
            print(f"Error generating summary for {row['entity']}: {e}")
            row['crisis_summary'] = f"Facing criticism across {row['mentions']} mentions"
            crisis_summaries.append(row['crisis_summary'])
    
    print("Generating holistic report...")
    df_pack = df_entity_grouped.head(20).copy()
    
    brief_blocks = []
    for _, row in df_pack.iterrows():
        entity = str(row['entity'])
        avg_imp = float(row['avg_negative_impact'])
        mentions = int(row['mentions'])
        
        urls = row['urls'][:3] if isinstance(row['urls'], list) else []
        titles = row['titles'][:3] if isinstance(row['titles'], list) else []
        
        evidence_lines = []
        for i, t in enumerate(titles, 1):
            if i-1 < len(urls):
                evidence_lines.append(f"  - [{t}]({urls[i-1]})")
            else:
                evidence_lines.append(f"  - {t}")
        
        brief_blocks.append(
            f"ENTITY: {entity}\n"
            f"AVG_NEG_IMPACT: {avg_imp:.3f}\n"
            f"MENTIONS: {mentions}\n"
            f"EVIDENCE:\n" + "\n".join(evidence_lines)
        )
    
    briefing = "\n\n".join(brief_blocks)
    
    try:
        openai_client = get_openai_client()
        report_resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "You are writing an internal monitoring brief for Sentivity.ai.\n"
                    "Using ONLY the briefing pack below, write holistic bullet points on what to watch.\n"
                    "Rules:\n"
                    "- Output ONLY bullet points (each line begins with '- ').\n"
                    "- Each bullet should mention the entity and key concern.\n"
                    "- Combine closely-related items when they refer to the same narrative.\n"
                    "- Be actionable: who/what, why it matters, and what to monitor next.\n"
                    "- Do NOT invent facts beyond the evidence.\n\n"
                    f"BRIEFING PACK:\n{briefing}"
                )
            }],
            temperature=0,
            max_tokens=450
        )
        
        holistic_report = report_resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating holistic report: {e}")
        holistic_report = "Crisis monitoring report generation failed. Please review individual company summaries."
    
    companies_in_crisis = []
    for idx, row in df_entity_grouped.head(20).iterrows():
        companies_in_crisis.append({
            "rank": int(idx + 1),
            "company": {
                "id": f"crisis_{row['entity'].lower().replace(' ', '_')}",
                "name": row['entity']
            },
            "crisis_metrics": {
                "avg_negative_impact": round(float(row['avg_negative_impact']), 2),
                "mentions": int(row['mentions']),
                "crisis_summary": row.get('crisis_summary', 'Summary unavailable')
            },
            "evidence": [
                {
                    "title": title,
                    "url": url
                } for title, url in zip(row['titles'][:3], row['urls'][:3])
            ]
        })
    
    print(f"Analysis complete. Returning {len(companies_in_crisis)} companies")
    
    return {
        "companies_in_crisis": companies_in_crisis,
        "crisis_report": holistic_report
    }

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Feature 3 - PR Crisis Detection",
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/analyze": "Run crisis analysis"
        },
        "note": "Uses influence algorithm + VADER + OpenAI for entity extraction and report generation"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        days = int(request.args.get('days', 7))
        limit = int(request.args.get('limit', 150))
        
        if limit < 100:
            limit = 100
            print(f"Limit increased to minimum: 100")
        
        print(f"Starting crisis analysis (days={days}, limit={limit})")
        
        result = analyze_crisis(days, limit)
        
        if "error" in result:
            return jsonify(result), 400
        
        response = {
            "data": result,
            "meta": {
                "total": len(result.get('companies_in_crisis', [])),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_range_days": days,
                "posts_per_subreddit": limit,
                "scoring": "influence algorithm (log scaling) + VADER + OpenAI"
            }
        }
        
        print(f"Analysis complete. Returning {response['meta']['total']} companies")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "tip": "Try reducing the 'limit' parameter or check OpenAI API key"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
