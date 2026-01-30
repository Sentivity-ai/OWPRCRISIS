from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
import praw
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = Flask(__name__)
CORS(app)

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', 'PH99oWZjM43GimMtYigFvA')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '3tJsXQKEtFFYInxzLEDqRZ0s_w5z0g')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'sentivity-crisis-api')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'AIzaSyAZwLva1HxzDbKFJuE9RVcxS5B4q_ol8yE')

print("Downloading NLTK VADER lexicon...")
try:
    nltk.download('vader_lexicon', quiet=True)
    print("VADER lexicon ready")
except:
    print("Warning: VADER lexicon download failed")

sia = SentimentIntensityAnalyzer()
nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        try:
            import spacy
            print("Loading spaCy model...")
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded")
        except:
            print("Warning: spaCy model not available")
            nlp = False
    return nlp if nlp is not False else None

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json,text/html",
    "Accept-Language": "en-US,en;q=0.9",
}

REDDIT_CRISIS_QUERIES = [
    "company scandal",
    "brand boycott",
    "company controversy",
    "corporate scandal",
]

YOUTUBE_CRISIS_QUERIES = [
    "company boycott",
    "brand backlash",
    "company apology",
]

STOP_WORDS = {
    "reddit","youtube","inc","llc","ltd","corp","co","company","companies",
    "brand","brands","ceo","fda","sec","ftc","eu","us","usa","uk",
}

def utc_now():
    return datetime.now(timezone.utc)

def window_to_utc_range(window: str):
    w = window.strip().lower()
    end = utc_now()
    
    if "hour" in w:
        n = int("".join([c for c in w if c.isdigit()]) or "24")
        start = end - timedelta(hours=n)
    elif "day" in w:
        n = int("".join([c for c in w if c.isdigit()]) or "7")
        start = end - timedelta(days=n)
    elif "week" in w:
        n = int("".join([c for c in w if c.isdigit()]) or "2")
        start = end - timedelta(weeks=n)
    elif "month" in w:
        n = int("".join([c for c in w if c.isdigit()]) or "6")
        start = end - timedelta(days=30 * n)
    else:
        start = end - timedelta(days=7)
    
    return start, end

def reddit_client():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        check_for_async=False,
    )

def fetch_reddit_posts(query: str, window: str = "Past 7 Days", limit_per_sub: int = 50):
    reddit = reddit_client()
    start, end = window_to_utc_range(window)
    
    subreddits = ["all"]
    rows = []
    
    for sub in subreddits:
        try:
            sr = reddit.subreddit(sub)
            coarse_tf = "week"
            
            for s in sr.search(query, sort="new", time_filter=coarse_tf, limit=limit_per_sub):
                created = datetime.fromtimestamp(s.created_utc, tz=timezone.utc)
                if created < start or created > end:
                    continue
                
                title = (s.title or "").strip()
                body = (getattr(s, "selftext", "") or "").strip()
                full_text = (title + "\n\n" + body).strip()
                
                rows.append({
                    "platform": "reddit",
                    "query": query,
                    "datetime": created.isoformat(),
                    "title": title,
                    "text": body,
                    "full_text": full_text,
                    "score": getattr(s, "score", None),
                    "num_comments": getattr(s, "num_comments", None),
                    "url": f"https://www.reddit.com{s.permalink}" if getattr(s, "permalink", None) else None,
                })
        except Exception as e:
            print(f"Error fetching Reddit for {query}: {e}")
            continue
    
    return pd.DataFrame(rows)

def compute_negativity(text: str):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    compound = sia.polarity_scores(text)["compound"]
    return max(0.0, -compound)

def compute_influence_reddit(row):
    score = max(row.get("score", 0) or 0, 0)
    comments = max(row.get("num_comments", 0) or 0, 0)
    return np.log1p(score) + 0.5 * np.log1p(comments)

def extract_orgs(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    
    nlp_model = get_nlp()
    
    if nlp_model is not None:
        doc = nlp_model(text[:5000])
        orgs = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                name = re.sub(r"\s+", " ", ent.text.strip())
                if len(name) >= 3 and name.lower() not in STOP_WORDS:
                    orgs.append(name)
        return list(dict.fromkeys(orgs))
    
    cands = re.findall(r"\b(?:[A-Z][A-Za-z&.-]{1,}\s){0,3}[A-Z][A-Za-z&.-]{1,}\b", text)
    orgs = []
    for c in cands:
        c = c.strip()
        if len(c) >= 3 and c.lower() not in STOP_WORDS:
            orgs.append(c)
    return list(dict.fromkeys(orgs))

def normalize_name(name: str):
    if not isinstance(name, str):
        return ""
    n = re.sub(r"['']", "", name)
    n = re.sub(r"\b(Inc|LLC|Ltd|Corporation|Corp|Co)\b\.?", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def analyze_pr_crisis(window: str = "Past 7 Days", limit: int = 50):
    print(f"Starting PR crisis analysis (window={window}, limit={limit})")
    
    dfs = []
    
    for query in REDDIT_CRISIS_QUERIES:
        df_r = fetch_reddit_posts(query, window=window, limit_per_sub=limit)
        df_r["crisis_query"] = query
        df_r["source"] = "reddit"
        dfs.append(df_r)
    
    if not dfs:
        return {"error": "No data collected"}
    
    df_crisis = pd.concat(dfs, ignore_index=True)
    
    if len(df_crisis) == 0:
        return {"error": "No posts found"}
    
    df_crisis["negativity"] = df_crisis["full_text"].apply(compute_negativity)
    df_crisis["influence"] = df_crisis.apply(compute_influence_reddit, axis=1)
    df_crisis["negative_impact_score"] = df_crisis["negativity"] * df_crisis["influence"]
    
    df_crisis["orgs"] = df_crisis["full_text"].apply(extract_orgs)
    
    df_e = df_crisis[df_crisis["orgs"].map(len) > 0].copy()
    if len(df_e) == 0:
        return {"error": "No companies found"}
    
    df_e = df_e.explode("orgs").rename(columns={"orgs":"entity"})
    df_e["entity_norm"] = df_e["entity"].apply(normalize_name)
    df_e = df_e[df_e["entity_norm"].str.len() >= 3]
    
    org_count = df_crisis["orgs"].map(len).replace(0, np.nan)
    df_e["entity_impact"] = df_e["negative_impact_score"].fillna(0.0) / org_count[df_e.index].fillna(1.0)
    
    company_scores = (
        df_e.groupby("entity_norm", as_index=False)
        .agg(
            total_negative_impact=("entity_impact", "sum"),
            post_count=("entity_impact", "count"),
            max_single_hit=("entity_impact", "max"),
        )
    )
    
    company_scores = company_scores[company_scores["post_count"] >= 2]
    
    def ranked_urls(entity):
        entity_posts = df_e[df_e["entity_norm"] == entity].sort_values("entity_impact", ascending=False)
        urls = entity_posts["url"].dropna().tolist()
        seen, out = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
            if len(out) >= 5:
                break
        return out
    
    company_scores["top_urls"] = company_scores["entity_norm"].apply(ranked_urls)
    
    top_companies = company_scores.sort_values("total_negative_impact", ascending=False).head(20)
    
    print(f"Analysis complete. Found {len(top_companies)} companies")
    
    return top_companies

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "service": "Feature 3 - PR Crisis Detection",
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/analyze": "Run crisis analysis"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        window = request.args.get('window', 'Past 7 Days')
        limit = int(request.args.get('limit', 30))
        
        print(f"Starting PR crisis analysis (window={window}, limit={limit})")
        
        result = analyze_pr_crisis(window=window, limit=limit)
        
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400
        
        ranked_companies = []
        for idx, row in result.iterrows():
            ranked_companies.append({
                "rank": int(idx + 1),
                "company": {
                    "id": f"crisis_{row['entity_norm'].lower().replace(' ', '_')}",
                    "name": row['entity_norm']
                },
                "crisis_metrics": {
                    "negative_impact_score": round(float(row['total_negative_impact']), 2),
                    "mentions": int(row['post_count']),
                    "severity": round(float(row['max_single_hit']), 2)
                },
                "evidence": row['top_urls'][:3] if len(row['top_urls']) > 0 else []
            })
        
        response = {
            "data": {
                "companies_in_crisis": ranked_companies
            },
            "meta": {
                "total": len(ranked_companies),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_window": window,
                "posts_per_query": limit
            }
        }
        
        print(f"Analysis complete. Found {len(ranked_companies)} companies")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Analysis failed",
            "message": str(e),
            "tip": "Try reducing the 'limit' parameter"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
