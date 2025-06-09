# utils/news_utils.py
import streamlit as st
import pandas as pd
import feedparser
from transformers import pipeline
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
import nltk
import re
from datetime import datetime, timedelta

nltk.download('stopwords')

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_data(ttl=1800)
def parse_rss_feed(url):
    try:
        feed = feedparser.parse(url)
        data = []
        for entry in feed.entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", "") or entry.get("updated", "")
            summary = entry.get("summary", "")
            data.append({
                "Title": title,
                "Link": link,
                "Published": published,
                "Summary": summary,
            })
        df = pd.DataFrame(data)
        df["Published_dt"] = pd.to_datetime(df["Published"], errors='coerce')

        def to_naive(dt):
            if pd.isna(dt):
                return dt
            if dt.tzinfo is not None:
                return dt.tz_convert('UTC').tz_localize(None)
            else:
                return dt

        df["Published_dt"] = df["Published_dt"].apply(to_naive)
        return df
    except Exception as e:
        st.error(f"Failed to parse RSS feed {url}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def analyze_sentiment(df, _sentiment_model):
    if df.empty:
        return df
    sentiments = _sentiment_model(df["Title"].tolist())
    df["Sentiment"] = [s["label"] for s in sentiments]
    df["Score"] = [round(s["score"], 2) for s in sentiments]
    return df

def plot_wordcloud(text_series):
    if text_series.empty:
        st.info("No data to generate WordCloud.")
        return
    text = " ".join(text_series)
    stop_words = set(stopwords.words("english"))
    wc = WordCloud(stopwords=stop_words, background_color="white", width=800, height=400).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def render_clickable_links(df):
    if df.empty:
        st.info("No articles to display.")
        return

    lines = []
    for _, row in df.iterrows():
        title = row["Title"]
        url = row["Link"]
        source = row.get("Source", "Unknown Source")
        published = row.get("Published", "")
        line = f"**[{title}]({url})**  \n*{source} | {published}*"
        lines.append(line)
    md = "\n\n---\n\n".join(lines)

    with st.container():
        st.markdown(
            """
            <style>
            .scrollable-container {
                max-height: 500px;
                overflow-y: auto;
                padding-right: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="scrollable-container">{md}</div>', unsafe_allow_html=True)

def is_trading_relevant(text):
    keywords = [
        r'\bstock\b', r'\bmarket\b', r'\btrading\b', r'\bbull\b', r'\bbear\b',
        r'\binvestment\b', r'\bshares\b', r'\bprice\b', r'\bbroker\b', r'\bvolatility\b',
        r'\bipo\b', r'\bdividend\b', r'\bearnings\b', r'\bprofit\b', r'\bloss\b', r'\bsector\b',
        r'\bindex\b', r'\bbonds\b', r'\bcommodities\b', r'\bforex\b', r'\beconomy\b',
        r'\bcrude\b', r'\boil\b', r'\bmetal\b', r'\bexchange\b', r'\bfed\b', r'\binflation\b',
        r'\binterest rate\b', r'\brevenue\b', r'\bdebt\b', r'\bfinancial\b'
    ]
    text = text.lower()
    return any(re.search(kw, text) for kw in keywords)

def filter_trading_relevant_news(df):
    if df.empty:
        return df
    mask = df.apply(lambda row: is_trading_relevant(row['Title']) or is_trading_relevant(row['Summary']), axis=1)
    return df[mask]

def news_dashboard():
    st.subheader("📰 AI-Powered News Dashboard")

    rss_feeds = {
        # India
        "NDTV Business": "https://feeds.feedburner.com/ndtvprofit-latest",
        "Economic Times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
        "Livemint": "https://www.livemint.com/rss/news.xml",
        # US & Global
        "CNN Top Stories": "http://rss.cnn.com/rss/cnn_topstories.rss",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "MarketWatch": "https://www.marketwatch.com/rss/topstories",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "Market Watch":"https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
        "Seeking Alpha":"https://seekingalpha.com/market_currents.xml",
        "Google News":"https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
        "CNBC TV 18":"https://www.cnbctv18.com/commonfeeds/v1/cne/rss/latest.xml",
    }

    choices = ["All Sources"] + list(rss_feeds.keys())
    feed_choice = st.selectbox("Choose a News Source", choices)

    cutoff_date = pd.Timestamp(datetime.utcnow() - timedelta(days=21))

    if st.button("Clear Cache and Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    if feed_choice == "All Sources":
        all_news = pd.DataFrame()
        progress_bar = st.progress(0)
        total = len(rss_feeds)

        for idx, (source, url) in enumerate(rss_feeds.items()):
            st.info(f"Fetching news from: {source}")
            df = parse_rss_feed(url)
            df = df[df["Published_dt"] >= cutoff_date]
            if not df.empty:
                df["Source"] = source
                all_news = pd.concat([all_news, df], ignore_index=True)
            progress_bar.progress((idx + 1) / total)

        if all_news.empty:
            st.warning("No news fetched from the RSS feeds in the last 3 weeks.")
            return

        all_news = all_news.sort_values(by="Published_dt", ascending=False).reset_index(drop=True)

        show_relevant = st.checkbox("Show only trading-relevant news")
        if show_relevant:
            filtered_news = filter_trading_relevant_news(all_news)
            if filtered_news.empty:
                st.warning("No trading-relevant news found.")
                return
            st.write(f"### Trading Relevant News ({len(filtered_news)})")
            render_clickable_links(filtered_news)
            all_news = filtered_news
        else:
            st.write(f"### Total articles fetched: {len(all_news)}")
            render_clickable_links(all_news[["Source", "Published", "Title", "Link"]])

        source_filter = st.multiselect("Filter by Source", options=all_news["Source"].unique(), default=all_news["Source"].unique())
        filtered_news = all_news[all_news["Source"].isin(source_filter)]

        min_date = filtered_news["Published_dt"].min()
        max_date = filtered_news["Published_dt"].max()
        date_range = st.date_input("Filter by Date Range", [min_date, max_date])
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_news = filtered_news[
                (filtered_news["Published_dt"] >= start_date) &
                (filtered_news["Published_dt"] <= end_date)
            ]

        if st.checkbox("Run Sentiment Analysis on filtered headlines"):
            sentiment_model = load_sentiment_model()
            filtered_news = analyze_sentiment(filtered_news, sentiment_model)
            AgGrid(filtered_news[["Source", "Published", "Title", "Sentiment", "Score", "Link"]])

        if st.checkbox("Show WordCloud of filtered headlines"):
            plot_wordcloud(filtered_news["Title"])

        keyword = st.text_input("Search headlines by keyword", "")
        if keyword:
            keyword_filtered = filtered_news[filtered_news["Title"].str.contains(keyword, case=False, na=False)]
            render_clickable_links(keyword_filtered[["Source", "Published", "Title", "Link"]])

    else:
        df_news = parse_rss_feed(rss_feeds[feed_choice])
        df_news = df_news[df_news["Published_dt"] >= cutoff_date]

        if df_news.empty:
            st.warning("No news fetched from the selected RSS feed in the last 3 weeks.")
            return

        df_news = df_news.sort_values(by="Published_dt", ascending=False).reset_index(drop=True)

        show_relevant = st.checkbox("Show only trading-relevant news")
        if show_relevant:
            filtered_news = filter_trading_relevant_news(df_news)
            if filtered_news.empty:
                st.warning("No trading-relevant news found.")
                return
            st.write(f"### Trading Relevant News ({len(filtered_news)})")
            render_clickable_links(filtered_news)
            df_news = filtered_news
        else:
            st.write("### Latest Headlines")
            render_clickable_links(df_news[["Published", "Title", "Link"]])

        if st.checkbox("Run Sentiment Analysis"):
            sentiment_model = load_sentiment_model()
            df_news = analyze_sentiment(df_news, sentiment_model)
            AgGrid(df_news[["Title", "Sentiment", "Score", "Link"]], height=350)

        if st.checkbox("Show WordCloud of Headlines"):
            plot_wordcloud(df_news["Title"])

        keyword = st.text_input("Search headlines by keyword", "")
        if keyword:
            filtered = df_news[df_news["Title"].str.contains(keyword, case=False, na=False)]
            render_clickable_links(filtered[["Published", "Title", "Link"]])
