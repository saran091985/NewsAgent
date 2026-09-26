"""
All knobs for the daily news pipeline live here.

Edit this file (not the code) to change sources, categories, topic quotas
or the time window.
"""

from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Time window — "today's news"
# ---------------------------------------------------------------------------
# A story is kept only if its publish time falls inside the last WINDOW_HOURS
# before the run. Every source gives an exact timestamp, so old news cannot
# slip in (this is the part the search-based agent could not guarantee).
WINDOW_HOURS = 24
LOCAL_TZ = ZoneInfo("Asia/Dubai")  # how times are shown in the review screen

# ---------------------------------------------------------------------------
# The Hindu — latest-news scraper (free)
# ---------------------------------------------------------------------------
HINDU_MAX_PAGES = 15  # hard stop; scraping normally stops earlier, at the window edge

# Categories kept from The Hindu (case-insensitive). Everything else — city
# and state pages such as Karnataka, Madurai, Coimbatore — is dropped.
HINDU_KEEP_CATEGORIES = {
    # chosen by you
    "business", "economy", "india", "industry", "news", "shorts",
    "premium", "sport", "videos", "world",
    # sport sub-categories (PSG / IPL / tennis stories are filed under these)
    "cricket", "tennis", "football", "hockey", "athletics", "races",
    "other sports", "chess", "motorsport",
    # science: the meteor and blue-micromoon picks on 31 May were filed here
    "science", "technology", "sci-tech", "environment",
}

# Titles matching these are round-ups of other stories, not stories.
HINDU_SKIP_TITLE_PATTERNS = [
    r"^top news of the day",
    r"^morning digest",
    r"^evening wrap",
    r"\| the hindu (health wrap|parley|podcast)",
    r"^the hindu (e-paper|crossword)",
    r"^daily quiz",
]

# URL sections to skip even inside kept categories (reader essays, editorials)
HINDU_SKIP_URL_PATTERNS = [r"/opinion/", r"/life-and-style/", r"/books/", r"/crossword"]

# ---------------------------------------------------------------------------
# Other free sources (RSS). Each has an exact publish time per item.
# Run  `uv run python -m newsagent check-feeds`  to test them on your machine.
# ---------------------------------------------------------------------------
# Google News RSS search: free, no key, supports "when:1d" and site: filters.
_GN = "https://news.google.com/rss/search?q={q}&hl=en-{gl}&gl={gl}&ceid={gl}:en"


def google_news(query: str, country: str = "IN") -> str:
    from urllib.parse import quote_plus
    return _GN.format(q=quote_plus(query), gl=country)


RSS_FEEDS = [
    # ---- Space ----
    {"name": "NASA", "bucket": "Space & Science",
     "url": "https://www.nasa.gov/news-release/feed/"},
    # Space.com's direct feed blocks scripts, so read it through Google News
    {"name": "Google News: Space.com", "bucket": "Space & Science",
     "url": google_news("site:space.com when:1d", "US")},
    {"name": "Google News: ISRO / space", "bucket": "Space & Science",
     "url": google_news("ISRO OR NASA OR SpaceX OR \"Blue Origin\" OR rocket OR astronaut when:1d")},
    # ---- World ----
    {"name": "BBC World", "bucket": "World",
     "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "BBC Science", "bucket": "Space & Science",
     "url": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"},
    {"name": "BBC Technology", "bucket": "Tech",
     "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    # ---- UAE ----
    {"name": "Google News: Gulf News", "bucket": "UAE",
     "url": google_news("site:gulfnews.com when:1d", "AE")},
    {"name": "Google News: Khaleej Times", "bucket": "UAE",
     "url": google_news("site:khaleejtimes.com when:1d", "AE")},
]

RSS_MAX_ITEMS_PER_FEED = 30

# "Today in history" (free Wikipedia API)
ON_THIS_DAY_URL = "https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/selected/{mm}/{dd}"
HOLIDAYS_URL = "https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/holidays/{mm}/{dd}"
ON_THIS_DAY_MAX = 8

# ---------------------------------------------------------------------------
# How many stories make the episode, and the topic mix.
# ---------------------------------------------------------------------------
# Change TOP_N to pick more or fewer stories (or pass --top on the command line).
TOP_N = 20

# Topic mix for TOP_N = 20. For any other TOP_N the numbers are scaled
# proportionally. If a topic has no good story today, its slots go to others.
BUCKET_TARGETS = {
    "World": 4,
    "India": 4,
    "UAE": 3,
    "Sports": 2,
    "Space & Science": 3,
    "Tech": 2,
    "Weather & Nature": 1,
    "Business": 1,
}

# Model that scores the headlines (about $0.005-0.01 per pick with gpt-4o-mini).
# A stronger model (e.g. "gpt-4.1-mini" or "gpt-4o") judges better for a few cents more.
SELECT_MODEL = "gpt-4o-mini"

# "single": one AI call scores all headlines (default, one trace in LangSmith).
# "per_topic": one call per topic (8 calls in parallel) — each call sees fewer headlines.
SELECT_MODE = "single"

# Stories scoring below this (1-10) are only used if a topic has nothing better
MIN_PICK_SCORE = 5

# Your taste, shown to the AI as examples (topic, headline).
# Every time you click "Save list" or "Write scripts", what you ticked and what
# you un-ticked from the AI's picks is saved to output/<date>/feedback.json, and
# the last FEEDBACK_DAYS days of that feedback are added to these examples
# automatically — so the AI keeps learning from your corrections.
LEARN_FROM_MY_PICKS = True
FEEDBACK_DAYS = 7

EDITOR_PICKS = [
    ("World", "Iran chief negotiator says no deal with U.S. until Iranian rights secured"),
    ("World", "Drone attack struck turbine building at Ukraine's Zaporizhzhia nuclear plant, IAEA says"),
    ("World", "WHO hails five Ebola recoveries as new treatment centre opens in eastern Congo"),
    ("World", "Perilous Laos cave rescue saves trapped miners"),
    ("World", "U.S. looking to become huge source of India's energy needs: Senior State Department official"),
    ("World", "Iran suggests deal to reopen Strait of Hormuz in seven days; Trump rejects proposal"),
    ("World", "Saudi says it intercepted two drones launched by Houthis towards Riyadh"),
    ("World", "Russia targeting 'ordinary life' with attacks on Ukraine's data centres, Zelensky says"),
    ("India", "India and U.S. launch bilateral trade and critical minerals talks"),
    ("Sports", "PSG wins back-to-back Champions League titles after shootout victory against Arsenal"),
    ("Space & Science", "Meteor explodes over U.S. with blast equivalent to 300 tonnes of TNT"),
    ("Space & Science", "Blue Origin rocket explodes during test as SpaceX launch succeeds"),
    ("Space & Science", "China's Shenzhou-21 crew safely returns to Earth"),
    ("Space & Science", "A rare 'blue micromoon' lights up the night sky"),
    ("Tech", "EA FC 27 launches today with better gameplay, promising new mode"),
    ("Weather & Nature", "Massive dust storm sweeps through Rajasthan"),
]

EDITOR_REJECTS = [
    ("World", "Thousands turn out to greet Pope in France"),
    ("World", "The treasured 'eternal snow' on this tropical island is about to disappear forever"),
]

# ---------------------------------------------------------------------------
# Writing the scripts
# ---------------------------------------------------------------------------
WRITE_MODEL = "gpt-4o-mini"      # one call per story + one for intro/outro (~$0.01 a day)
SCRIPT_MINUTES = 12               # target video length; words per story are worked out from this
WORDS_PER_MINUTE = 130            # a 10-year-old reading clearly
DETAILED_WORDS = (110, 170)       # detailed version, per story
STYLE_GUIDE = "style/style_guide.md"   # tone rules + example stories the AI copies — edit freely
INCLUDE_SPECIAL_TODAY = True      # "Special today" segment (international days, famous firsts)
SEARCH_THIN_STORIES = True        # 1 Serper news search for stories whose article can't be read (~$0.001 each)
SHOW_NAME = "The Learning Kids"   # used in the intro/outro — change to your channel's name
HOST_NAME = ""                    # e.g. "Aarav"; leave empty to keep it generic

# Keyword rules applied in order; first match wins. Checked against
# title + URL path (lower-cased). Source-assigned buckets (RSS) take priority.
BUCKET_KEYWORDS = [
    ("Space & Science", r"\b(isro|nasa|spacex|space ?x|blue origin|rocket|astronaut|satellite|"
                        r"orbit|moon|lunar|mars|planet|meteor|asteroid|comet|galaxy|telescope|"
                        r"eclipse|solar|shenzhou|gaganyaan|chandrayaan|scientist|discover|species|fossil)\b"
                        r"|/sci-tech/science/|/sci-tech/energy-and-environment/"),
    ("Tech", r"\b(video games?|gaming|gamers?|playstation|ps5|xbox|nintendo|switch 2|ea fc|minecraft|roblox|fortnite|"
             r"gta|esports|ai|artificial intelligence|robot|robotic|chip|semiconductor|smartphone|iphone|"
             r"app|electric vehicle|ev|startup|google|apple|microsoft|openai|5g|6g)\b|/technology/"),
    ("Weather & Nature", r"\b(cyclones?|storms?|rains?|rainfall|monsoon|flood\w*|heatwaves?|heat wave|"
                         r"earthquakes?|quakes?|landslides?|wildfires?|tsunami|volcan\w*|imd|weather|dust storm|"
                         r"tigers?|elephants?|wildlife|hurricanes?|typhoons?|temperatures?|humidity|fog|climate|"
                         r"red alert|orange alert|yellow alert)\b"),
    ("Sports", r"\b(cricket|ipl|football|fifa|tennis|hockey|olympic|chess|champions league|"
               r"world cup|medal|tournament|match|final|psg|kabaddi|athletics|grand prix|f1|asian games|"
               r"coach|swim|swimmer|world record|wicket|badminton|olympiad|triathlon|marathon|"
               r"singapore open|french open|us open|wimbledon|gulf cup|premier league|la liga)\b|/sport/"),
    ("UAE", r"\b(uae|dubai|abu dhabi|sharjah|emirates|emirati|ajman|ras al khaimah|fujairah|umm al quwain|"
            r"global village|burj|expo city|rta|dirham|dh\d|al ain|mohammed bin rashid|mohamed bin zayed)\b"),
    ("World", r"\b(u\.?s\.?|america|trump|china|chinese|japan|russia|ukraine|iran|israel|gaza|lebanon|"
              r"pakistan|nepal|sri lanka|bangladesh|u\.?k\.?|britain|france|paris|germany|europe|"
              r"africa|congo|laos|vietnam|philippines|korea|venezuela|brazil|canada|australia|"
              r"pope|united nations|imf|nato)\b"),
    ("Business", r"/business/|\b(economy|gdp|inflation|rbi|sensex|nifty|trade|tariff|budget)\b"),
    ("World", r"/international/|/world/"),
]
DEFAULT_BUCKET = "India"

# ---------------------------------------------------------------------------
# Kid-safety: stories matching these are NOT deleted — they are flagged and
# hidden by default on the review screen, so you still decide.
# ---------------------------------------------------------------------------
SENSITIVE_PATTERN = (
    r"\b(murder|murdered|kill|killed|killing|hacks?|stabb|rape|raped|sexual|pocso|molest|"
    r"suicide|dies by|body found|bodies|corpse|hooch|lynch|assault|abduct|kidnap|"
    r"terror|militant|bomb|shot dead|gunman|massacre|beheaded|dowry|acid attack|"
    r"extramarital|affair|drug|narcotic|ganja)\b"
)

# Headlines shorter than this are section pages or ads, not stories
MIN_HEADLINE_WORDS = 5

# Dropped from every source: stock-market chatter, adverts, section pages
NOISE_PATTERN = (
    r"\b(share price|shares? (rise|fall|jump|slump)|stock (price|market)|strong buy|outperform|price target|"
    r"target price|dow jones|nasdaq|futures|ipo|sensex|nifty|analysts? (say|rate)|% off|deal of the day|"
    r"prime day|discount|sale ends|coupon|best broker|sponsored|top stories$|latest .* news)\b"
)

# Near-duplicate threshold (0–1) for titles from different sources
DEDUPE_SIMILARITY = 0.82

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
}
HTTP_TIMEOUT = 30
