#!/usr/bin/env python3
"""SYNTHETIC PIPELINE FIXTURE: 1,600 template-generated pairs. NOT training data.

Use only for: DPOTrainer schema/dry-run tests and verifier unit tests.
Do not train the study on this set or publish it as evidence (Ground Rules 1-2).

Same TRL DPOTrainer format and guardrails as the 40-pair set, generated from
parameterized templates so every prompt is unique (one pair per prompt).
Target mix per the study doc:
  25% wrong_param_value, 25% wrong_function_selection,
  15% missing_required_parameter, 15% spurious_tool_call,
  10% missed_tool_call, 5% hallucinated_parameter, 5% malformed_syntax.

Pairs that break the similarity floor (chosen/rejected length gap > 40%) are
filtered out during generation, mirroring the doc's hard-negative rule.
Deterministic: seeded RNG, same output on every run.

The structural checks below are a FIXTURE self-test: they prove the generated
pairs are well-formed against their own tool schemas. They are NOT the verifier
gate described in HANDOFF.md 2.2 ("0 false positives, 0 misses"), which needs
`mining/mine_pairs.py` and cannot be cited until that verifier actually runs.

Usage:
    python tests/fixtures/scale_pairs_fixed.py            # write next to this file
    python tests/fixtures/scale_pairs_fixed.py --out-dir DIR
    python tests/fixtures/scale_pairs_fixed.py --check    # regenerate + byte-compare
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

rng = random.Random(20260804)

TOTAL = 1600
ALLOC = {
    "wrong_param_value": 400,
    "wrong_function_selection": 400,
    "missing_required_parameter": 240,
    "spurious_tool_call": 240,
    "missed_tool_call": 160,
    "hallucinated_parameter": 80,
    "malformed_syntax": 80,
}
EVAL_PER_TYPE = {k: v // 10 for k, v in ALLOC.items()}  # stratified 10% held-out

OUT_NAMES = {
    "train": "fixture_pairs_train.jsonl",
    "eval": "fixture_pairs_eval.jsonl",
    "audit": "fixture_audit_sample_50.jsonl",
}
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------ helpers -
def sys_prompt(tools):
    return ("You have access to the following tools: " + json.dumps(tools)
            + " Use a tool only when needed.")

def call(name, **arguments):
    return json.dumps({"name": name, "arguments": arguments})

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

def ordinal(n):
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"

def nice_date(iso):
    y, m, d = (int(x) for x in iso.split("-"))
    return f"{MONTHS[m - 1]} {ordinal(d)}"

def nice_time(hhmm):
    h, m = (int(x) for x in hhmm.split(":"))
    ap = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12}{ap}" if m == 0 else f"{h12}:{m:02d}{ap}"

def weekday(iso):
    return datetime.date.fromisoformat(iso).strftime("%A")

# future dates only (today is 2026-08-04)
DATES = [f"2026-{m:02d}-{d:02d}" for m in range(8, 13) for d in range(2, 27, 3)]
DATES += [f"2027-{m:02d}-{d:02d}" for m in range(1, 6) for d in range(2, 27, 3)]
TIMES = ["07:30", "08:00", "09:00", "09:30", "10:00", "11:00", "11:30", "12:00",
         "13:00", "13:30", "14:00", "15:00", "15:30", "16:00", "17:00", "17:30",
         "18:00", "19:00", "19:30", "20:00"]

# ------------------------------------------------------------------ schemas -
STOCK = [{"name": "get_stock_price", "parameters": {"symbol": {"type": "string", "required": True}, "exchange": {"type": "string", "required": False}}}]
CALENDAR = [{"name": "create_calendar_event", "parameters": {"title": {"type": "string", "required": True}, "date": {"type": "string", "required": True}, "start_time": {"type": "string", "required": True}, "duration_minutes": {"type": "integer", "required": False}}}]
TIMER = [{"name": "set_timer", "parameters": {"duration_seconds": {"type": "integer", "required": True}, "label": {"type": "string", "required": False}}}]
CURRENCY = [{"name": "convert_currency", "parameters": {"amount": {"type": "number", "required": True}, "from_currency": {"type": "string", "required": True}, "to_currency": {"type": "string", "required": True}}}]
WEATHER = [{"name": "get_weather", "parameters": {"location": {"type": "string", "required": True}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "required": False}}}]
FLIGHT_SEARCH = [{"name": "search_flights", "parameters": {"origin": {"type": "string", "required": True}, "destination": {"type": "string", "required": True}, "date": {"type": "string", "required": True}}}]
TRANSLATE = [{"name": "translate_text", "parameters": {"text": {"type": "string", "required": True}, "target_language": {"type": "string", "required": True}, "source_language": {"type": "string", "required": False}}}]
HOTEL = [{"name": "book_hotel", "parameters": {"city": {"type": "string", "required": True}, "check_in": {"type": "string", "required": True}, "check_out": {"type": "string", "required": True}, "guests": {"type": "integer", "required": False}}}]
TIP = [{"name": "calculate_tip", "parameters": {"bill_amount": {"type": "number", "required": True}, "tip_percent": {"type": "number", "required": True}}}]
TRACKING = [{"name": "track_package", "parameters": {"tracking_number": {"type": "string", "required": True}, "carrier": {"type": "string", "required": False}}}]
DEFINITION = [{"name": "get_definition", "parameters": {"word": {"type": "string", "required": True}}}]
SPORTS = [{"name": "get_sports_scores", "parameters": {"league": {"type": "string", "required": True}, "date": {"type": "string", "required": False}}}]
DIRECTIONS = [{"name": "get_directions", "parameters": {"origin": {"type": "string", "required": True}, "destination": {"type": "string", "required": True}, "mode": {"type": "string", "enum": ["driving", "walking", "transit"], "required": False}}}]
EMAIL = [{"name": "send_email", "parameters": {"to": {"type": "string", "required": True}, "subject": {"type": "string", "required": True}, "body": {"type": "string", "required": True}}}]
MUSIC_PLAY = [{"name": "play_music", "parameters": {"track": {"type": "string", "required": True}, "artist": {"type": "string", "required": False}}}]
CLOCK = [{"name": "get_current_time", "parameters": {"location": {"type": "string", "required": True}}}]
RESERVE = [{"name": "make_reservation", "parameters": {"restaurant_name": {"type": "string", "required": True}, "date": {"type": "string", "required": True}, "time": {"type": "string", "required": True}, "party_size": {"type": "integer", "required": True}}}]

WEATHER2 = [{"name": "get_current_weather", "parameters": {"location": {"type": "string", "required": True}}},
            {"name": "get_weather_forecast", "parameters": {"location": {"type": "string", "required": True}, "days": {"type": "integer", "required": True}}}]
FLIGHTS2 = [FLIGHT_SEARCH[0],
            {"name": "book_flight", "parameters": {"origin": {"type": "string", "required": True}, "destination": {"type": "string", "required": True}, "date": {"type": "string", "required": True}}}]
RESTAURANT2 = [{"name": "search_restaurants", "parameters": {"query": {"type": "string", "required": True}, "location": {"type": "string", "required": False}}}, RESERVE[0]]
MUSIC2 = [MUSIC_PLAY[0],
          {"name": "get_song_info", "parameters": {"track": {"type": "string", "required": True}, "artist": {"type": "string", "required": False}}}]
BANK2 = [{"name": "get_account_balance", "parameters": {"account_type": {"type": "string", "enum": ["checking", "savings"], "required": True}}},
         {"name": "get_transaction_history", "parameters": {"account_type": {"type": "string", "enum": ["checking", "savings"], "required": True}, "days": {"type": "integer", "required": False}}}]
EVENT2 = [{"name": "create_event", "parameters": {"title": {"type": "string", "required": True}, "date": {"type": "string", "required": True}, "time": {"type": "string", "required": True}}},
          {"name": "create_reminder", "parameters": {"title": {"type": "string", "required": True}, "date": {"type": "string", "required": True}, "time": {"type": "string", "required": True}}}]
STOCK2 = [STOCK[0], {"name": "get_stock_news", "parameters": {"symbol": {"type": "string", "required": True}}}]
EMAIL2 = [EMAIL[0],
          {"name": "create_email_draft", "parameters": {"to": {"type": "string", "required": True}, "subject": {"type": "string", "required": True}, "body": {"type": "string", "required": True}}}]
TRAVEL2 = [DIRECTIONS[0],
           {"name": "get_travel_time", "parameters": {"origin": {"type": "string", "required": True}, "destination": {"type": "string", "required": True}, "mode": {"type": "string", "enum": ["driving", "walking", "transit"], "required": False}}}]
RIDE2 = [{"name": "get_ride_estimate", "parameters": {"pickup": {"type": "string", "required": True}, "dropoff": {"type": "string", "required": True}, "ride_type": {"type": "string", "required": False}}},
         {"name": "request_ride", "parameters": {"pickup": {"type": "string", "required": True}, "dropoff": {"type": "string", "required": True}, "ride_type": {"type": "string", "required": False}}}]
TODO2 = [{"name": "add_todo", "parameters": {"task": {"type": "string", "required": True}}},
         {"name": "complete_todo", "parameters": {"task": {"type": "string", "required": True}}}]
NEWS2 = [{"name": "get_news_headlines", "parameters": {"topic": {"type": "string", "required": True}}},
         {"name": "search_news_archive", "parameters": {"topic": {"type": "string", "required": True}, "year": {"type": "integer", "required": False}}}]

# -------------------------------------------------------------------- pools -
COMPANIES = [("Apple", "AAPL"), ("Microsoft", "MSFT"), ("Alphabet", "GOOGL"), ("Amazon", "AMZN"),
             ("Nvidia", "NVDA"), ("Meta", "META"), ("Tesla", "TSLA"), ("Netflix", "NFLX"),
             ("Adobe", "ADBE"), ("Salesforce", "CRM"), ("Oracle", "ORCL"), ("Intel", "INTC"),
             ("Advanced Micro Devices", "AMD"), ("Cisco", "CSCO"), ("Qualcomm", "QCOM"),
             ("PayPal", "PYPL"), ("Shopify", "SHOP"), ("Uber", "UBER"), ("Airbnb", "ABNB"),
             ("Palantir", "PLTR"), ("Snowflake", "SNOW"), ("Spotify", "SPOT"), ("Pinterest", "PINS"),
             ("Coinbase", "COIN"), ("Robinhood", "HOOD"), ("Ford", "F"), ("General Motors", "GM"),
             ("Boeing", "BA"), ("Caterpillar", "CAT"), ("Deere", "DE"), ("Honeywell", "HON"),
             ("Disney", "DIS"), ("Nike", "NKE"), ("Starbucks", "SBUX"), ("McDonald's", "MCD"),
             ("Coca-Cola", "KO"), ("PepsiCo", "PEP"), ("Pfizer", "PFE"), ("Moderna", "MRNA"),
             ("Walmart", "WMT"), ("Target", "TGT"), ("Costco", "COST"), ("Home Depot", "HD"),
             ("Lowe's", "LOW"), ("FedEx", "FDX"), ("Delta Air Lines", "DAL"), ("United Airlines", "UAL"),
             ("Southwest Airlines", "LUV"), ("Visa", "V"), ("Mastercard", "MA"), ("JPMorgan Chase", "JPM"),
             ("Goldman Sachs", "GS"), ("Morgan Stanley", "MS"), ("Bank of America", "BAC"),
             ("Wells Fargo", "WFC"), ("Chevron", "CVX"), ("ExxonMobil", "XOM"), ("Verizon", "VZ"),
             ("Comcast", "CMCSA"), ("Intuit", "INTU"), ("ServiceNow", "NOW"), ("Workday", "WDAY"),
             ("Datadog", "DDOG"), ("CrowdStrike", "CRWD"), ("MongoDB", "MDB"), ("Atlassian", "TEAM"),
             ("Roku", "ROKU"), ("Etsy", "ETSY"), ("eBay", "EBAY"), ("Chipotle", "CMG"),
             ("Lululemon", "LULU"), ("Rivian", "RIVN")]
SYMBOLS = [s for _, s in COMPANIES]

US_CITIES = ["Boston, MA", "Chicago, IL", "Denver, CO", "Seattle, WA", "Portland, OR", "Austin, TX",
             "Dallas, TX", "Houston, TX", "Phoenix, AZ", "Miami, FL", "Atlanta, GA", "Nashville, TN",
             "Memphis, TN", "New Orleans, LA", "Charlotte, NC", "Raleigh, NC", "Baltimore, MD",
             "Philadelphia, PA", "Pittsburgh, PA", "Cleveland, OH", "Columbus, OH", "Cincinnati, OH",
             "Detroit, MI", "Milwaukee, WI", "Minneapolis, MN", "St. Louis, MO", "Kansas City, MO",
             "Omaha, NE", "Tulsa, OK", "Albuquerque, NM", "Santa Fe, NM", "Salt Lake City, UT",
             "Boise, ID", "Las Vegas, NV", "Reno, NV", "Sacramento, CA", "San Diego, CA",
             "Anchorage, AK", "Honolulu, HI", "Tampa, FL", "Orlando, FL", "Jacksonville, FL",
             "Savannah, GA", "Louisville, KY", "Indianapolis, IN", "Des Moines, IA", "Madison, WI",
             "Boulder, CO", "Tucson, AZ", "El Paso, TX", "Spokane, WA", "Eugene, OR",
             "Providence, RI", "Hartford, CT", "Buffalo, NY", "Rochester, NY", "Burlington, VT"]
WORLD_CITIES = ["Oslo", "London", "Paris", "Berlin", "Madrid", "Rome", "Lisbon", "Dublin", "Vienna",
                "Prague", "Warsaw", "Stockholm", "Copenhagen", "Helsinki", "Athens", "Istanbul",
                "Cairo", "Nairobi", "Cape Town", "Lagos", "Dubai", "Doha", "Mumbai", "Delhi",
                "Singapore", "Bangkok", "Hanoi", "Jakarta", "Manila", "Tokyo", "Osaka", "Seoul",
                "Beijing", "Shanghai", "Taipei", "Sydney", "Melbourne", "Auckland", "Toronto",
                "Vancouver", "Montreal", "Mexico City", "Sao Paulo", "Buenos Aires", "Lima",
                "Santiago", "Bogota", "Reykjavik", "Zurich", "Geneva", "Brussels", "Munich",
                "Hamburg", "Barcelona", "Seville", "Porto", "Edinburgh", "Manchester"]
ALL_CITIES = US_CITIES + WORLD_CITIES

AIRPORTS = [("Boston Logan", "BOS"), ("New York JFK", "JFK"), ("LaGuardia", "LGA"), ("Newark", "EWR"),
            ("Chicago O'Hare", "ORD"), ("Chicago Midway", "MDW"), ("Denver International", "DEN"),
            ("Seattle-Tacoma", "SEA"), ("Portland International", "PDX"), ("Austin-Bergstrom", "AUS"),
            ("Dallas-Fort Worth", "DFW"), ("Houston Bush", "IAH"), ("Phoenix Sky Harbor", "PHX"),
            ("Miami International", "MIA"), ("Atlanta Hartsfield", "ATL"), ("Nashville International", "BNA"),
            ("Charlotte Douglas", "CLT"), ("Philadelphia International", "PHL"), ("Detroit Metro", "DTW"),
            ("Minneapolis-St. Paul", "MSP"), ("Salt Lake City International", "SLC"),
            ("Las Vegas Harry Reid", "LAS"), ("San Francisco International", "SFO"),
            ("Los Angeles International", "LAX"), ("San Diego International", "SAN"),
            ("Orlando International", "MCO"), ("Tampa International", "TPA"),
            ("Washington Dulles", "IAD"), ("Reagan National", "DCA"), ("Baltimore-Washington", "BWI")]

CURRENCY_WORDS = [("US dollars", "USD"), ("euros", "EUR"), ("British pounds", "GBP"),
                  ("Japanese yen", "JPY"), ("Swiss francs", "CHF"), ("Canadian dollars", "CAD"),
                  ("Australian dollars", "AUD"), ("Mexican pesos", "MXN"), ("Indian rupees", "INR"),
                  ("South Korean won", "KRW"), ("Chinese yuan", "CNY"), ("Swedish kronor", "SEK"),
                  ("Norwegian kroner", "NOK"), ("Danish kroner", "DKK"), ("Polish zloty", "PLN"),
                  ("Thai baht", "THB"), ("Malaysian ringgit", "MYR"), ("Singapore dollars", "SGD"),
                  ("South African rand", "ZAR"), ("Brazilian reais", "BRL")]

LANGUAGES = [("Spanish", "es"), ("French", "fr"), ("German", "de"), ("Italian", "it"),
             ("Portuguese", "pt"), ("Dutch", "nl"), ("Swedish", "sv"), ("Norwegian", "no"),
             ("Danish", "da"), ("Finnish", "fi"), ("Polish", "pl"), ("Czech", "cs"),
             ("Hungarian", "hu"), ("Greek", "el"), ("Turkish", "tr"), ("Russian", "ru"),
             ("Ukrainian", "uk"), ("Arabic", "ar"), ("Hebrew", "he"), ("Hindi", "hi"),
             ("Bengali", "bn"), ("Thai", "th"), ("Vietnamese", "vi"), ("Indonesian", "id"),
             ("Malay", "ms"), ("Japanese", "ja"), ("Korean", "ko"), ("Swahili", "sw"),
             ("Tagalog", "tl"), ("Mandarin Chinese", "zh")]
PHRASES = ["good morning", "thank you very much", "where is the train station",
           "how much does this cost", "see you tomorrow", "nice to meet you", "can you help me",
           "the weather is beautiful today", "I would like a coffee", "happy birthday",
           "excuse me", "what time is it", "have a safe trip", "the food was delicious",
           "my name is Alex"]

EVENT_TITLES = ["Dentist appointment", "Team standup", "Quarterly review", "Project kickoff",
                "Design review", "Sprint planning", "One on one with Sam", "Budget meeting",
                "Client call", "Product demo", "Interview with candidate", "Yoga class", "Haircut",
                "Car service pickup", "Parent teacher conference", "Book club", "Volunteer shift",
                "Piano lesson", "Vet appointment", "Oil change", "Board meeting", "Marketing sync",
                "Sales pipeline review", "Code review session", "Lunch with Riley",
                "Coffee with Morgan", "Study group", "Tax prep session", "Gym session",
                "Physical therapy"]
NAMES = ["sam", "riley", "morgan", "jordan", "avery", "casey", "quinn", "taylor", "alex", "jamie",
         "dana", "reese", "skyler", "rowan", "emerson", "hayden", "parker", "blake", "devon",
         "kendall"]
SUBJECTS = ["Schedule change", "Project update", "Quick question", "Invoice attached",
            "Lunch plans", "Trip details", "Draft feedback", "Meeting notes", "Follow up",
            "Budget approval"]
SAYINGS = ["the meeting moved to 3pm", "the report is ready for review", "I will be out on Friday",
           "the demo went well", "the invoice is attached", "we should push the deadline a week",
           "the venue is confirmed", "I reviewed the draft and left comments",
           "the shipment arrived today", "the budget was approved", "the flight got rebooked",
           "we should sync tomorrow morning"]

REST_A = ["Cedar", "Copper", "Willow", "Juniper", "Harbor", "Lantern", "Marble", "Fern", "Sable",
          "Golden", "Rustic", "Ember", "Saffron", "Ivy", "Birch"]
REST_B = ["and Vine", "Kettle", "Table", "Hearth", "and Thistle", "Grove Kitchen", "Street Bistro",
          "House", "and Salt", "Fork", "Larder", "and Stone", "Garden Cafe", "Pantry", "and Oak"]
RESTAURANTS = [f"{a} {b}" for a, b in itertools.product(REST_A, REST_B)]

TRACK_A = ["Paper", "Glass", "Silver", "Neon", "Quiet", "Velvet", "Amber", "Hollow", "Electric",
           "Winter", "Copper", "Midnight", "Plastic", "Gravel", "Lucid"]
TRACK_B = ["Lanterns", "Rivers", "Static", "Harvest", "Engines", "Thunder", "Tides", "Summits",
           "Meadow", "Satellites", "Skies", "Orchard", "Sunrise", "Hearts", "Avenues"]
TRACKS = [f"{a} {b}" for a, b in itertools.product(TRACK_A, TRACK_B)]
ARTISTS = ["Halloway", "The Morning Static", "June Parade", "Cobalt Fields", "The Paper Foxes",
           "Marlowe Drift", "Atlas Verde", "The Hollow Kites", "Nova Bloom", "Saint Signal",
           "The Glass Antlers", "Echo Motel", "Fenwick Sons", "The Velvet Larks", "Orbit Daughters"]

TASKS = ["buy groceries", "call the plumber", "renew the car registration", "water the plants",
         "schedule a dentist visit", "pick up dry cleaning", "pay the electric bill",
         "return library books", "back up the laptop", "clean the garage", "email the landlord",
         "order printer ink", "update the resume", "book a flu shot", "fix the leaky faucet",
         "wash the car", "mow the lawn", "plan the birthday party", "cancel the unused subscription",
         "take out the recycling", "refill the prescription", "charge the camera batteries",
         "rotate the tires", "descale the coffee maker", "organize the pantry",
         "replace the smoke detector battery", "mail the package", "draft the newsletter",
         "review the insurance policy", "clean the gutters"]
TOPICS = ["electric vehicles", "renewable energy", "artificial intelligence", "the housing market",
          "semiconductor manufacturing", "space exploration", "the bond market", "cybersecurity",
          "offshore wind", "quantum computing", "drought conditions", "the airline industry",
          "battery technology", "urban transit", "vaccine research", "the labor market",
          "streaming services", "robotics", "agriculture technology", "rare earth minerals",
          "the shipping industry", "wildfire season", "professional cycling", "the art market",
          "college athletics"]
LANDMARKS = ["Union Station", "Riverside Park", "Lakeview Terminal", "City Hall", "Grandview Mall",
             "Harbor Point", "Westgate Stadium", "Oakdale University", "Pine Street Market",
             "Summit Convention Center", "Maple Ferry Dock", "Northside Arena", "Central Library",
             "Fairview Hospital", "Kingsbury Airport"]
WORDS = ["ephemeral", "ubiquitous", "serendipity", "laconic", "quixotic", "pragmatic", "esoteric",
         "gregarious", "meticulous", "resilient", "candor", "austere", "prosaic", "vestigial",
         "halcyon", "sanguine", "taciturn", "zephyr", "lucid", "arcane"]
AIRLINES = ["Delta", "United", "American", "Southwest", "JetBlue", "Alaska", "Spirit", "Frontier"]
AMBIG_CITIES = [("Springfield", "Illinois", "IL"), ("Portland", "Maine", "ME"),
                ("Columbus", "Ohio", "OH"), ("Charleston", "South Carolina", "SC"),
                ("Aurora", "Colorado", "CO"), ("Richmond", "Virginia", "VA"),
                ("Jackson", "Mississippi", "MS"), ("Salem", "Oregon", "OR"),
                ("Albany", "New York", "NY"), ("Franklin", "Tennessee", "TN"),
                ("Georgetown", "Texas", "TX"), ("Bloomington", "Indiana", "IN"),
                ("Rochester", "Minnesota", "MN"), ("Athens", "Georgia", "GA"),
                ("Norman", "Oklahoma", "OK"), ("Bellevue", "Washington", "WA"),
                ("Arlington", "Texas", "TX"), ("Newport", "Rhode Island", "RI")]

# fact tables for spurious answers (chosen must be genuinely correct)
COUNTRY_CURRENCY = [("Japan", "Japanese yen", "JPY"), ("the United Kingdom", "British pound", "GBP"),
    ("Switzerland", "Swiss franc", "CHF"), ("India", "Indian rupee", "INR"),
    ("Mexico", "Mexican peso", "MXN"), ("Brazil", "Brazilian real", "BRL"),
    ("South Korea", "South Korean won", "KRW"), ("China", "Chinese yuan", "CNY"),
    ("Sweden", "Swedish krona", "SEK"), ("Norway", "Norwegian krone", "NOK"),
    ("Denmark", "Danish krone", "DKK"), ("Poland", "Polish zloty", "PLN"),
    ("Czechia", "Czech koruna", "CZK"), ("Hungary", "Hungarian forint", "HUF"),
    ("Turkey", "Turkish lira", "TRY"), ("Thailand", "Thai baht", "THB"),
    ("Vietnam", "Vietnamese dong", "VND"), ("Indonesia", "Indonesian rupiah", "IDR"),
    ("the Philippines", "Philippine peso", "PHP"), ("Malaysia", "Malaysian ringgit", "MYR"),
    ("Singapore", "Singapore dollar", "SGD"), ("Australia", "Australian dollar", "AUD"),
    ("New Zealand", "New Zealand dollar", "NZD"), ("Canada", "Canadian dollar", "CAD"),
    ("Egypt", "Egyptian pound", "EGP"), ("South Africa", "South African rand", "ZAR"),
    ("Nigeria", "Nigerian naira", "NGN"), ("Kenya", "Kenyan shilling", "KES"),
    ("Israel", "Israeli shekel", "ILS"), ("Saudi Arabia", "Saudi riyal", "SAR"),
    ("the United Arab Emirates", "UAE dirham", "AED"), ("Argentina", "Argentine peso", "ARS"),
    ("Chile", "Chilean peso", "CLP"), ("Colombia", "Colombian peso", "COP"),
    ("Peru", "Peruvian sol", "PEN"), ("Iceland", "Icelandic krona", "ISK"),
    ("Morocco", "Moroccan dirham", "MAD"), ("Bangladesh", "Bangladeshi taka", "BDT"),
    ("Pakistan", "Pakistani rupee", "PKR"), ("Costa Rica", "Costa Rican colon", "CRC")]
COUNTRY_CAPITAL = [("Japan", "Tokyo"), ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Greece", "Athens"),
    ("Poland", "Warsaw"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("Canada", "Ottawa"),
    ("Mexico", "Mexico City"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"),
    ("Peru", "Lima"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"),
    ("Morocco", "Rabat"), ("Turkey", "Ankara"), ("India", "New Delhi"), ("China", "Beijing"),
    ("South Korea", "Seoul"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"), ("the Philippines", "Manila"), ("Malaysia", "Kuala Lumpur"),
    ("Australia", "Canberra"), ("New Zealand", "Wellington"), ("Ethiopia", "Addis Ababa"),
    ("Ghana", "Accra"), ("Saudi Arabia", "Riyadh"), ("Jordan", "Amman")]
COUNTRY_LANGUAGE = [("Japan", "Japanese"), ("France", "French"), ("Germany", "German"),
    ("Italy", "Italian"), ("Spain", "Spanish"), ("Portugal", "Portuguese"),
    ("Brazil", "Portuguese"), ("Mexico", "Spanish"), ("Argentina", "Spanish"),
    ("South Korea", "Korean"), ("Thailand", "Thai"), ("Vietnam", "Vietnamese"),
    ("Indonesia", "Indonesian"), ("the Netherlands", "Dutch"), ("Greece", "Greek"),
    ("Poland", "Polish"), ("Russia", "Russian"), ("Turkey", "Turkish"), ("Egypt", "Arabic"),
    ("Saudi Arabia", "Arabic"), ("Iceland", "Icelandic"), ("Hungary", "Hungarian"),
    ("Czechia", "Czech"), ("Denmark", "Danish"), ("Norway", "Norwegian"), ("Sweden", "Swedish"),
    ("Finland", "Finnish"), ("Albania", "Albanian")]
ELEMENTS = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("copper", "Cu"), ("lead", "Pb"),
    ("tin", "Sn"), ("sodium", "Na"), ("potassium", "K"), ("oxygen", "O"), ("hydrogen", "H"),
    ("helium", "He"), ("carbon", "C"), ("nitrogen", "N"), ("calcium", "Ca"), ("zinc", "Zn"),
    ("nickel", "Ni"), ("mercury", "Hg"), ("aluminum", "Al"), ("silicon", "Si"), ("sulfur", "S"),
    ("chlorine", "Cl"), ("magnesium", "Mg"), ("phosphorus", "P"), ("tungsten", "W")]
CONTINENTS = [("Kenya", "Africa"), ("Brazil", "South America"), ("Japan", "Asia"),
    ("Germany", "Europe"), ("India", "Asia"), ("Peru", "South America"), ("Nigeria", "Africa"),
    ("Thailand", "Asia"), ("Poland", "Europe"), ("Chile", "South America"), ("Morocco", "Africa"),
    ("Vietnam", "Asia"), ("Portugal", "Europe"), ("Ethiopia", "Africa"),
    ("Argentina", "South America"), ("Mongolia", "Asia"), ("Norway", "Europe"),
    ("Ghana", "Africa"), ("Nepal", "Asia"), ("Ecuador", "South America")]
TEAM_SIZE = [("soccer", "eleven", "MLS"), ("basketball", "five", "NBA"), ("baseball", "nine", "MLB"),
    ("ice hockey", "six", "NHL"), ("American football", "eleven", "NFL"),
    ("volleyball", "six", "volleyball"), ("rugby union", "fifteen", "rugby"),
    ("cricket", "eleven", "cricket"), ("water polo", "seven", "water polo"),
    ("handball", "seven", "handball"), ("field hockey", "eleven", "field hockey"),
    ("futsal", "five", "futsal")]
GAME_LEN = [("an NBA game", "48 minutes of play, in four 12 minute quarters", "NBA"),
    ("an NFL game", "60 minutes of play, in four 15 minute quarters", "NFL"),
    ("an NHL game", "60 minutes of play, in three 20 minute periods", "NHL"),
    ("a professional soccer match", "90 minutes of play, in two 45 minute halves", "MLS"),
    ("a college basketball game", "40 minutes of play, in two 20 minute halves", "NCAA basketball")]
ACRONYMS = [("HTTP", "HyperText Transfer Protocol"), ("HTML", "HyperText Markup Language"),
    ("URL", "Uniform Resource Locator"), ("CPU", "Central Processing Unit"),
    ("GPU", "Graphics Processing Unit"), ("RAM", "Random Access Memory"),
    ("USB", "Universal Serial Bus"), ("PDF", "Portable Document Format"),
    ("SQL", "Structured Query Language"), ("API", "Application Programming Interface"),
    ("GPS", "Global Positioning System"), ("ATM", "Automated Teller Machine"),
    ("NASA", "National Aeronautics and Space Administration"),
    ("NATO", "North Atlantic Treaty Organization"), ("DNS", "Domain Name System"),
    ("VPN", "Virtual Private Network"), ("FAQ", "Frequently Asked Questions"),
    ("LED", "Light-Emitting Diode"), ("LCD", "Liquid Crystal Display"),
    ("SIM", "Subscriber Identity Module"), ("RADAR", "Radio Detection and Ranging"),
    ("SCUBA", "Self-Contained Underwater Breathing Apparatus"),
    ("LASER", "Light Amplification by Stimulated Emission of Radiation"),
    ("ASAP", "As Soon As Possible"), ("DIY", "Do It Yourself"),
    ("WHO", "World Health Organization"), ("FBI", "Federal Bureau of Investigation"),
    ("CIA", "Central Intelligence Agency")]
NO_TOOL_TAILS = ["no live lookup is needed for that", "that does not require a tool call",
                 "no tool call is needed here", "that is general knowledge rather than live data"]

# ------------------------------------------------------------ pair assembly -
GLOBAL_PROMPTS = set()
PAIRS = []

def floor_ok(chosen, rejected):
    lc, lr = len(chosen), len(rejected)
    return abs(lc - lr) / max(lc, lr) <= 0.40

def emit(template_id, error_type, combos, build, need):
    combos = list(combos)
    rng.shuffle(combos)
    made = 0
    for c in combos:
        built = build(c)
        if built is None:
            continue
        tools, user, chosen, rejected = built
        if user in GLOBAL_PROMPTS or not floor_ok(chosen, rejected):
            continue
        GLOBAL_PROMPTS.add(user)
        PAIRS.append({"error_type": error_type, "template": template_id, "tools": tools,
                      "user": user, "chosen": chosen, "rejected": rejected})
        made += 1
        if made == need:
            return
    raise SystemExit(f"capacity shortfall in {template_id}: {made}/{need}")

def tail():
    return rng.choice(NO_TOOL_TAILS)

# ===================================================== wrong_param_value 400
emit("wpv_ticker", "wrong_param_value",
     itertools.product(COMPANIES, range(5)), lambda c: (
        STOCK,
        ["How is {n} doing on the market today?", "Pull up the {n} share price for me.",
         "Check the stock price for {n}.", "What did {n} shares open at today?",
         "Give me a quote on {n} stock."][c[1]].format(n=c[0][0]),
        call("get_stock_price", symbol=c[0][1]),
        call("get_stock_price", symbol=c[0][0])), 70)

def b_caldate(c):
    title, d, t, style = c
    corrupt = {0: f"{int(d[5:7]):02d}/{int(d[8:10]):02d}/{d[:4]}",
               1: f"{MONTHS[int(d[5:7]) - 1][:3]} {int(d[8:10])}",
               2: f"{d[8:10]}-{d[5:7]}-{d[:4]}"}[style]
    user = (f"Put {title.lower() if title[0].isupper() and ' ' in title else title} on my calendar "
            f"for {nice_date(d)} at {nice_time(t)}. It should take about 45 minutes.")
    good = call("create_calendar_event", title=title, date=d, start_time=t, duration_minutes=45)
    bad = call("create_calendar_event", title=title, date=corrupt, start_time=t, duration_minutes=45)
    return CALENDAR, user, good, bad
emit("wpv_caldate", "wrong_param_value",
     itertools.product(EVENT_TITLES, rng.sample(DATES, 20), TIMES[:6], range(3)), b_caldate, 55)

emit("wpv_timer", "wrong_param_value",
     itertools.product([3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 90],
                       ["pasta", "rice", "tea", "laundry", "bread dough", "roast", "eggs",
                        "stretching", "reading break", "focus block"]), lambda c: (
        TIMER,
        f"Set a timer for {c[0]} minutes for the {c[1]}.",
        call("set_timer", duration_seconds=c[0] * 60, label=c[1]),
        call("set_timer", duration_seconds=c[0], label=c[1])), 40)

def b_curword(c):
    (fw, fc), (tw, tc), amt = c
    if fc == tc:
        return None
    user = f"How much is {amt} {fw} in {tw}?"
    return (CURRENCY, user,
            call("convert_currency", amount=amt, from_currency=fc, to_currency=tc),
            call("convert_currency", amount=amt, from_currency=fw, to_currency=tc))
emit("wpv_currencyword", "wrong_param_value",
     itertools.product(CURRENCY_WORDS, CURRENCY_WORDS, [40, 75, 120, 150, 240, 300, 480, 650]),
     b_curword, 45)

def b_unit(c):
    city, style = c
    label, good, bad = [("Celsius", "celsius", "Celsius"), ("Fahrenheit", "fahrenheit", "Fahrenheit"),
                        ("Celsius", "celsius", "C"), ("Fahrenheit", "fahrenheit", "F"),
                        ("Celsius", "celsius", "metric")][style]
    user = f"What's the temperature in {city} right now, in {label}?"
    return (WEATHER, user, call("get_weather", location=city, units=good),
            call("get_weather", location=city, units=bad))
emit("wpv_unitenum", "wrong_param_value", itertools.product(ALL_CITIES, range(5)), b_unit, 50)

def b_airport(c):
    (na, ca), (nb, cb), d = c
    if ca == cb:
        return None
    user = f"Find flights from {na} to {nb} on {nice_date(d)}."
    return (FLIGHT_SEARCH, user,
            call("search_flights", origin=ca, destination=cb, date=d),
            call("search_flights", origin=ca, destination=nb, date=d))
emit("wpv_airport", "wrong_param_value",
     itertools.product(AIRPORTS, AIRPORTS, rng.sample(DATES, 8)), b_airport, 50)

emit("wpv_language", "wrong_param_value", itertools.product(LANGUAGES, PHRASES), lambda c: (
        TRANSLATE,
        f"Translate '{c[1]}' into {c[0][0]}.",
        call("translate_text", text=c[1], target_language=c[0][1]),
        call("translate_text", text=c[1], target_language=c[0][0])), 30)

def b_hoteldate(c):
    city, d, mode = c
    dt = datetime.date.fromisoformat(d)
    d2 = (dt + datetime.timedelta(days=2)).isoformat()
    bad_in = d2 if mode == 0 else f"{int(d[:4]) + 1}{d[4:]}"
    user = (f"Book a hotel in {city.split(',')[0]}, checking in {nice_date(d)} and out "
            f"{nice_date(d2)}, for two guests.")
    return (HOTEL, user,
            call("book_hotel", city=city.split(",")[0], check_in=d, check_out=d2, guests=2),
            call("book_hotel", city=city.split(",")[0], check_in=bad_in, check_out=d2, guests=2))
emit("wpv_hoteldate", "wrong_param_value",
     itertools.product(US_CITIES, rng.sample(DATES, 10), range(2)), b_hoteldate, 25)

emit("wpv_tip", "wrong_param_value",
     itertools.product([10, 12, 15, 18, 20, 22, 25], [36, 42, 55, 64, 72, 88, 95, 110, 128, 150]),
     lambda c: (TIP,
                f"What's a {c[0]} percent tip on a {c[1]} dollar bill?",
                call("calculate_tip", bill_amount=c[1], tip_percent=c[0]),
                call("calculate_tip", bill_amount=c[1], tip_percent=round(c[0] / 100, 2))), 15)

def b_track(c):
    i, carrier = c
    digits = "".join(str(rng.randint(0, 9)) for _ in range(10))
    tn = ("1Z999AA1" + digits) if carrier == "UPS" else ("94" + digits)
    bad = tn[:-1] + str((int(tn[-1]) + 1) % 10)
    user = f"Where is my package? The tracking number is {tn} and it shipped {carrier}."
    return (TRACKING, user, call("track_package", tracking_number=tn, carrier=carrier),
            call("track_package", tracking_number=bad, carrier=carrier))
emit("wpv_tracking", "wrong_param_value",
     itertools.product(range(40), ["UPS", "USPS", "FedEx"]), b_track, 20)

# ============================================== wrong_function_selection 400
emit("wfs_weather", "wrong_function_selection", itertools.product(ALL_CITIES, range(3)), lambda c: (
        WEATHER2,
        ["Will it rain in {x} this weekend?", "What's the forecast for {x} over the next 3 days?",
         "Is it going to rain in {x} later this week?"][c[1]].format(x=c[0]),
        call("get_weather_forecast", location=c[0], days=[5, 3, 4][c[1]]),
        call("get_current_weather", location=c[0])), 45)

def b_flights2(c):
    (na, ca), (nb, cb), d, ph = c
    if ca == cb:
        return None
    user = ["What are my options for flying from {a} to {b} on {d}?",
            "Can you look up flights from {a} to {b} for {d}?",
            "Show me flights going from {a} to {b} on {d}."][ph].format(a=ca, b=cb, d=nice_date(d))
    return (FLIGHTS2, user, call("search_flights", origin=ca, destination=cb, date=d),
            call("book_flight", origin=ca, destination=cb, date=d))
emit("wfs_flights", "wrong_function_selection",
     itertools.product(AIRPORTS, AIRPORTS, rng.sample(DATES, 6), range(3)), b_flights2, 45)

def b_rest2(c):
    name, d, t, size, word = c
    user = f"Book us a table for {word} at {name} this {weekday(d)} at {nice_time(t)}."
    return (RESTAURANT2, user,
            call("make_reservation", restaurant_name=name, date=d, time=t, party_size=size),
            call("search_restaurants", query=f"{name} table for {word} {weekday(d)} {nice_time(t)}"))
emit("wfs_restaurant", "wrong_function_selection",
     ((n, d, t, s, w) for n, d, t in itertools.product(
         rng.sample(RESTAURANTS, 40), rng.sample(DATES, 6), ["18:00", "19:00", "19:30", "20:00"])
      for s, w in [(2, "two"), (4, "four"), (6, "six")]), b_rest2, 40)

emit("wfs_music", "wrong_function_selection",
     itertools.product(rng.sample(TRACKS, 60), ARTISTS), lambda c: (
        MUSIC2, f"Play '{c[0]}' by {c[1]}.",
        call("play_music", track=c[0], artist=c[1]),
        call("get_song_info", track=c[0], artist=c[1])), 40)

emit("wfs_bank", "wrong_function_selection",
     itertools.product(["checking", "savings"], range(6)), lambda c: (
        BANK2,
        ["How much money is in my {t} account right now?", "What's my {t} account balance?",
         "Check the balance on my {t} account.", "How much do I have sitting in {t}?",
         "What's the current balance in my {t} account?",
         "Tell me my {t} balance, please."][c[1]].format(t=c[0]),
        call("get_account_balance", account_type=c[0]),
        call("get_transaction_history", account_type=c[0])), 12)

emit("wfs_event", "wrong_function_selection",
     itertools.product(EVENT_TITLES, rng.sample(DATES, 8), TIMES[:8]), lambda c: (
        EVENT2,
        f"Put {c[0].lower()} on the calendar for {nice_date(c[1])} at {nice_time(c[2])}.",
        call("create_event", title=c[0], date=c[1], time=c[2]),
        call("create_reminder", title=c[0], date=c[1], time=c[2])), 40)

emit("wfs_stocknews", "wrong_function_selection", itertools.product(SYMBOLS, range(2)), lambda c: (
        STOCK2,
        ["What's the latest news on {s}?", "Any recent headlines for {s}?"][c[1]].format(s=c[0]),
        call("get_stock_news", symbol=c[0]),
        call("get_stock_price", symbol=c[0])), 40)

def b_email2(c):
    name, subj, say = c
    addr = f"{name}@example.com"
    body = say[0].upper() + say[1:] + "."
    user = f"Send an email to {addr} with the subject '{subj}' saying {say}."
    return (EMAIL2, user, call("send_email", to=addr, subject=subj, body=body),
            call("create_email_draft", to=addr, subject=subj, body=body))
emit("wfs_email", "wrong_function_selection",
     itertools.product(NAMES, SUBJECTS, SAYINGS), b_email2, 30)

def b_travel2(c):
    a, b, mode = c
    if a == b:
        return None
    verb = {"driving": "drive", "walking": "walk", "transit": "take transit"}[mode]
    user = f"How long would it take to {verb} from {a} to {b}?"
    return (TRAVEL2, user,
            call("get_travel_time", origin=a, destination=b, mode=mode),
            call("get_directions", origin=a, destination=b, mode=mode))
emit("wfs_travel", "wrong_function_selection",
     itertools.product(LANDMARKS, LANDMARKS, ["driving", "walking", "transit"]), b_travel2, 40)

def b_ride2(c):
    a, b = c
    if a == b:
        return None
    user = f"Roughly how much would a ride from {a} to {b} cost?"
    return (RIDE2, user, call("get_ride_estimate", pickup=a, dropoff=b),
            call("request_ride", pickup=a, dropoff=b))
emit("wfs_ride", "wrong_function_selection", itertools.product(LANDMARKS, LANDMARKS), b_ride2, 40)

emit("wfs_todo", "wrong_function_selection", itertools.product(TASKS, range(2)), lambda c: (
        TODO2,
        ["Add '{t}' to my todo list.", "Put '{t}' on my list for this week."][c[1]].format(t=c[0]),
        call("add_todo", task=c[0]), call("complete_todo", task=c[0])), 28)

# ============================================ missing_required_parameter 240
emit("mrp_caldate", "missing_required_parameter",
     itertools.product(EVENT_TITLES, rng.sample(DATES, 12), TIMES[8:14]), lambda c: (
        CALENDAR,
        f"Schedule '{c[0]}' for {nice_date(c[1])} at {nice_time(c[2])}. Block off 45 minutes.",
        call("create_calendar_event", title=c[0], date=c[1], start_time=c[2], duration_minutes=45),
        call("create_calendar_event", title=c[0], start_time=c[2], duration_minutes=45)), 40)

def b_mrp_cur(c):
    (fw, fc), (tw, tc), amt = c
    if fc == tc:
        return None
    return (CURRENCY, f"Convert {amt} {fw} into {tw} for me.",
            call("convert_currency", amount=amt, from_currency=fc, to_currency=tc),
            call("convert_currency", amount=amt, from_currency=fc))
emit("mrp_currency", "missing_required_parameter",
     itertools.product(CURRENCY_WORDS, CURRENCY_WORDS, [25, 60, 75, 90, 200, 350, 500]),
     b_mrp_cur, 30)

def b_mrp_email(c):
    name, subj, say = c
    addr = f"{name}@example.com"
    body = say[0].upper() + say[1:] + "."
    user = f"Email {addr} with the subject '{subj}' and tell them {say}."
    return (EMAIL, user, call("send_email", to=addr, subject=subj, body=body),
            call("send_email", to=addr, body=body))
emit("mrp_email", "missing_required_parameter",
     itertools.product(NAMES, SUBJECTS, SAYINGS), b_mrp_email, 30)

def b_mrp_dir(c):
    a, b, mode = c
    if a == b:
        return None
    word = {"driving": "driving", "walking": "walking", "transit": "transit"}[mode]
    user = f"Give me {word} directions from {a} to {b}."
    return (DIRECTIONS, user,
            call("get_directions", origin=a, destination=b, mode=mode),
            call("get_directions", destination=b, mode=mode))
emit("mrp_directions", "missing_required_parameter",
     itertools.product(LANDMARKS, LANDMARKS, ["driving", "walking", "transit"]), b_mrp_dir, 30)

def b_mrp_hotel(c):
    city, d = c
    dt = datetime.date.fromisoformat(d)
    d2 = (dt + datetime.timedelta(days=2)).isoformat()
    user = f"Get me a hotel room in {city.split(',')[0]} from {nice_date(d)} to {nice_date(d2)}."
    return (HOTEL, user, call("book_hotel", city=city.split(",")[0], check_in=d, check_out=d2),
            call("book_hotel", city=city.split(",")[0], check_in=d))
emit("mrp_hotel", "missing_required_parameter",
     itertools.product(US_CITIES, rng.sample(DATES, 6)), b_mrp_hotel, 30)

emit("mrp_translate", "missing_required_parameter",
     itertools.product(LANGUAGES, PHRASES), lambda c: (
        TRANSLATE, f"How do you say '{c[1]}' in {c[0][0]}?",
        call("translate_text", text=c[1], target_language=c[0][1]),
        call("translate_text", text=c[1])), 25)

def b_mrp_flight(c):
    (na, ca), (nb, cb), d = c
    if ca == cb:
        return None
    user = f"Find a flight from {ca} to {cb} leaving on {nice_date(d)}."
    return (FLIGHT_SEARCH, user, call("search_flights", origin=ca, destination=cb, date=d),
            call("search_flights", origin=ca, destination=cb))
emit("mrp_flightdate", "missing_required_parameter",
     itertools.product(AIRPORTS, AIRPORTS, rng.sample(DATES, 5)), b_mrp_flight, 30)

def b_mrp_rest(c):
    name, d, t, size, word = c
    user = f"Book a table for {word} at {name} on {nice_date(d)} at {nice_time(t)}."
    return (RESERVE, user,
            call("make_reservation", restaurant_name=name, date=d, time=t, party_size=size),
            call("make_reservation", restaurant_name=name, date=d, time=t))
emit("mrp_reservation", "missing_required_parameter",
     ((n, d, t, s, w) for n, d, t in itertools.product(
         rng.sample(RESTAURANTS, 30), rng.sample(DATES, 5), ["18:30", "19:00", "20:00"])
      for s, w in [(2, "two"), (3, "three"), (5, "five")]), b_mrp_rest, 25)

# ==================================================== spurious_tool_call 240
emit("stc_currency", "spurious_tool_call", itertools.product(COUNTRY_CURRENCY, range(2)), lambda c: (
        CURRENCY,
        ["What currency does {x} use?", "What currency is used in {x}?"][c[1]].format(x=c[0][0]),
        f"{c[0][0][0].upper() + c[0][0][1:]} uses the {c[0][1]}, which trades under the code "
        f"{c[0][2]} in currency markets.",
        call("convert_currency", amount=1, from_currency="USD", to_currency=c[0][2])), 40)

emit("stc_capital", "spurious_tool_call", itertools.product(COUNTRY_CAPITAL, range(2)), lambda c: (
        WEATHER,
        ["What's the capital of {x}?", "Which city is the capital of {x}?"][c[1]].format(x=c[0][0]),
        f"The capital of {c[0][0]} is {c[0][1]}; {tail()}.",
        call("get_weather", location=c[0][0], units="celsius")), 36)

emit("stc_language", "spurious_tool_call", COUNTRY_LANGUAGE, lambda c: (
        TRANSLATE,
        f"What language do they speak in {c[0]}?",
        f"The official language of {c[0]} is {c[1]}; {tail()}.",
        call("translate_text", text=f"What language do they speak in {c[0]}?",
             target_language="en")), 26)

emit("stc_ticker", "spurious_tool_call", itertools.product(COMPANIES, range(2)), lambda c: (
        STOCK,
        ["What company does the ticker {s} belong to?",
         "Which company trades under the ticker {s}?"][c[1]].format(s=c[0][1]),
        f"{c[0][1]} is the ticker symbol for {c[0][0]}; {tail()}.",
        call("get_stock_price", symbol=c[0][1])), 34)

emit("stc_element", "spurious_tool_call", ELEMENTS, lambda c: (
        DEFINITION,
        f"What is the chemical symbol for {c[0]}?",
        f"The chemical symbol for {c[0]} is {c[1]}; {tail()}.",
        call("get_definition", word=f"chemical symbol for {c[0]}")), 22)

emit("stc_continent", "spurious_tool_call", CONTINENTS, lambda c: (
        WEATHER,
        f"What continent is {c[0]} located in?",
        f"{c[0]} is located in {c[1]}; {tail()}.",
        call("get_weather", location=c[0], units="celsius")), 20)

emit("stc_teamsize", "spurious_tool_call", TEAM_SIZE, lambda c: (
        SPORTS,
        f"How many players does a {c[0]} team field per side?",
        f"A {c[0]} team fields {c[1]} players per side; {tail()}.",
        call("get_sports_scores", league=c[2], date="2026-08-04")), 12)

emit("stc_gamelen", "spurious_tool_call", GAME_LEN, lambda c: (
        SPORTS,
        f"How long is {c[0]}, in regulation?",
        f"In regulation, {c[0]} has {c[1]}.",
        call("get_sports_scores", league=c[2], date="2026-08-04")), 5)

emit("stc_acronym", "spurious_tool_call", ACRONYMS, lambda c: (
        STOCK,
        f"What does {c[0]} stand for?",
        f"{c[0]} stands for {c[1]}; {tail()}.",
        call("get_stock_price", symbol=c[0], exchange="NYSE")), 28)

MISC_SPURIOUS = [
    (CURRENCY, "What's the difference between a currency's code and its symbol?",
     "The code, like USD, is the three letter ISO identifier, while the symbol is the written "
     "sign used in prices; no live lookup is needed for that.",
     call("convert_currency", amount=1, from_currency="USD", to_currency="EUR")),
    (STOCK, "What's the difference between a stock ticker and a company name?",
     "A ticker is the short symbol an exchange assigns to a listing, while the company name is "
     "the business itself.",
     call("get_stock_price", symbol="AAPL", exchange="NASDAQ")),
    (WEATHER, "Is Boston usually snowy in January?",
     "Yes. January is typically Boston's snowiest month, with freezing temperatures and multiple "
     "storms in most years.",
     call("get_weather", location="Boston, MA", units="fahrenheit")),
    (TRACKING, "How long does USPS ground shipping usually take?",
     "USPS Ground Advantage usually delivers in two to five business days within the continental "
     "United States.",
     call("track_package", tracking_number="unknown", carrier="USPS")),
    (TIP, "Is 15 percent still considered a normal tip in the US?",
     "It is on the low side now. For table service in the US, eighteen to twenty percent has "
     "become the usual baseline.",
     call("calculate_tip", bill_amount=100, tip_percent=15)),
    (SPORTS, "How many minutes long is a regulation WNBA game?",
     "A regulation WNBA game has forty minutes of playing time, divided into four ten minute "
     "quarters.",
     call("get_sports_scores", league="WNBA", date="2026-08-04")),
    (WEATHER, "Why is the sky blue during the day?",
     "Air molecules scatter short blue wavelengths of sunlight far more than red, so scattered "
     "blue light fills the daytime sky.",
     call("get_weather", location="the sky", units="celsius")),
    (TIP, "What does gratuity mean on a restaurant bill?",
     "Gratuity is the tip, an extra amount added on top of the bill for service, sometimes "
     "included automatically for large parties.",
     call("calculate_tip", bill_amount=100, tip_percent=18)),
    (CURRENCY, "Why do exchange rates change from day to day?",
     "Rates float with supply and demand for each currency, which shifts with trade flows, "
     "interest rates, and market sentiment.",
     call("convert_currency", amount=100, from_currency="USD", to_currency="EUR")),
    (TIMER, "How long should I boil eggs for a hard boiled result?",
     "About nine to twelve minutes at a gentle boil gives a fully set yolk; start timing once "
     "the water is boiling.",
     call("set_timer", duration_seconds=600, label="eggs")),
    (STOCK, "What does it mean when a stock pays a dividend?",
     "A dividend is a portion of company profits paid out to shareholders, usually on a "
     "quarterly schedule.",
     call("get_stock_price", symbol="KO", exchange="NYSE")),
    (CLOCK, "Why do we have time zones at all?",
     "Time zones keep clocks roughly aligned with the sun as the Earth turns, so noon lands "
     "near midday everywhere.",
     call("get_current_time", location="Greenwich")),
    (DIRECTIONS, "What's the difference between a freeway and a highway?",
     "A freeway is a highway with fully controlled access and no cross traffic, while highway "
     "is the broader term for major roads.",
     call("get_directions", origin="here", destination="the freeway", mode="driving")),
    (HOTEL, "What does a hotel's check-in time usually mean?",
     "It is the earliest hour a room is guaranteed to be ready, commonly mid afternoon; "
     "arriving earlier may mean waiting or a fee.",
     call("book_hotel", city="Boston", check_in="2026-08-04", check_out="2026-08-05")),
    (BANK2, "What's the difference between a checking and a savings account?",
     "Checking is built for everyday spending, while savings pays interest and is meant for "
     "money you set aside.",
     call("get_account_balance", account_type="checking")),
    (SPORTS, "How many strikes make an out in baseball?",
     "Three strikes make an out in baseball; that is general knowledge rather than live data.",
     call("get_sports_scores", league="MLB", date="2026-08-04")),
    (TIMER, "What is the Pomodoro technique, in short?",
     "It is a focus method: work in twenty five minute blocks separated by short breaks, with "
     "a longer break every four blocks.",
     call("set_timer", duration_seconds=1500, label="pomodoro")),
]
emit("stc_misc", "spurious_tool_call", MISC_SPURIOUS, lambda c: c, 17)

# ====================================================== missed_tool_call 160
CONDS = ["sunny", "overcast", "drizzly", "windy", "partly cloudy", "hazy"]
WINDS = ["a light breeze", "calm air", "gusty winds", "a steady wind"]
def b_mtc_weather(c):
    city, unit = c
    user = f"What's the weather like in {city} right now?"
    t = rng.randint(41, 96)
    bad = f"It is {rng.choice(CONDS)} and about {t} degrees in {city} right now, with {rng.choice(WINDS)}."
    return (WEATHER, user, call("get_weather", location=city, units=unit), bad)
emit("mtc_weather", "missed_tool_call",
     itertools.product(ALL_CITIES, ["fahrenheit", "celsius"]), b_mtc_weather, 45)

def b_mtc_stock(c):
    name, sym = c[0]
    user = ["What is {n} trading at right now?", "Where is {n} stock right now?"][c[1]].format(n=name)
    bad = f"{name} is trading at about {rng.randint(14, 640)} dollars per share at the moment."
    return (STOCK, user, call("get_stock_price", symbol=sym), bad)
emit("mtc_stock", "missed_tool_call", itertools.product(COMPANIES, range(2)), b_mtc_stock, 35)

def b_mtc_fx(c):
    (fw, fc), (tw, tc), amt = c
    if fc == tc:
        return None
    r = round(rng.uniform(0.55, 1.65), 2)
    user = f"Convert {amt} {fw} to {tw} at today's rate."
    bad = (f"At the current rate of roughly {r}, {amt} {fw} comes to about "
           f"{round(amt * r)} {tw}.")
    return (CURRENCY, user,
            call("convert_currency", amount=amt, from_currency=fc, to_currency=tc), bad)
emit("mtc_fx", "missed_tool_call",
     itertools.product(CURRENCY_WORDS, CURRENCY_WORDS, [50, 80, 125, 250, 400, 600]), b_mtc_fx, 30)

emit("mtc_timer", "missed_tool_call",
     [(n,) for n in [3, 4, 6, 7, 9, 11, 14, 16, 17, 19, 21, 22, 24, 26, 28, 32, 38, 42]],
     lambda c: (TIMER, f"Set a timer for {c[0]} minutes.",
                call("set_timer", duration_seconds=c[0] * 60),
                f"Done. Your {c[0]} minute timer is running and will go off at the mark."), 15)

def b_mtc_time(c):
    city = c
    h, m = rng.randint(1, 12), rng.choice([5, 10, 20, 25, 40, 50])
    tod = rng.choice(["morning", "afternoon", "evening"])
    user = f"What time is it in {city} at the moment?"
    bad = f"It is currently about {h}:{m:02d} in the {tod} over in {city}."
    return (CLOCK, user, call("get_current_time", location=city), bad)
emit("mtc_time", "missed_tool_call", list(WORLD_CITIES), b_mtc_time, 20)

def b_mtc_track(c):
    i, carrier = c
    digits = "".join(str(rng.randint(0, 9)) for _ in range(10))
    tn = ("1Z777BB2" + digits) if carrier == "UPS" else ("92" + digits)
    hub = rng.choice(["Louisville", "Memphis", "Indianapolis", "Ontario", "Rockford"])
    user = f"Can you track package {tn}? It's with {carrier}."
    bad = f"Your package cleared the {hub} facility this morning and is now out for delivery."
    return (TRACKING, user, call("track_package", tracking_number=tn, carrier=carrier), bad)
emit("mtc_track", "missed_tool_call",
     itertools.product(range(20), ["UPS", "FedEx"]), b_mtc_track, 15)

# ================================================== hallucinated_parameter 80
def b_hal_air(c):
    airline, (na, ca), (nb, cb), d = c
    if ca == cb:
        return None
    user = f"Find {airline} flights from {ca} to {cb} on {nice_date(d)}."
    return (FLIGHT_SEARCH, user,
            call("search_flights", origin=ca, destination=cb, date=d),
            call("search_flights", origin=ca, destination=cb, date=d, airline=airline))
emit("hal_airline", "hallucinated_parameter",
     itertools.product(AIRLINES, AIRPORTS[:12], AIRPORTS[:12], rng.sample(DATES, 4)), b_hal_air, 25)

emit("hal_state", "hallucinated_parameter", AMBIG_CITIES, lambda c: (
        WEATHER,
        f"What's the weather in {c[0]}, {c[1]}? Fahrenheit please.",
        call("get_weather", location=f"{c[0]}, {c[2]}", units="fahrenheit"),
        call("get_weather", location=c[0], state=c[2], units="fahrenheit")), 15)

def b_hal_room(c):
    rt, city, d = c
    dt = datetime.date.fromisoformat(d)
    d2 = (dt + datetime.timedelta(days=1)).isoformat()
    user = f"Get me a {rt} room in {city.split(',')[0]}, {nice_date(d)} to {nice_date(d2)}."
    return (HOTEL, user,
            call("book_hotel", city=city.split(",")[0], check_in=d, check_out=d2),
            call("book_hotel", city=city.split(",")[0], check_in=d, check_out=d2, room_type=rt))
emit("hal_room", "hallucinated_parameter",
     itertools.product(["king", "queen", "double", "non-smoking", "accessible"],
                       US_CITIES[:20], rng.sample(DATES, 4)), b_hal_room, 15)

def b_hal_loc(c):
    title, loc, d, t = c
    user = f"Schedule '{title}' at {loc} on {nice_date(d)} at {nice_time(t)}."
    return (CALENDAR, user,
            call("create_calendar_event", title=title, date=d, start_time=t),
            call("create_calendar_event", title=title, date=d, start_time=t, location=loc))
emit("hal_location", "hallucinated_parameter",
     itertools.product(EVENT_TITLES[:12],
                       ["headquarters", "the downtown office", "Conference Room B", "the annex",
                        "the client site"], rng.sample(DATES, 4), TIMES[2:6]), b_hal_loc, 15)

emit("hal_formality", "hallucinated_parameter",
     itertools.product(LANGUAGES[:12], PHRASES[:8]), lambda c: (
        TRANSLATE,
        f"Translate '{c[1]}' into {c[0][0]}, and keep it polite.",
        call("translate_text", text=c[1], target_language=c[0][1]),
        call("translate_text", text=c[1], target_language=c[0][1], formality="polite")), 10)

# ========================================================= malformed_syntax 80
def corrupt(good, mode, key_value=None):
    if mode == 0:                                  # truncated braces
        return good[:-2]
    if mode == 1:                                  # python style single quotes
        return good.replace('"', "'")
    if mode == 2:                                  # trailing comma inside arguments
        return good[:-2] + ",}}"
    if mode == 3 and key_value:                    # unquoted string value
        return good.replace(f'"{key_value}"', key_value, 1)
    return good.replace('": "', '" "', 1)          # missing colon

def b_mal(c):
    kind, item, mode = c
    if kind == "def":
        good = call("get_definition", word=item)
        user = f"Look up the definition of '{item}' for me."
        kv = item
        tools = DEFINITION
    elif kind == "music":
        t, a = item
        good = call("play_music", track=t, artist=a)
        user = f"Start playing '{t}' by {a}."
        kv = a
        tools = MUSIC_PLAY
    elif kind == "weather":
        good = call("get_weather", location=item, units="celsius")
        user = f"Pull the current weather for {item}."
        kv = "celsius"
        tools = WEATHER
    elif kind == "stock":
        good = call("get_stock_price", symbol=item)
        user = f"Get a stock quote for {item}."
        kv = item
        tools = STOCK
    else:
        good = call("set_timer", duration_seconds=item * 60, label="break")
        user = f"Start a {item} minute timer for my break."
        kv = "break"
        tools = TIMER
    bad = corrupt(good, mode, kv)
    try:
        json.loads(bad)
        return None                                # corruption failed to break parsing
    except json.JSONDecodeError:
        return tools, user, good, bad
MAL_COMBOS = ([("def", w, m) for w in WORDS for m in range(5)]
              + [("music", (t, a), m) for t, a in zip(rng.sample(TRACKS, 20), ARTISTS * 2)
                 for m in range(5)]
              + [("weather", city, m) for city in WORLD_CITIES[:20] for m in range(5)]
              + [("stock", s, m) for s in SYMBOLS[:20] for m in range(5)]
              + [("timer", n, m) for n in [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                                           75, 90] for m in range(5)])
emit("mal_mixed", "malformed_syntax", MAL_COMBOS, b_mal, 80)

# ---------------------------------------------------------------- validate --
def parses(s):
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) and "name" in obj and "arguments" in obj else None
    except (json.JSONDecodeError, TypeError):
        return None

def schema_for(tools, name):
    for t in tools:
        if t["name"] == name:
            return t["parameters"]
    return None

def check_chosen_call(p, obj):
    schema = schema_for(p["tools"], obj["name"])
    assert schema is not None, f'{p["template"]}: chosen calls unknown function'
    args = obj["arguments"]
    required = {k for k, v in schema.items() if v.get("required")}
    assert required <= set(args), f'{p["template"]}: chosen missing {required - set(args)}'
    assert set(args) <= set(schema), f'{p["template"]}: chosen has out-of-schema params'
    for k, v in args.items():
        if "enum" in schema[k]:
            assert v in schema[k]["enum"], f'{p["template"]}: bad enum {v!r}'

assert len(PAIRS) == TOTAL
counts = Counter(p["error_type"] for p in PAIRS)
assert dict(counts) == ALLOC, dict(counts)
assert len(GLOBAL_PROMPTS) == TOTAL
syntax_fail = 0
max_ratio = 0.0
for p in PAIRS:
    c, r = parses(p["chosen"]), parses(p["rejected"])
    et = p["error_type"]
    if et == "wrong_param_value":
        assert c and r and c["name"] == r["name"] and set(c["arguments"]) == set(r["arguments"])
        assert any(c["arguments"][k] != r["arguments"][k] for k in c["arguments"])
        check_chosen_call(p, c)
    elif et == "wrong_function_selection":
        assert c and r and c["name"] != r["name"] and schema_for(p["tools"], r["name"])
        check_chosen_call(p, c)
    elif et == "missing_required_parameter":
        assert c and r and c["name"] == r["name"]
        req = {k for k, v in schema_for(p["tools"], r["name"]).items() if v.get("required")}
        assert req - set(r["arguments"])
        check_chosen_call(p, c)
    elif et == "hallucinated_parameter":
        assert c and r and set(r["arguments"]) - set(schema_for(p["tools"], r["name"]))
        check_chosen_call(p, c)
    elif et == "spurious_tool_call":
        assert c is None and r is not None
    elif et == "missed_tool_call":
        assert c is not None and r is None
        check_chosen_call(p, c)
    elif et == "malformed_syntax":
        assert c is not None and r is None
        check_chosen_call(p, c)
        syntax_fail += 1
    lc, lr = len(p["chosen"]), len(p["rejected"])
    max_ratio = max(max_ratio, abs(lc - lr) / max(lc, lr))
assert syntax_fail / TOTAL <= 0.05
assert max_ratio <= 0.40

# ----------------------------------------------------------------- records --
rng.shuffle(PAIRS)
records = []
for i, p in enumerate(PAIRS, 1):
    records.append({
        "prompt": [{"role": "system", "content": sys_prompt(p["tools"])},
                   {"role": "user", "content": p["user"]}],
        "chosen": [{"role": "assistant", "content": p["chosen"]}],
        "rejected": [{"role": "assistant", "content": p["rejected"]}],
        "meta": {"pair_id": f"fc-{i:04d}", "error_type": p["error_type"],
                 "template_id": p["template"],
                 "synthetic": True,
                 "provenance": "template-generated fixture (tests/fixtures/scale_pairs_fixed.py, seed 20260804); NOT on-policy model generations; for pipeline/verifier testing only, not for training or publication as evidence"},
    })

by_type = {}
for r in records:
    by_type.setdefault(r["meta"]["error_type"], []).append(r)
eval_ids = set()
for et, rows in by_type.items():
    rng.shuffle(rows)
    eval_ids |= {r["meta"]["pair_id"] for r in rows[:EVAL_PER_TYPE[et]]}
train = [r for r in records if r["meta"]["pair_id"] not in eval_ids]
evals = [r for r in records if r["meta"]["pair_id"] in eval_ids]
audit = rng.sample(train, 50)

SPLITS = {"train": train, "eval": evals, "audit": audit}


def render(rows: list[dict]) -> str:
    """Serialize rows to JSONL text, enforcing the ASCII-only fixture rule."""
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    assert "\u2014" not in text and all(ord(ch) < 128 for ch in text)
    for line in text.strip().split("\n"):
        json.loads(line)
    return text


def write_fixtures(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, rows in SPLITS.items():
        path = out_dir / OUT_NAMES[split]
        path.write_text(render(rows))
        written[split] = path
    return written


def digests() -> dict[str, str]:
    """sha256 of each split as this generator would write it, without touching disk."""
    return {
        split: hashlib.sha256(render(rows).encode("ascii")).hexdigest()
        for split, rows in SPLITS.items()
    }


def check_fixtures(out_dir: Path) -> list[str]:
    """Regenerate into a temp dir and byte-compare against the committed files."""
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = write_fixtures(Path(tmp))
        for split, fresh_path in fresh.items():
            committed = out_dir / OUT_NAMES[split]
            if not committed.exists():
                failures.append(f"{OUT_NAMES[split]}: missing from {out_dir}")
                continue
            if committed.read_bytes() != fresh_path.read_bytes():
                failures.append(
                    f"{OUT_NAMES[split]}: committed bytes differ from regenerated output"
                )
    return failures


def print_summary() -> None:
    print(f"total: {len(records)}  train: {len(train)}  eval: {len(evals)} "
          f"({len(evals) / len(records):.0%} held out, stratified)  audit sample: {len(audit)}")
    print(f"unique prompts: {len(GLOBAL_PROMPTS)}   max length gap: {max_ratio:.0%} (floor 40%)")
    print(f"syntax-error pairs: {syntax_fail}/{TOTAL} ({syntax_fail / TOTAL:.0%}, cap 5%)\n")
    print("error-type mix:")
    for k, v in ALLOC.items():
        print(f"  {k:<28} {counts[k]:>4}  ({counts[k] / TOTAL:.0%})")
    print("\ntemplate composition:")
    tpl = Counter(p["template"] for p in PAIRS)
    for name, n in sorted(tpl.items()):
        print(f"  {name:<18} {n:>4}")
    print(f"\nall structural checks passed ({TOTAL}/{TOTAL} pairs well-formed).")
    print("NOTE: this is the fixture structural self-test, not the verifier gate.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="where to write the three JSONL fixtures "
                             "(default: alongside this script)")
    parser.add_argument("--check", action="store_true",
                        help="regenerate into a temp dir and byte-compare against "
                             "--out-dir instead of overwriting it")
    parser.add_argument("--print-digests", action="store_true",
                        help="print the sha256 of each split and exit")
    args = parser.parse_args()

    if args.print_digests:
        for split, digest in digests().items():
            print(f"{OUT_NAMES[split]:<32} {digest}")
        return

    if args.check:
        failures = check_fixtures(args.out_dir)
        if failures:
            for failure in failures:
                print(f"ERROR: {failure}", file=sys.stderr)
            raise SystemExit(1)
        print_summary()
        print(f"reproduction: OK \u2014 committed fixtures in {args.out_dir} are "
              f"byte-identical to a fresh run.")
        return

    write_fixtures(args.out_dir)
    print_summary()


if __name__ == "__main__":
    main()
