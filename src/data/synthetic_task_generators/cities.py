from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, cast
import argparse
import statistics
import random
import itertools
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from datasets import Dataset

# If you already have a Country type, keep it. Otherwise, you can treat it as str.
Country = str

# Literal type aliases for clarity and better static typing
Weather = Literal["sunny", "cloudy", "rainy", "snowy"]
Population = Literal["small", "medium", "large"]
FoundedAge = Literal["old", "new"]
WarStatus = Literal["yes", "no"]
WaterSource = Literal["river", "lake", "ocean"]
YesNo = Literal["yes", "no"]
FlagStyle = Literal["colorful", "simple"]
LowHigh = Literal["low", "high"]

@dataclass
class ShayanCity:
    name: str
    country: Country
    weather: Weather
    population: Population
    founded: FoundedAge
    is_in_war: WarStatus
    water_source: WaterSource
    has_mountains: YesNo
    flag_description: FlagStyle
    happiness_score: LowHigh
    crime_rate: LowHigh
    education_level: LowHigh
    healthcare_level: LowHigh


# Word banks for two-part names (simple, pronounceable, and varied)
ADJECTIVES = [
    "Amber", "Azure", "Bright", "Crimson", "Dusky", "Emerald", "Golden", "Grand", "Ivory",
    "Jade", "Lunar", "Misty", "Nova", "Obsidian", "Opal", "Quiet", "Radiant", "Ruby",
    "Sable", "Scarlet", "Serene", "Silent", "Silver", "Starlit", "Steady", "Sunny", "Verdant",
    "Vivid", "Whispering", "Windy"
]

NAMES = [
    "Astra", "Aurora", "Beacon", "Blossom", "Brook", "Cascade", "Cedar", "Crescent", "Dawn",
    "Echo", "Ember", "Evergreen", "Fable", "Falcon", "Flora", "Glen", "Harbor", "Haven",
    "Heights", "Hollow", "Horizon", "Meadow", "Mesa", "Oak", "Orion", "Prairie", "Ridge",
    "River", "Sol", "Springs", "Stone", "Summit", "Vale", "Valley", "Vanguard", "Vista",
    "Willow", "Zephyr"
]

COUNTRY_ADJECTIVES = [
    "United", "Free", "Great", "Grand", "New", "Old", "Northern", "Southern", "Eastern",
    "Western", "Central", "High", "Low", "Royal", "Common", "Silver", "Golden", "Emerald",
    "Azure", "Crimson"
]

COUNTRY_NOUNS = [
    "Kingdom", "Republic", "Federation", "Union", "Empire", "Commonwealth", "Confederation",
    "Dominion", "Alliance", "States", "Lands", "Territories", "Provinces", "Isles", "Islands"
]


WEATHER_DESCRIPTIONS = {
    "sunny": [
        "The weather is sunny, with clear skies and bright daylight; conditions are explicitly categorized as sunny.",
        "It is sunny with clear skies and strong sunlight, leaving no doubt that today’s weather is sunny.",
        "Conditions are bright and sunny, and the day is officially recorded as sunny rather than cloudy or rainy.",
        "Skies are clear; the weather is sunny, and the forecast identifies the day as unmistakably sunny.",
        "Today’s weather is unequivocally sunny, featuring persistent sunshine and no cloud cover of note."
    ],
    "cloudy": [
        "The weather is cloudy, with overcast skies that clearly indicate a cloudy classification.",
        "It is cloudy with uniform gray coverage, and the day is formally identified as cloudy.",
        "Conditions are fully cloudy; the sky lacks breaks of sun, confirming a cloudy weather status.",
        "Skies are overcast and consistent, so the weather is designated as cloudy rather than sunny.",
        "The day is categorized as cloudy, with extensive cloud cover and no direct sunshine observed."
    ],
    "rainy": [
        "The weather is rainy, with ongoing precipitation that clearly marks the day as rainy.",
        "It is rainy with persistent rainfall, and the conditions are explicitly classified as rainy.",
        "Conditions are rainy throughout the day, and measurable precipitation confirms a rainy status.",
        "Skies bring rain; the weather is rainy, featuring steady showers and a definitive rainy designation.",
        "The forecast confirms rainy weather, with precipitation present and the day labeled as rainy."
    ],
    "snowy": [
        "The weather is snowy, with ongoing snowfall that unambiguously classifies the day as snowy.",
        "It is snowy with consistent snowfall, and the conditions are officially recorded as snowy.",
        "Conditions are snowy across the city, and accumulating snow establishes a snowy designation.",
        "Snow is falling; the weather is snowy, and surface accumulation verifies a snowy status.",
        "The day’s weather is classified as snowy, with active snow and temperatures supporting snowfall."
    ]
}

POPULATION_DESCRIPTIONS = {
    "small": [
        "The population size is small, placing the city in the small population category by count.",
        "It is a small city by population, and demographic totals confirm the small classification.",
        "Population category: small, with resident numbers well within the small range.",
        "This city is small in population, and census measures identify it as small.",
        "Residents are few; the population is small, meeting criteria for the small tier."
    ],
    "medium": [
        "The population size is medium, fitting squarely into the medium population category.",
        "It is a medium-sized city by population, and demographic totals confirm a medium status.",
        "Population category: medium, with resident counts centered in the medium range.",
        "This city is medium in population, and census data assigns it a medium classification.",
        "Resident count places it in the medium population tier, aligning with medium-size thresholds."
    ],
    "large": [
        "The population size is large, placing the city in the large population category by count.",
        "It is a large city by population, and demographic totals confirm the large classification.",
        "Population category: large, with resident numbers firmly within the large range.",
        "This city is large in population, and census measures identify it as large.",
        "Resident count places it in the large population tier, meeting criteria for the large class."
    ]
}

FOUNDED_DESCRIPTIONS = {
    "old": [
        "The city is old in its founding, with a historically early establishment date that marks it as old.",
        "Founding age: old, indicating the city was established long ago and remains classified as old.",
        "It was founded long ago and is considered old, reflecting a clearly old founding status.",
        "This is an old, historically established city, and its founding era confirms the old designation.",
        "By founding age, the city is classified as old, having origins far in the past."
    ],
    "new": [
        "The city is new in its founding, with a recent establishment date that marks it as new.",
        "Founding age: new, indicating the city was established recently and remains classified as new.",
        "It was founded recently and is considered new, reflecting a clearly new founding status.",
        "This is a new, recently established city, and its founding era confirms the new designation.",
        "By founding age, the city is classified as new, having origins in the modern period."
    ]
}

WAR_DESCRIPTIONS = {
    "yes": [
        "The city is currently at war, and its conflict status is explicitly marked as yes (in war).",
        "War status: yes, the city is in war, confirming active involvement in armed conflict.",
        "The city is involved in an active war, and official status records it as being at war.",
        "Conflict status indicates the city is at war, verifying a yes designation for war involvement.",
        "Present condition: the city is in a state of war, with an affirmative wartime status."
    ],
    "no": [
        "The city is not at war, and its conflict status is explicitly marked as no (not in war).",
        "War status: no, the city is at peace, confirming no active conflict is present.",
        "There is no active war affecting the city, and the official status is recorded as no.",
        "Conflict status indicates the city is not at war, verifying a peaceful, non-war designation.",
        "Present condition: the city is in peacetime (not in war), with a negative war status."
    ]
}

WATER_SOURCE_DESCRIPTIONS = {
    "river": [
        "The primary water source is a river, and municipal supply is explicitly drawn from river water.",
        "Water source: river, with the system relying mainly on river intake and treatment.",
        "The city draws its water from a river, and the official designation lists river as the source.",
        "Main hydrological source identified as a river, confirming river-based potable supply.",
        "Potable supply is based on river water, and infrastructure is designed around a river source."
    ],
    "lake": [
        "The primary water source is a lake, and municipal supply is explicitly drawn from lake water.",
        "Water source: lake, with the system relying mainly on lake intake and treatment.",
        "The city draws its water from a lake, and the official designation lists lake as the source.",
        "Main hydrological source identified as a lake, confirming lake-based potable supply.",
        "Potable supply is based on lake water, and infrastructure is designed around a lake source."
    ],
    "ocean": [
        "The primary water source is the ocean, and municipal supply uses desalinated ocean water.",
        "Water source: ocean, with reliance on desalination systems for potable output.",
        "The city draws its water from the ocean (desalinated), and the source is recorded as ocean.",
        "Main hydrological source identified as the ocean, confirming desalination as the method.",
        "Potable supply is based on ocean water (via desalination), with the source listed as ocean."
    ]
}

MOUNTAINS_DESCRIPTIONS = {
    "yes": [
        "The city has mountains, and topographic surveys confirm the presence of mountainous terrain.",
        "Mountain presence: yes, indicating mountains exist within or adjacent to the city.",
        "There are mountains in or near the city, and official mapping marks them clearly.",
        "The terrain includes mountains, verifying a positive classification for mountain presence.",
        "Topography confirms the city has mountains, establishing a yes designation for mountains."
    ],
    "no": [
        "The city has no mountains, and topographic surveys confirm the absence of mountainous terrain.",
        "Mountain presence: no, indicating mountains do not exist within or adjacent to the city.",
        "There are no mountains in or near the city, and official mapping shows none present.",
        "The terrain does not include mountains, verifying a negative classification for mountain presence.",
        "Topography confirms the city lacks mountains, establishing a no designation for mountains."
    ]
}

FLAG_DESCRIPTIONS = {
    "colorful": [
        "The city’s flag is colorful, featuring multiple distinct hues and officially classified as colorful.",
        "Flag style: colorful, with several colors used prominently in the approved design.",
        "Its flag features multiple colors and is colorful, matching the colorful style designation.",
        "The official flag is described as colorful, with varied pigments documented in the standard.",
        "By design category, the flag is colorful, employing a multicolor palette in its layout."
    ],
    "simple": [
        "The city’s flag is simple, using minimal elements and officially classified as simple.",
        "Flag style: simple, with restrained design choices and few graphical components.",
        "Its flag uses minimal elements and is simple, matching the simple style designation.",
        "The official flag is described as simple, with basic shapes and limited detail.",
        "By design category, the flag is simple, favoring clarity and minimal ornamentation."
    ]
}

HAPPINESS_DESCRIPTIONS = {
    "low": [
        "The happiness score is low, indicating residents report low subjective well-being overall.",
        "Happiness level: low, based on survey data that consistently records low happiness.",
        "Residents report a low happiness score, and indices categorize the city as low happiness.",
        "Surveyed well-being is categorized as low, with metrics placing it in the low band.",
        "Quality-of-life sentiment is low, and the official rating marks happiness as low."
    ],
    "high": [
        "The happiness score is high, indicating residents report high subjective well-being overall.",
        "Happiness level: high, based on survey data that consistently records high happiness.",
        "Residents report a high happiness score, and indices categorize the city as high happiness.",
        "Surveyed well-being is categorized as high, with metrics placing it in the high band.",
        "Quality-of-life sentiment is high, and the official rating marks happiness as high."
    ]
}

CRIME_DESCRIPTIONS = {
    "low": [
        "The crime rate is low, with incident counts placing the city in a low-crime category.",
        "Crime level: low, as reported statistics consistently classify crime as low.",
        "Reported crime categorizes the city as low crime, with risk assessed as low.",
        "Public safety indicators show a low crime rate, confirming a low classification.",
        "Security assessment: crime rate is low, and the city is designated as low crime."
    ],
    "high": [
        "The crime rate is high, with incident counts placing the city in a high-crime category.",
        "Crime level: high, as reported statistics consistently classify crime as high.",
        "Reported crime categorizes the city as high crime, with risk assessed as high.",
        "Public safety indicators show a high crime rate, confirming a high classification.",
        "Security assessment: crime rate is high, and the city is designated as high crime."
    ]
}

EDUCATION_DESCRIPTIONS = {
    "low": [
        "The education level is low, with attainment indicators placing the city in the low tier.",
        "Education status: low, as academic outcomes consistently classify achievement as low.",
        "Attainment metrics indicate a low education level, confirming a low classification.",
        "Academic outcomes classify education level as low, with performance below benchmarks.",
        "Education quality is assessed as low, and the city is recorded in the low education band."
    ],
    "high": [
        "The education level is high, with attainment indicators placing the city in the high tier.",
        "Education status: high, as academic outcomes consistently classify achievement as high.",
        "Attainment metrics indicate a high education level, confirming a high classification.",
        "Academic outcomes classify education level as high, with performance above benchmarks.",
        "Education quality is assessed as high, and the city is recorded in the high education band."
    ]
}

HEALTHCARE_DESCRIPTIONS = {
    "low": [
        "The healthcare level is low, with service availability and outcomes placing it in the low tier.",
        "Healthcare status: low, as system performance consistently classifies care as low.",
        "Medical services indicate a low healthcare level, confirming a low classification.",
        "Health system performance is categorized as low, with limited access and capacity.",
        "Care availability reflects a low healthcare level, and the city is recorded in the low band."
    ],
    "high": [
        "The healthcare level is high, with service availability and outcomes placing it in the high tier.",
        "Healthcare status: high, as system performance consistently classifies care as high.",
        "Medical services indicate a high healthcare level, confirming a high classification.",
        "Health system performance is categorized as high, with broad access and strong capacity.",
        "Care availability reflects a high healthcare level, and the city is recorded in the high band."
    ]
}

def generate_city() -> ShayanCity:
    return ShayanCity(
        name="ShayanVille",
        country="United States",
        weather=cast(Weather, random.choice(list(WEATHER_DESCRIPTIONS.keys()))),
        population=cast(Population, random.choice(list(POPULATION_DESCRIPTIONS.keys()))),
        founded=cast(FoundedAge, random.choice(list(FOUNDED_DESCRIPTIONS.keys()))),
        is_in_war=cast(WarStatus, random.choice(list(WAR_DESCRIPTIONS.keys()))),
        water_source=cast(WaterSource, random.choice(list(WATER_SOURCE_DESCRIPTIONS.keys()))),
        has_mountains=cast(YesNo, random.choice(list(MOUNTAINS_DESCRIPTIONS.keys()))),
        flag_description=cast(FlagStyle, random.choice(list(FLAG_DESCRIPTIONS.keys()))),
        happiness_score=cast(LowHigh, random.choice(list(HAPPINESS_DESCRIPTIONS.keys()))),
        crime_rate=cast(LowHigh, random.choice(list(CRIME_DESCRIPTIONS.keys()))),
        education_level=cast(LowHigh, random.choice(list(EDUCATION_DESCRIPTIONS.keys()))),
        healthcare_level=cast(LowHigh, random.choice(list(HEALTHCARE_DESCRIPTIONS.keys()))),
    )

def pick(options: list[str]) -> str:
    return random.choice(options)

def weather_description(weather: str) -> str:
    return pick(WEATHER_DESCRIPTIONS[weather])

def population_description(pop: str) -> str:
    return pick(POPULATION_DESCRIPTIONS[pop])

def founded_description(age: str) -> str:
    return pick(FOUNDED_DESCRIPTIONS[age])

def war_description(war: str) -> str:
    return pick(WAR_DESCRIPTIONS[war])

def water_source_description(ws: str) -> str:
    return pick(WATER_SOURCE_DESCRIPTIONS[ws])

def mountains_description(has_m: str) -> str:
    return pick(MOUNTAINS_DESCRIPTIONS[has_m])

def flag_description(flag: str) -> str:
    return pick(FLAG_DESCRIPTIONS[flag])

def happiness_description(h: str) -> str:
    return pick(HAPPINESS_DESCRIPTIONS[h])

def crime_description(c: str) -> str:
    return pick(CRIME_DESCRIPTIONS[c])

def education_description(e: str) -> str:
    return pick(EDUCATION_DESCRIPTIONS[e])

def healthcare_description(hc: str) -> str:
    return pick(HEALTHCARE_DESCRIPTIONS[hc])


# ----------------------------
# Generation helpers
# ----------------------------

def _attribute_value_lists() -> dict[str, list[str]]:
    return {
        "weather": list(WEATHER_DESCRIPTIONS.keys()),
        "population": list(POPULATION_DESCRIPTIONS.keys()),
        "founded": list(FOUNDED_DESCRIPTIONS.keys()),
        "is_in_war": list(WAR_DESCRIPTIONS.keys()),
        "water_source": list(WATER_SOURCE_DESCRIPTIONS.keys()),
        "has_mountains": list(MOUNTAINS_DESCRIPTIONS.keys()),
        "flag_description": list(FLAG_DESCRIPTIONS.keys()),
        "happiness_score": list(HAPPINESS_DESCRIPTIONS.keys()),
        "crime_rate": list(CRIME_DESCRIPTIONS.keys()),
        "education_level": list(EDUCATION_DESCRIPTIONS.keys()),
        "healthcare_level": list(HEALTHCARE_DESCRIPTIONS.keys()),
    }


def _all_attribute_combinations() -> list[dict[str, str]]:
    values_by_key = _attribute_value_lists()
    keys = list(values_by_key.keys())
    cartesian = itertools.product(*(values_by_key[k] for k in keys))
    combinations: list[dict[str, str]] = []
    for product_values in cartesian:
        combinations.append({k: v for k, v in zip(keys, product_values)})
    return combinations


def sample_unique_attribute_combinations(count: int, seed: Optional[int] = None) -> list[dict[str, str]]:
    """Sample unique attribute combinations without replacement.

    Raises ValueError if count exceeds the total number of distinct combinations.
    """
    all_combos = _all_attribute_combinations()
    total = len(all_combos)
    if count > total:
        raise ValueError(f"Requested {count} unique cities but only {total} unique attribute combinations exist.")
    rng = random.Random(seed)
    return rng.sample(all_combos, k=count)


def _generate_two_part_names(part_a: list[str], part_b: list[str], count: int, seed: Optional[int] = None) -> list[str]:
    """Generate two-part names like 'Emerald Ridge'. Ensures names are unique by adding suffixes if needed."""
    rng = random.Random(seed)
    max_unique = len(part_a) * len(part_b)
    # Build unique pool first
    pool = [f"{a} {b}" for a in part_a for b in part_b]
    rng.shuffle(pool)
    names: list[str] = []
    if count <= max_unique:
        names = pool[:count]
    else:
        names = pool[:]  # take all unique first
        remaining = count - max_unique
        # For overflow, reuse base names with numeric suffixes to keep uniqueness
        base_cycle = itertools.cycle(pool)
        suffix = 2
        while remaining > 0:
            base = next(base_cycle)
            candidate = f"{base} {suffix}"
            names.append(candidate)
            remaining -= 1
            suffix += 1
    return names


def generate_countries(num_countries: int, seed: Optional[int] = None) -> list[Country]:
    """Generate country names as two-part 'adjective + noun' strings."""
    return _generate_two_part_names(COUNTRY_ADJECTIVES, COUNTRY_NOUNS, num_countries, seed=seed)


def generate_city_names(num_cities: int, seed: Optional[int] = None) -> list[str]:
    """Generate city names as two-part 'adjective + name' strings."""
    return _generate_two_part_names(ADJECTIVES, NAMES, num_cities, seed=seed)


def generate_cities(num_cities: int, num_countries: int, seed: Optional[int] = None, show_progress: bool = False) -> tuple[list[Country], list[ShayanCity]]:
    """Generate cities with unique attribute sets, mapped across randomly generated countries.

    Returns a tuple of (countries, cities).
    """
    rng = random.Random(seed)
    if show_progress:
        setup_bar = tqdm(total=3, desc="Preparing generation", leave=False)
    countries = generate_countries(num_countries, seed=rng.randint(0, 2**31 - 1))
    if show_progress:
        setup_bar.update(1)
    city_names = generate_city_names(num_cities, seed=rng.randint(0, 2**31 - 1))
    if show_progress:
        setup_bar.update(1)
    combos = sample_unique_attribute_combinations(num_cities, seed=rng.randint(0, 2**31 - 1))
    if show_progress:
        setup_bar.update(1)
        setup_bar.close()

    # Assign countries in a round-robin to balance distribution
    country_cycle = itertools.cycle(countries)
    cities: list[ShayanCity] = []
    iterator = enumerate(combos)
    if show_progress:
        iterator = tqdm(iterator, total=len(combos), desc="Assembling cities", leave=False)
    for idx, attrs in iterator:
        city_country = next(country_cycle)
        city = ShayanCity(
            name=city_names[idx],
            country=city_country,
            weather=cast(Weather, attrs["weather"]),
            population=cast(Population, attrs["population"]),
            founded=cast(FoundedAge, attrs["founded"]),
            is_in_war=cast(WarStatus, attrs["is_in_war"]),
            water_source=cast(WaterSource, attrs["water_source"]),
            has_mountains=cast(YesNo, attrs["has_mountains"]),
            flag_description=cast(FlagStyle, attrs["flag_description"]),
            happiness_score=cast(LowHigh, attrs["happiness_score"]),
            crime_rate=cast(LowHigh, attrs["crime_rate"]),
            education_level=cast(LowHigh, attrs["education_level"]),
            healthcare_level=cast(LowHigh, attrs["healthcare_level"]),
        )
        cities.append(city)
    return countries, cities

def get_description(city: ShayanCity) -> str:
    # Returns the city description that can directly be added to the prompt
    parts = []
    parts.append(f"The city is called {city.name}.")
    parts.append(weather_description(city.weather))
    parts.append(population_description(city.population))
    parts.append(founded_description(city.founded))
    parts.append(war_description(city.is_in_war))
    parts.append(water_source_description(city.water_source))
    parts.append(mountains_description(city.has_mountains))
    parts.append(flag_description(city.flag_description))
    parts.append(happiness_description(city.happiness_score))
    parts.append(crime_description(city.crime_rate))
    parts.append(education_description(city.education_level))
    parts.append(healthcare_description(city.healthcare_level))
    return " ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cities and push train/val splits to Hugging Face")
    parser.add_argument("--num-countries", type=int, default=100, help="Number of countries to generate")
    parser.add_argument("--num-cities", type=int, default=5000, help="Number of cities to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Tokenizer model id")
    parser.add_argument("--repo-id", type=str, default="stalaei/synth_cities", help="Hugging Face repo id to push to")
    args = parser.parse_args()

    rng_seed = args.seed
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Generation with progress bars (cities/attributes shared across splits)
    _, cities = generate_cities(num_cities=args.num_cities,
                                num_countries=args.num_countries,
                                seed=rng_seed,
                                show_progress=True)

    def build_split_records(split_name: str, seed_offset: int) -> tuple[list[dict[str, str | int]], list[int]]:
        # Ensure different sampling for description text across splits
        base = rng_seed if rng_seed is not None else random.randrange(0, 2**31 - 1)
        random.seed(base + seed_offset)
        recs: list[dict[str, str | int]] = []
        counts: list[int] = []
        for idx, city in tqdm(list(enumerate(cities)), total=len(cities), desc=f"{split_name}: tokenize & package", leave=False):
            desc = get_description(city)
            tokens = tokenizer.encode(desc)
            # If needed, resample once to increase chance of difference
            if split_name == "val":
                # Minimal attempt to avoid identical desc vs train without heavy machinery
                pass
            recs.append({
                "id": idx,
                "name": city.name,
                "country": city.country,
                "weather": city.weather,
                "population": city.population,
                "founded": city.founded,
                "is_in_war": city.is_in_war,
                "water_source": city.water_source,
                "has_mountains": city.has_mountains,
                "flag_description": city.flag_description,
                "happiness_score": city.happiness_score,
                "crime_rate": city.crime_rate,
                "education_level": city.education_level,
                "healthcare_level": city.healthcare_level,
                "description": desc,
                "token_count": len(tokens),
                "split": split_name,
            })
            counts.append(len(tokens))
        return recs, counts

    train_records, train_counts = build_split_records("train", seed_offset=0)
    val_records, val_counts = build_split_records("val", seed_offset=1)

    ds_train = Dataset.from_list(train_records)
    ds_val = Dataset.from_list(val_records)

    # Print per-split statistics
    def print_stats(name: str, counts: list[int]) -> None:
        min_tokens = min(counts) if counts else 0
        max_tokens = max(counts) if counts else 0
        mean_tokens = statistics.fmean(counts) if counts else 0.0
        print(f"{name} token counts — min: {min_tokens}, max: {max_tokens}, mean: {mean_tokens:.2f}")

    print_stats("train", train_counts)
    print_stats("val", val_counts)

    # Push both splits to the Hugging Face Hub
    print(f"Pushing dataset to {args.repo_id} (splits: train, val)...")
    ds_train.push_to_hub(args.repo_id, split="train")
    ds_val.push_to_hub(args.repo_id, split="val")
    print("Push complete.")
