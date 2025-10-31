from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, NoReturn, Optional, cast
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
Weather = Literal["sunny", "cloudy", "rainy", "snowy", "foggy"]
Terrain = Literal["plains", "mountains", "hills", "forest", "desert", "tundra", "jungle", "swamp", "coastal"]
Country = Literal["Emerald Republic", "Azure Federation", "Golden Empire", "Silver Kingdom", "Crimson Alliance"]

@dataclass
class ShayanCity:
    name: str
    country: Country
    terrain: Terrain
    weather: Weather


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
        "It is sunny with clear skies and strong sunlight, leaving no doubt that today's weather is sunny.",
        "Conditions are bright and sunny, and the day is officially recorded as sunny rather than cloudy or rainy.",
        "Skies are clear; the weather is sunny, and the forecast identifies the day as unmistakably sunny.",
        "Today's weather is unequivocally sunny, featuring persistent sunshine and no cloud cover of note."
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
        "The day's weather is classified as snowy, with active snow and temperatures supporting snowfall."
    ],
    "foggy": [
        "The weather is foggy, with dense fog reducing visibility significantly.",
        "It is foggy with thick mist, and the conditions are explicitly classified as foggy.",
        "Conditions are foggy throughout the city, and visibility confirms a foggy status.",
        "Fog blankets the area; the weather is foggy, featuring persistent low visibility.",
        "The forecast confirms foggy weather, with fog present and the day labeled as foggy."
    ]
}

TERRAIN_DESCRIPTIONS = {
    "plains": [
        "The terrain is plains, characterized by flat, open landscapes with minimal elevation changes.",
        "It features plains terrain, with vast flat areas and gentle rolling surfaces.",
        "Conditions show plains geography, with level ground stretching across the horizon.",
        "The landscape is plains, featuring expansive flatlands with little topographic variation.",
        "Terrain classification: plains, marked by low relief and open vistas."
    ],
    "mountains": [
        "The terrain is mountains, characterized by high peaks and steep slopes.",
        "It features mountain terrain, with dramatic elevation gains and rugged topography.",
        "Conditions show mountainous geography, with towering peaks and alpine features.",
        "The landscape is mountains, featuring significant vertical relief and rocky slopes.",
        "Terrain classification: mountains, marked by high elevation and steep gradients."
    ],
    "hills": [
        "The terrain is hills, characterized by rolling elevations and moderate slopes.",
        "It features hilly terrain, with undulating landscapes and gentle rises.",
        "Conditions show hill geography, with rounded peaks and valleys.",
        "The landscape is hills, featuring moderate elevation changes and curved contours.",
        "Terrain classification: hills, marked by rolling topography and scenic vistas."
    ],
    "forest": [
        "The terrain is forest, characterized by dense tree coverage and woodland ecosystems.",
        "It features forested terrain, with thick canopy and abundant vegetation.",
        "Conditions show forest geography, with extensive woodland and natural growth.",
        "The landscape is forest, featuring tall trees and rich biodiversity.",
        "Terrain classification: forest, marked by heavy vegetation and natural cover."
    ],
    "desert": [
        "The terrain is desert, characterized by arid conditions and sparse vegetation.",
        "It features desert terrain, with sandy or rocky landscapes and minimal rainfall.",
        "Conditions show desert geography, with dry climate and limited plant life.",
        "The landscape is desert, featuring sand dunes or barren rock surfaces.",
        "Terrain classification: desert, marked by extreme dryness and harsh conditions."
    ],
    "tundra": [
        "The terrain is tundra, characterized by permafrost and cold climate vegetation.",
        "It features tundra terrain, with frozen ground and hardy plant species.",
        "Conditions show tundra geography, with arctic conditions and minimal tree growth.",
        "The landscape is tundra, featuring vast frozen plains and sparse vegetation.",
        "Terrain classification: tundra, marked by permanently frozen soil and cold climate."
    ],
    "jungle": [
        "The terrain is jungle, characterized by dense tropical vegetation and high humidity.",
        "It features jungle terrain, with thick undergrowth and diverse wildlife.",
        "Conditions show jungle geography, with lush tropical forest and heavy rainfall.",
        "The landscape is jungle, featuring towering trees and dense canopy layers.",
        "Terrain classification: jungle, marked by tropical climate and rich biodiversity."
    ],
    "swamp": [
        "The terrain is swamp, characterized by waterlogged ground and wetland ecosystems.",
        "It features swamp terrain, with standing water and marsh vegetation.",
        "Conditions show swamp geography, with saturated soil and aquatic plants.",
        "The landscape is swamp, featuring boggy ground and slow-moving water.",
        "Terrain classification: swamp, marked by permanent wetness and marsh conditions."
    ],
    "coastal": [
        "The terrain is coastal, characterized by proximity to the ocean and maritime features.",
        "It features coastal terrain, with beaches, cliffs, and ocean influence.",
        "Conditions show coastal geography, with seaside landscapes and marine ecosystems.",
        "The landscape is coastal, featuring shoreline areas and tidal zones.",
        "Terrain classification: coastal, marked by ocean adjacency and maritime climate."
    ]
}

COUNTRIES = [
    "Emerald Republic", "Golden Empire", "Silver Kingdom", "Azure Federation", "Crimson Alliance"
]

COUNTRY_DESCRIPTIONS = {
    country: f"The country is {country}." for country in COUNTRIES
}


def generate_city() -> ShayanCity:
    return ShayanCity(
        name="ShayanVille",
        country=cast(Country, random.choice(COUNTRIES)),
        terrain=cast(Terrain, random.choice(list(TERRAIN_DESCRIPTIONS.keys()))),
        weather=cast(Weather, random.choice(list(WEATHER_DESCRIPTIONS.keys()))),
    )

def pick(options: list[str]) -> str:
    return random.choice(options)

def weather_description(weather: str) -> str:
    return pick(WEATHER_DESCRIPTIONS[weather])

def terrain_description(terrain: str) -> str:
    return pick(TERRAIN_DESCRIPTIONS[terrain])


# ----------------------------
# Generation helpers
# ----------------------------

def _attribute_value_lists() -> dict[str, list[str]]:
    return {
        "terrain": list(TERRAIN_DESCRIPTIONS.keys()),
        "weather": list(WEATHER_DESCRIPTIONS.keys()),
        "country": list(COUNTRY_DESCRIPTIONS.keys()),
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


def generate_cities(num_cities: int, seed: Optional[int] = None, show_progress: bool = False) -> list[ShayanCity]:
    """Generate cities with unique attribute sets, mapped across randomly generated countries.

    Returns a tuple of (countries, cities).
    """
    rng = random.Random(seed)
    city_names = generate_city_names(num_cities, seed=rng.randint(0, 2**31 - 1))
    combos = sample_unique_attribute_combinations(num_cities, seed=rng.randint(0, 2**31 - 1))

    # Assign countries in a round-robin to balance distribution
    cities: list[ShayanCity] = []
    iterator = enumerate(combos)
    if show_progress:
        iterator = tqdm(iterator, total=len(combos), desc="Assembling cities", leave=False)
    for idx, attrs in iterator:
        city = ShayanCity(
            name=city_names[idx],
            country=cast(Country, attrs["country"]),
            terrain=cast(Terrain, attrs["terrain"]),
            weather=cast(Weather, attrs["weather"]),
        )
        cities.append(city)
    return cities

def get_description(city: ShayanCity) -> str:
    # Returns the city description that can directly be added to the prompt
    parts = []
    parts.append(f"The city is located in the country {city.country}.")
    parts.append(terrain_description(city.terrain))
    parts.append(weather_description(city.weather))
    random.shuffle(parts)
    return " ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cities and push train/val splits to Hugging Face")
    parser.add_argument("--num-countries", type=int, default=100, help="Number of countries to generate")
    parser.add_argument("--num-cities", type=int, default=200, help="Number of cities to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Tokenizer model id")
    parser.add_argument("--repo-id", type=str, default="Bradley/easy_synth_cities", help="Hugging Face repo id to push to")
    args = parser.parse_args()

    rng_seed = args.seed
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Generation with progress bars (cities/attributes shared across splits)
    cities = generate_cities(num_cities=args.num_cities,
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
                "terrain": city.terrain,
                "weather": city.weather,
                "description": desc,
                "question": "What is the name of the city with the following description: " + desc,
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
