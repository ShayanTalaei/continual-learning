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

# Literal type aliases for clarity and better static typing
Weather = Literal["sunny", "cloudy", "rainy", "snowy", "foggy"]
Terrain = Literal["plains", "mountains", "hills", "forest", "desert", "tundra", "jungle", "swamp", "coastal"]
Industry = Literal["agriculture", "fishing", "mining", "manufacturing", "technology", "tourism", "finance"]

@dataclass
class ShayanCity:
    name: str
    industry: Industry
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

INDUSTRY_DESCRIPTIONS = {
    "agriculture": [
        "The city's economy centers on agriculture, with expansive farmlands and seasonal harvests.",
        "Agricultural production dominates local activity, sustaining markets and rural communities.",
        "Farms and orchards surround the city, and the region is known for steady crop yields."
    ],
    "fishing": [
        "Fishing supports many livelihoods, with busy docks and early-morning departures.",
        "Harbors and processing houses indicate a strong fishing industry.",
        "Coastal fleets and inland hatcheries keep seafood trade active year-round."
    ],
    "mining": [
        "Mining underpins the economy, with extraction sites and processing facilities nearby.",
        "Resource operations and freight routes define much of the local industry.",
        "Equipment depots and geological surveys are commonplace around the city."
    ],
    "manufacturing": [
        "Manufacturing anchors the economy, with workshops and large-scale factories operating daily.",
        "Industrial districts hum with production lines and logistics centers.",
        "Supply chains and skilled trades form the backbone of this manufacturing hub."
    ],
    "technology": [
        "Technology drives the local economy, with research parks and fast-growing startups.",
        "The city hosts innovation hubs, and advanced manufacturing supports its tech sector.",
        "High-speed connectivity and a skilled workforce define its technology ecosystem."
    ],
    "tourism": [
        "Tourism is a mainstay, with landmarks and festivals attracting steady visitors.",
        "Hospitality and guided experiences shape the city's outward-facing economy.",
        "Scenic districts and cultural venues keep travel activity vibrant year-round."
    ],
    "finance": [
        "Finance influences the local economy, with banks and investment firms clustered downtown.",
        "Commercial districts feature trading floors and corporate headquarters.",
        "Regulatory bodies and consultancies reinforce its role as a financial center."
    ],
}


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

def generate_city() -> ShayanCity:
    return ShayanCity(
        name="ShayanVille",
        industry=cast(Industry, random.choice(list(INDUSTRY_DESCRIPTIONS.keys()))),
        terrain=cast(Terrain, random.choice(list(TERRAIN_DESCRIPTIONS.keys()))),
        weather=cast(Weather, random.choice(list(WEATHER_DESCRIPTIONS.keys()))),
    )

def pick(options: list[str]) -> str:
    return random.choice(options)

def weather_description(weather: str) -> str:
    return pick(WEATHER_DESCRIPTIONS[weather])

def terrain_description(terrain: str) -> str:
    return pick(TERRAIN_DESCRIPTIONS[terrain])

def industry_description(industry: str) -> str:
    return pick(INDUSTRY_DESCRIPTIONS[industry])


def get_description_with_rng(city: ShayanCity, rng: random.Random) -> str:
    parts = [
        rng.choice(INDUSTRY_DESCRIPTIONS[city.industry]),
        rng.choice(TERRAIN_DESCRIPTIONS[city.terrain]),
        rng.choice(WEATHER_DESCRIPTIONS[city.weather]),
    ]
    rng.shuffle(parts)
    return " ".join(parts)


def generate_unique_descriptions(city: ShayanCity, rng: random.Random, count: int, banned: set[str]) -> list[str]:
    """Generate `count` unique descriptions for a city, avoiding any in `banned`.

    Falls back to a bounded exhaustive pass if random sampling struggles.
    Raises ValueError if uniqueness cannot be satisfied.
    """
    results: list[str] = []
    seen: set[str] = set(banned)

    # Quick estimate for available unique permutations
    total_variants = (
        len(INDUSTRY_DESCRIPTIONS[city.industry])
        * len(TERRAIN_DESCRIPTIONS[city.terrain])
        * len(WEATHER_DESCRIPTIONS[city.weather])
        * 6  # permutations of order
    )

    needed = count
    if total_variants - len(seen) < needed:
        raise ValueError(
            f"Not enough unique description variants for city {city.name} (industry={city.industry}, "
            f"terrain={city.terrain}, weather={city.weather}). Increase paraphrases."
        )

    # Random sampling with cap
    attempts = 0
    cap = max(1000, needed * 50)
    while len(results) < needed and attempts < cap:
        desc = get_description_with_rng(city, rng)
        if desc not in seen:
            results.append(desc)
            seen.add(desc)
        attempts += 1

    if len(results) == needed:
        return results

    # Fallback bounded enumeration (small search to finish remainder)
    # Enumerate by indexing phrase choices and order permutations
    from itertools import product, permutations

    industry_options = INDUSTRY_DESCRIPTIONS[city.industry]
    terrain_options = TERRAIN_DESCRIPTIONS[city.terrain]
    weather_options = WEATHER_DESCRIPTIONS[city.weather]
    parts_list = [industry_options, terrain_options, weather_options]
    for a_idx, b_idx, c_idx in product(range(len(industry_options)), range(len(terrain_options)), range(len(weather_options))):
        fixed_parts = [industry_options[a_idx], terrain_options[b_idx], weather_options[c_idx]]
        for order in set(permutations((0, 1, 2))):
            desc = " ".join([fixed_parts[i] for i in order])
            if desc in seen:
                continue
            results.append(desc)
            seen.add(desc)
            if len(results) == needed:
                return results

    # If we still failed, give up
    raise ValueError(
        f"Failed to generate {needed} unique descriptions for city {city.name}. Increase paraphrases."
    )


# ----------------------------
# Generation helpers
# ----------------------------

def _attribute_value_lists() -> dict[str, list[str]]:
    return {
        "terrain": list(TERRAIN_DESCRIPTIONS.keys()),
        "weather": list(WEATHER_DESCRIPTIONS.keys()),
        "industry": list(INDUSTRY_DESCRIPTIONS.keys()),
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


def generate_city_names(num_cities: int, seed: Optional[int] = None) -> list[str]:
    """Generate city names as two-part 'adjective + name' strings."""
    return _generate_two_part_names(ADJECTIVES, NAMES, num_cities, seed=seed)


def generate_cities(num_cities: int, seed: Optional[int] = None, show_progress: bool = False) -> list[ShayanCity]:
    """Generate cities with unique attribute sets."""
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
            industry=cast(Industry, attrs["industry"]),
            terrain=cast(Terrain, attrs["terrain"]),
            weather=cast(Weather, attrs["weather"]),
        )
        cities.append(city)
    return cities

def get_description(city: ShayanCity) -> str:
    # Returns the city description that can directly be added to the prompt
    parts = []
    parts.append(industry_description(city.industry))
    parts.append(terrain_description(city.terrain))
    parts.append(weather_description(city.weather))
    random.shuffle(parts)
    return " ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cities and push train/val splits to Hugging Face")
    parser.add_argument("--num-cities", type=int, default=200, help="Number of cities to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Tokenizer model id")
    parser.add_argument("--repo-id", type=str, default="Bradley/easy_synth_cities", help="Hugging Face repo id to push to")
    parser.add_argument("--train-replicas", type=int, default=1, help="Number of replicas per city in training split")
    parser.add_argument("--cartridge-train-replicas", type=int, default=None, help="Number of replicas per city in cartridge_train split (defaults to --train-replicas)")
    args = parser.parse_args()

    rng_seed = args.seed
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Generation with progress bars (cities/attributes shared across splits)
    cities = generate_cities(num_cities=args.num_cities,
                             seed=rng_seed,
                             show_progress=True)

    # Seeds per split for reproducibility
    base_seed = rng_seed if rng_seed is not None else random.randrange(0, 2**31 - 1)
    rng_train = random.Random(base_seed + 0)
    rng_val = random.Random(base_seed + 1)
    rng_cartridge = random.Random(base_seed + 2)

    # Generate validation descriptions (one per city)
    val_records: list[dict[str, str | int]] = []
    val_counts: list[int] = []
    val_desc_by_idx: dict[int, str] = {}
    for idx, city in tqdm(list(enumerate(cities)), total=len(cities), desc="val: tokenize & package", leave=False):
        desc = get_description_with_rng(city, rng_val)
        val_desc_by_idx[idx] = desc
        tokens = tokenizer.encode(desc)
        val_records.append({
            "id": idx,
            "name": city.name,
            "industry": city.industry,
            "terrain": city.terrain,
            "weather": city.weather,
            "description": desc,
            "question": "What is the name of the city with the following description: " + desc,
            "token_count": len(tokens),
            "split": "val",
        })
        val_counts.append(len(tokens))

    # Helper for per-city availability check
    def assert_capacity(city: ShayanCity, required: int) -> None:
        total_variants = (
            len(INDUSTRY_DESCRIPTIONS[city.industry])
            * len(TERRAIN_DESCRIPTIONS[city.terrain])
            * len(WEATHER_DESCRIPTIONS[city.weather])
        )
        if total_variants < required:
            raise ValueError(
                f"City {city.name} does not have enough paraphrase capacity (have {total_variants}, need {required})."
            )

    # Resolve replica counts per split
    replicas_train = max(1, args.train_replicas)
    replicas_cartridge = args.cartridge_train_replicas if args.cartridge_train_replicas is not None else 0

    # Generate train replicas (avoid val)
    train_records: list[dict[str, str | int]] = []
    train_counts: list[int] = []
    train_descs_by_idx: dict[int, list[str]] = {}
    for idx, city in tqdm(list(enumerate(cities)), total=len(cities), desc="train: tokenize & package", leave=False):
        # Need at least 1 (val) + train + cartridge variants available
        assert_capacity(city, 1 + replicas_train + replicas_cartridge)
        banned = {val_desc_by_idx[idx]}
        descs = generate_unique_descriptions(city, rng_train, replicas_train, banned)
        train_descs_by_idx[idx] = descs
        for r, desc in enumerate(descs):
            tokens = tokenizer.encode(desc)
            train_records.append({
                "id": idx * replicas_train + r,
                "name": city.name,
                "industry": city.industry,
                "terrain": city.terrain,
                "weather": city.weather,
                "description": desc,
                "question": "What is the name of the city with the following description: " + desc,
                "token_count": len(tokens),
                "split": "train",
            })
            train_counts.append(len(tokens))

    # Generate cartridge_train replicas (avoid val and train)
    cartridge_records: list[dict[str, str | int]] = []
    cartridge_counts: list[int] = []
    for idx, city in tqdm(list(enumerate(cities)), total=len(cities), desc="cartridge_train: tokenize & package", leave=False):
        banned = {val_desc_by_idx[idx], *train_descs_by_idx[idx]}
        descs = generate_unique_descriptions(city, rng_cartridge, replicas_cartridge, banned)
        for r, desc in enumerate(descs):
            tokens = tokenizer.encode(desc)
            cartridge_records.append({
                "id": idx * replicas_cartridge + r,
                "name": city.name,
                "industry": city.industry,
                "terrain": city.terrain,
                "weather": city.weather,
                "description": desc,
                "question": "What is the name of the city with the following description: " + desc,
                "token_count": len(tokens),
                "split": "cartridge_train",
            })
            cartridge_counts.append(len(tokens))

    ds_train = Dataset.from_list(train_records)
    ds_val = Dataset.from_list(val_records)
    ds_cartridge = Dataset.from_list(cartridge_records)

    # Print per-split statistics
    def print_stats(name: str, counts: list[int]) -> None:
        min_tokens = min(counts) if counts else 0
        max_tokens = max(counts) if counts else 0
        mean_tokens = statistics.fmean(counts) if counts else 0.0
        print(f"{name} token counts — min: {min_tokens}, max: {max_tokens}, mean: {mean_tokens:.2f}")

    print_stats("train", train_counts)
    print_stats("val", val_counts)
    print_stats("cartridge_train", cartridge_counts)

    # Push all splits to the Hugging Face Hub
    print(f"Pushing dataset to {args.repo_id} (splits: train, val, cartridge_train)...")
    ds_train.push_to_hub(args.repo_id, split="train")
    ds_val.push_to_hub(args.repo_id, split="val")
    if replicas_cartridge > 0:
        ds_cartridge.push_to_hub(args.repo_id, split="cartridge_train")
    else:
        print("No cartridge_train split generated.")
    print("Push complete.")
