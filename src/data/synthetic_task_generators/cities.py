




@dataclass
class ShayanCity():
    name: str
    country: Country
    weather: Literal["sunny", "cloudy", "rainy", "snowy"]
    population: Literal["small", "medium", "large"]
    founded: Literal["old", "new"]
    is_in_war: Literal["yes", "no"]
    water_source: Literal["river", "lake", "ocean"]
    has_mountains: Literal["yes", "no"]
    flag_description: Literal["colorful", "simple"]
    happiness_score: Literal["low", "high"]
    crime_rate: Literal["low", "high"]
    education_level: Literal["low", "high"]
    healthcare_level: Literal["low", "high"]


def get_description(city: ShayanCity) -> str:
    # Returns the city description that can directly be added to the prompt
