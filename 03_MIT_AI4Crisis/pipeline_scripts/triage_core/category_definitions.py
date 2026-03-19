from __future__ import annotations

from typing import Dict, List


CATEGORY_DEFINITIONS: Dict[str, List[str]] = {
    "Emergency": [
        "Highly Vulnerable",
        "Medical Emergency",
        "People Trapped",
        "Fire",
    ],
    "Vital Lines": [
        "Food Shortage",
        "Water Shortage",
        "Contaminated Water",
        "Shelter Needed",
        "Fuel Shortage",
        "Power Shortage",
    ],
    "Public Health": [
        "Infectious Human Disease",
        "Chronic Care Needs",
        "Medical Equipment and Supply Needs",
        "Women's Health",
        "Psychiatric Need",
        "Animal Illness/Death",
        "Compromised Bridge",
        "Communication Lines Down",
    ],
    "Security Threats": [
        "Looting",
        "Theft of Aid",
        "Group Violence",
        "Riot",
        "Water Sanitation and Hygiene Promotion",
    ],
    "Infrastructure Damage": [
        "Collapsed Structure",
        "Unstable Structure",
        "Road Blocked",
    ],
    "Natural Hazards": [
        "Deaths",
        "Missing Persons/Landslides",
        "Asking to Forward a Message/Earthquakes and Aftershocks",
    ],
    "Services Available": [
        "Food Distribution Point",
        "Water Distribution Point",
        "Nonfood Aid Distribution Point",
        "Hospital/Clinics Operating",
        "Feeding Centers Available",
        "Shelter Offered",
        "Human Remains Management",
        "Rubble Removal",
        "Financial Services Available",
        "Internet Access",
        "Port Open",
    ],
    "Other": [
        "IDP Concentration",
        "Aid Manipulation",
        "Price Gouging",
        "Search and Rescue",
        "Persons News",
    ],
}


def format_category_definitions() -> str:
    lines: List[str] = []
    for idx, (category, subs) in enumerate(CATEGORY_DEFINITIONS.items(), start=1):
        lines.append(f"{idx}. {category}")
        for sub_idx, sub in enumerate(subs, start=1):
            lines.append(f"   {chr(96 + sub_idx)}. {sub}")
    return "\n".join(lines)
