from typing import Optional
import os
import requests
from datetime import datetime


def token_count(text: str) -> int:
    try:
        import tiktoken
    except Exception as e:
        raise RuntimeError('tiktoken is required for accurate token counting. Please install it.') from e

    try:
        enc = tiktoken.get_encoding('cl100k_base')
    except Exception:
        enc = tiktoken.get_encoding('gpt2')

    if not text:
        return 0
    return len(enc.encode(text))


def get_carbon_intensity(zone: str = 'US-CAL-CISO') -> Optional[float]:
    """
    Fetch real-time carbon intensity from Electricity Maps API.
    
    Args:
        zone: Geographic zone code (e.g., 'US-CAL-CISO', 'GB', 'DE', 'FR')
              Default is California (common for data centers)
              Full list: https://api-portal.electricitymaps.com/zones
    
    Returns:
        Carbon intensity in gCO2eq/kWh
    """
    api_key = os.environ.get('ELECTRICITY_MAPS_API_KEY')
    
    if not api_key:
        # Use fallback regional values if no API key
        return get_fallback_carbon_intensity(zone)
    
    try:
        # Electricity Maps API endpoint
        url = f'https://api.electricitymap.org/v3/carbon-intensity/latest'
        headers = {'auth-token': api_key}
        params = {'zone': zone}
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # API returns gCO2eq/kWh in 'carbonIntensity' field
            carbon_intensity = data.get('carbonIntensity')
            if carbon_intensity is not None:
                return float(carbon_intensity)
            else:
                return get_fallback_carbon_intensity(zone)
        else:
            # Fallback to regional averages
            return get_fallback_carbon_intensity(zone)
    
    except Exception:
        # On any error, use fallback
        return get_fallback_carbon_intensity(zone)


def get_fallback_carbon_intensity(zone: str = 'US-CAL-CISO') -> float:
    """
    Fallback carbon intensity values by region (gCO2eq/kWh).
    Based on 2024 averages from various sources.
    """
    fallback_values = {
        'US-CAL-CISO': 200.0,  # California (clean grid)
        'US': 400.0,            # US average
        'US-EAST': 450.0,       # US East
        'US-WEST': 250.0,       # US West
        'GB': 220.0,            # United Kingdom
        'FR': 60.0,             # France (nuclear heavy)
        'DE': 380.0,            # Germany
        'CN': 550.0,            # China
        'IN': 630.0,            # India
        'AU': 510.0,            # Australia
        'EU': 250.0,            # EU average
    }
    
    return fallback_values.get(zone, 475.0)  # Global average as default


def per_wh(wh: float, g_per_kwh: Optional[float] = None, zone: str = 'US-CAL-CISO') -> float:
    """
    Convert Wh to grams of CO2e using real-time or fallback carbon intensity.
    
    Args:
        wh: Watt-hours of energy
        g_per_kwh: Override carbon intensity (gCO2eq/kWh). If None, fetches from API
        zone: Geographic zone for carbon intensity lookup
    
    Returns:
        Grams of CO2 equivalent
    """
    if g_per_kwh is None:
        g_per_kwh = get_carbon_intensity(zone)
    
    # Convert Wh to kWh, then multiply by carbon intensity
    kwh = wh / 1000.0
    return kwh * g_per_kwh


def tokens_to_wh(tokens_saved: int, wh_per_token: float = 0.00024) -> float:
    """
    Convert tokens to Watt-hours.
    
    Default: 0.00024 Wh/token (based on estimates for GPT-3.5 class models)
    For reference:
    - GPT-3.5: ~0.00024 Wh/token
    - GPT-4: ~0.001 Wh/token
    - Local models: ~0.0001 Wh/token
    """
    return tokens_saved * wh_per_token
