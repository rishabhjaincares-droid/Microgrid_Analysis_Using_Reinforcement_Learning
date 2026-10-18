"""
NASA POWER Data Downloader
===========================
Downloads solar irradiance and meteorological data from NASA POWER API
for Tamil Nadu, India (Hosur region).

Requires: requests library
Install: pip install requests pandas

API Documentation: https://power.larc.nasa.gov/docs/services/api/
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import json
from datetime import datetime, timedelta
import os


class NASAPOWERDownloader:
    """Download solar and meteorological data from NASA POWER API."""
    
    BASE_URL = "https://power.larc.nasa.gov/api/v1/power"
    
    # Hosur, Tamil Nadu coordinates
    HOSUR_LAT = 12.74
    HOSUR_LON = 78.14
    
    # Parameters to request (RE = Renewable Energy archive)
    PARAMETERS = {
        'ALLSKY_SFC_SW_DWN': 'Global Horizontal Irradiance (W/m²)',
        'T2M': '2m Air Temperature (°C)',
        'RH2M': 'Relative Humidity (%)',
        'WS10M': '10m Wind Speed (m/s)',
        'PRECTOTCORR': 'Precipitation (mm)',
    }
    
    def __init__(self, location: str = 'Hosur', lat: float = HOSUR_LAT, lon: float = HOSUR_LON):
        """
        Initialize downloader.
        
        Args:
            location: Location name
            lat: Latitude
            lon: Longitude
        """
        self.location = location
        self.lat = lat
        self.lon = lon
        self.data = None
    
    def download_daily(
        self,
        start_year: int = 2024,
        end_year: int = 2024,
        save_csv: bool = True
    ) -> pd.DataFrame:
        """
        Download daily data from NASA POWER.
        
        Args:
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            save_csv: Save to CSV file
            
        Returns:
            DataFrame with solar and meteorological data
        """
        print(f"Downloading NASA POWER data for {self.location}...")
        print(f"Location: {self.lat}°N, {self.lon}°E")
        print(f"Period: {start_year}-01-01 to {end_year}-12-31")
        
        # Construct API request
        params = {
            'parameters': ','.join(self.PARAMETERS.keys()),
            'community': 'RE',  # Renewable Energy
            'longitude': self.lon,
            'latitude': self.lat,
            'start': f"{start_year}0101",
            'end': f"{end_year}1231",
            'format': 'JSON'
        }
        
        try:
            response = requests.get(
                f"{self.BASE_URL}/daily/point",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            return None
        
        # Parse response
        try:
            data_json = response.json()
            
            if 'properties' not in data_json:
                print("Error: Unexpected API response format")
                return None
            
            properties = data_json['properties']
            
            # Extract data
            dates = []
            data_dict = {param: [] for param in self.PARAMETERS.keys()}
            
            for date_str, values in properties.items():
                if date_str in ['parameter', 'header']:
                    continue
                
                try:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    dates.append(pd.Timestamp(year=year, month=month, day=day))
                    
                    for param in self.PARAMETERS.keys():
                        val = values.get(param, np.nan)
                        data_dict[param].append(val)
                
                except (ValueError, KeyError):
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(data_dict, index=dates)
            df.index.name = 'date'
            df = df.sort_index()
            
            print(f"✓ Downloaded {len(df)} days of data")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            
            # Display summary
            print("\nData summary:")
            print(df.describe())
            
            self.data = df
            
            # Save to CSV
            if save_csv:
                filename = f"nasa_power_{self.location}_{start_year}-{end_year}.csv"
                df.to_csv(filename)
                print(f"\n✓ Saved to {filename}")
            
            return df
        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return None
    
    def interpolate_to_hourly(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Interpolate daily data to hourly.
        
        Args:
            df: DataFrame with daily data
            
        Returns:
            DataFrame with hourly data
        """
        if df is None:
            df = self.data
        
        if df is None:
            print("No data loaded. Call download_daily() first.")
            return None
        
        print(f"\nInterpolating {len(df)} daily values to hourly...")
        
        # Create hourly index
        hourly_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max() + timedelta(hours=23),
            freq='H'
        )
        
        # Reindex daily data to hourly
        df_hourly = df.reindex(df.index)
        
        # Forward fill (use daily value for all 24 hours)
        # Better approach: use solar profile curve
        df_hourly = df_hourly.reindex(hourly_index, method='ffill')
        
        # For solar irradiance, apply diurnal cycle
        if 'ALLSKY_SFC_SW_DWN' in df_hourly.columns:
            df_hourly['ALLSKY_SFC_SW_DWN'] = self._apply_diurnal_cycle(df_hourly['ALLSKY_SFC_SW_DWN'])
        
        print(f"✓ Created {len(df_hourly)} hourly data points")
        
        return df_hourly
    
    @staticmethod
    def _apply_diurnal_cycle(daily_values: pd.Series) -> pd.Series:
        """
        Apply realistic diurnal (daily) cycle to solar irradiance.
        
        Converts daily average to hourly by assuming sinusoidal profile
        peaking at solar noon.
        """
        hourly_values = pd.Series(index=daily_values.index, dtype=float)
        
        for date in daily_values.index.normalize().unique():
            daily_irradiance = daily_values[date].mean()
            
            # Get hours for this day
            day_hours = daily_values[date].index.hour
            
            # Simple sine curve: peaks at 12 noon, zero at 6 AM and 6 PM
            for hour in range(24):
                if 6 <= hour <= 18:  # Daylight hours
                    fraction = np.sin((hour - 6) * np.pi / 12)
                    hourly_irradiance = daily_irradiance * 1.5 * fraction  # 1.5x for peak
                else:
                    hourly_irradiance = 0
                
                hourly_values[pd.Timestamp(date.year, date.month, date.day, hour)] = hourly_irradiance
        
        return hourly_values.clip(lower=0)
    
    def get_annual_data(self, year: int = 2024) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get one year of hourly data for simulation.
        
        Args:
            year: Year to extract
            
        Returns:
            (solar_power, temperature, humidity, precipitation)
        """
        if self.data is None:
            print("No data loaded. Downloading...")
            self.download_daily(start_year=year, end_year=year)
        
        # Interpolate to hourly
        df_hourly = self.interpolate_to_hourly()
        
        if df_hourly is None:
            return None
        
        # Filter to requested year
        df_year = df_hourly[df_hourly.index.year == year]
        
        if len(df_year) == 0:
            print(f"No data for year {year}")
            return None
        
        # Extract arrays
        solar = df_year['ALLSKY_SFC_SW_DWN'].values.astype(np.float32)
        temp = df_year['T2M'].values.astype(np.float32)
        humidity = df_year['RH2M'].values.astype(np.float32)
        precip = df_year['PRECTOTCORR'].values.astype(np.float32)
        
        # Handle NaNs
        solar = np.nan_to_num(solar, nan=0.0)
        temp = np.nan_to_num(temp, nan=25.0)  # Default 25°C
        humidity = np.nan_to_num(humidity, nan=60.0)  # Default 60%
        precip = np.nan_to_num(precip, nan=0.0)
        
        print(f"\nAnnual data summary ({year}):")
        print(f"  Solar irradiance: {solar.min():.1f} - {solar.max():.1f} W/m² (mean: {solar.mean():.1f})")
        print(f"  Temperature: {temp.min():.1f} - {temp.max():.1f} °C (mean: {temp.mean():.1f})")
        print(f"  Humidity: {humidity.min():.1f} - {humidity.max():.1f} % (mean: {humidity.mean():.1f})")
        print(f"  Data points: {len(solar)}")
        
        return solar, temp, humidity, precip


def download_hosur_data(start_year: int = 2024, end_year: int = 2024) -> Dict:
    """
    Convenience function to download Hosur data.
    
    Args:
        start_year: Start year
        end_year: End year
        
    Returns:
        Dictionary with solar, temperature, humidity data
    """
    downloader = NASAPOWERDownloader(location='Hosur', lat=12.74, lon=78.14)
    
    # Download
    df = downloader.download_daily(start_year=start_year, end_year=end_year)
    
    if df is None:
        return None
    
    # Get annual data
    result = {}
    for year in range(start_year, end_year + 1):
        solar, temp, humidity, precip = downloader.get_annual_data(year=year)
        if solar is not None:
            result[year] = {
                'solar': solar,
                'temperature': temp,
                'humidity': humidity,
                'precipitation': precip
            }
    
    return result


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("NASA POWER Data Downloader for Smart Microgrid")
    print("=" * 60)
    
    # Download Hosur data
    print("\nDownloading NASA POWER data for Hosur, Tamil Nadu...")
    data = download_hosur_data(start_year=2024, end_year=2024)
    
    if data and 2024 in data:
        print("\n✓ Data successfully downloaded and processed!")
        print(f"  Solar irradiance shape: {data[2024]['solar'].shape}")
        print(f"  Temperature shape: {data[2024]['temperature'].shape}")
        
        # Save for use in environment
        import pickle
        with open('hosur_2024_data.pkl', 'wb') as f:
            pickle.dump(data[2024], f)
        print(f"  Saved to hosur_2024_data.pkl")
    else:
        print("\n✗ Failed to download data. Check your internet connection.")
        print("  You can manually download from: https://power.larc.nasa.gov/data-access-viewer/")
