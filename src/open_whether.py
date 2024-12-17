import asyncio
import os
from dataclasses import dataclass

import httpx
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
logfire.configure()

OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')


@dataclass
class MyDeps:
    openweather_api_key: str | None
    http_client: httpx.AsyncClient

weather_agent: Agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt='あなたは親切な日本語のアシスタントです。',
    deps_type=MyDeps,
    retries=2,
)

@weather_agent.tool
async def get_lat_lon(ctx: RunContext[MyDeps], city_name: str) -> list[dict]:
    """与えられた都市名から緯度経度を返す

    引数:
        ctx: コンテキスト
        city_name: 都市名（ローマ字）
    """
    params = {
        'q': city_name,
        'appid': ctx.deps.openweather_api_key,
    }
    response = await ctx.deps.http_client.get(
        "http://api.openweathermap.org/geo/1.0/direct",
        params=params
    )
    response.raise_for_status()

    location_data = response.json()
    if location_data:
        return location_data
    else:
        raise ModelRetry('Could not find the location.')

@weather_agent.tool
async def get_weather_forecast(ctx: RunContext[MyDeps], lat: float, lon: float) -> dict:
    """与えられた緯度経度の5日間（3時間おき）の天気予報を返す

    引数:
        ctx: コンテキスト
        lat: 緯度
        lon: 経度
    """
    params = {
        'lat': lat,
        'lon': lon,
        'appid': ctx.deps.openweather_api_key,
        'units': 'metric',
    }
    response = await ctx.deps.http_client.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params=params
    )
    response.raise_for_status()

    forecast_data = response.json()
    if forecast_data:
        return forecast_data
    else:
        raise ModelRetry('Could not find the weather information.')

async def main(query: str) -> None:
    async with httpx.AsyncClient() as client:
        deps = MyDeps(OPENWEATHER_API_KEY, client)
        result = await weather_agent.run(query, deps=deps)
        print(result.data)


if __name__ == '__main__':
    asyncio.run(main("神戸の明日の天気を知りたい"))