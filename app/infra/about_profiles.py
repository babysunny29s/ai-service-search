"""Public about-text of a profile the person already owns.

YouTube and GitHub publish a description a bot can read, and people put their
other accounts there (a Facebook link in the channel description). Facebook,
Instagram, LinkedIn, TikTok and X do not. Age, email and phone stay in the
text when the profile states them.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.urls import profile_of

log = logging.getLogger(__name__)

def public_text(text: str) -> str:
    """About text with whitespace collapsed. Contact details stated there are kept."""
    return " ".join((text or "").split())


def youtube_from_info(info: dict) -> tuple[str, str | None]:
    """Channel description for the model. Links in it are left for the model to copy."""
    description = info.get("description") if isinstance(info.get("description"), str) else ""
    count = info.get("channel_follower_count")
    audience = f"{count} subscribers" if isinstance(count, int) and count > 0 else None
    return public_text("\n".join(p for p in (description, audience or "") if p)), audience


def github_from_payload(data: dict) -> tuple[str, str | None]:
    bio = data.get("bio") if isinstance(data.get("bio"), str) else ""
    lines = [bio]
    company = data.get("company")
    location = data.get("location")
    if isinstance(company, str) and company.strip():
        lines.append(f"Company: {company.strip().lstrip('@')}")
    if isinstance(location, str) and location.strip():
        lines.append(f"Location: {location.strip()}")
    followers = data.get("followers")
    audience = f"{followers} followers" if isinstance(followers, int) and followers > 0 else None
    if audience:
        lines.append(audience)
    blog = data.get("blog") if isinstance(data.get("blog"), str) else ""
    blog = blog.strip()
    if blog and not blog.startswith(("http://", "https://")):
        blog = "https://" + blog
    if blog.startswith("http"):
        lines.append(f"Website: {blog}")
    twitter = data.get("twitter_username")
    if isinstance(twitter, str) and twitter.strip():
        lines.append(f"Twitter: https://x.com/{twitter.strip().lstrip('@')}")
    email = data.get("email")
    if isinstance(email, str) and "@" in email:
        lines.append(f"Email: {email.strip()}")
    return public_text("\n".join(lines)), audience


def _youtube_info(url: str) -> dict:
    import yt_dlp

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "playlistend": 1,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    return info if isinstance(info, dict) else {}


async def youtube_about(url: str) -> tuple[str, str | None]:
    """Channel description for the model to read. No video download."""
    if not (profile_of(url) and profile_of(url).platform == "youtube"):
        return "", None
    try:
        info = await asyncio.to_thread(_youtube_info, url)
    except Exception as e:
        log.warning("youtube about %s failed: %s", url, e)
        return "", None
    return youtube_from_info(info)


async def github_about(url: str, timeout_s: float, user_agent: str) -> tuple[str, str | None]:
    """Bio, company, location and written links from the public GitHub user API."""
    profile = profile_of(url)
    if not profile or profile.platform != "github":
        return "", None
    try:
        async with httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": user_agent, "Accept": "application/vnd.github+json"}) as client:
            resp = await client.get(f"https://api.github.com/users/{profile.handle}")
        if resp.status_code != 200:
            return "", None
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("github about %s failed: %s", url, e)
        return "", None
    return github_from_payload(data if isinstance(data, dict) else {})
