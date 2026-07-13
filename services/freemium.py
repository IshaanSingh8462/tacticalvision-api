import os
from datetime import datetime, timezone, timedelta
from supabase import Client

async def check_and_enforce_limits(
    user_id: str,
    video_duration_seconds: float,
    supabase: Client,
) -> dict:
    """
    Check whether a user is allowed to submit a processing request.

    Returns a dict with:
    allowed: bool — whether processing should proceed
    reason: str — human-readable reason if not allowed
    tier: str — "free", "pro", or "team"

    This runs BEFORE creating a job row, so denied requests never
    appear in the jobs table.
    """

    # Fetch subscription tier
    sub_response = supabase.table("subscriptions") \
        .select("tier") \
        .eq("user_id", user_id) \
        .single() \
        .execute()

    if not sub_response.data:
        return {"allowed": False, "reason": "No subscription found", "tier": "free"}

    tier = sub_response.data["tier"]
    is_paid = tier in ("pro", "team")

    # Fetch usage limits
    usage_response = supabase.table("usage_limits") \
        .select("*") \
        .eq("user_id", user_id) \
        .single() \
        .execute()

    if not usage_response.data:
        return {"allowed": False, "reason": "No usage record found", "tier": tier}

    usage = usage_response.data

    # Check if weekly counters need resetting
    week_reset_at = datetime.fromisoformat(usage["week_reset_at"])
    now = datetime.now(timezone.utc)
    if week_reset_at.tzinfo is None:
        week_reset_at = week_reset_at.replace(tzinfo=timezone.utc)

    if (now - week_reset_at).days >= 7:
        # Reset the counters
        supabase.table("usage_limits") \
            .update({
                "ai_analyses_used_this_week": 0,
                "uploads_this_week": 0,
                "week_reset_at": now.isoformat(),
            }) \
            .eq("user_id", user_id) \
            .execute()
        # Refresh local copy
        usage["ai_analyses_used_this_week"] = 0
        usage["uploads_this_week"] = 0


    # ── Enforce video length limit ────────────────────────────────────────────
    # Free: max 10 minutes. Paid: unlimited.
    if not is_paid and video_duration_seconds > 600:
        return {
            "allowed": False,
            "reason": f"Free tier limit is 10 minutes. Your video is "
                    f"{video_duration_seconds/60:.1f} minutes. Upgrade to Pro for unlimited.",
            "tier": tier,
        }

    # ── Enforce weekly upload quota ────────────────────────────────────────────
    if not is_paid:
        uploads_used = usage.get("uploads_this_week", 0)
        uploads_limit = usage.get("uploads_limit_per_week", 1)
        if uploads_used >= uploads_limit:
            return {
                "allowed": False,
                "reason": f"Free tier allows {uploads_limit} upload per week. "
                        f"You have used {uploads_used}. Upgrade to Pro for unlimited.",
                "tier": tier,
            }

    return {"allowed": True, "reason": "", "tier": tier}


async def increment_upload_count(user_id: str, supabase: Client):
    """Increment the weekly upload counter after a job is accepted."""
    supabase.rpc("increment_uploads", {"p_user_id": user_id}).execute()
    # We use a Postgres RPC function for atomic increment to avoid
    # race conditions if two requests come in simultaneously.
    # Create this function in Supabase SQL Editor:
    #
    # CREATE OR REPLACE FUNCTION increment_uploads(p_user_id UUID)
    # RETURNS void AS $$
    #   UPDATE usage_limits
    #   SET uploads_this_week = uploads_this_week + 1
    #   WHERE user_id = p_user_id;
    # $$ LANGUAGE sql SECURITY DEFINER;

