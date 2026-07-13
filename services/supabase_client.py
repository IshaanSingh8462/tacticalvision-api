import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
# load_dotenv() reads your .env file and sets environment variables.
# On Render, environment variables are set in the dashboard instead,
# but load_dotenv() safely no-ops when the .env file isn't present.

def get_supabase() -> Client:
    """
    Create a Supabase client using the SERVICE ROLE key.

    The service role key bypasses Row Level Security entirely.
    This is intentional — the FastAPI server writes data on behalf of users
    after verifying their identity itself. The service role key must NEVER
    be exposed to the frontend.

    This function is called fresh per request (not a module-level singleton)
    to avoid connection state issues in async contexts.
    """
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)
