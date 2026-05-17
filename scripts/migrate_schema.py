import asyncio
import asyncpg
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

async def migrate():
    # Database connection string
    db_url = os.getenv("DATABASE_URL", "postgresql://lexai:lexai_password@localhost:5433/lexaidb")
    
    # We are connecting locally to the exposed port 5433 by default
    print(f"Connecting to {db_url}...")
    
    try:
        conn = await asyncpg.connect(db_url)
        print("Connected successfully. Running migrations on documents table...")
        
        # Add section_title
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS section_title TEXT;")
        
        # Add is_continuation
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_continuation BOOLEAN DEFAULT FALSE;")
        
        # Add chunk_index
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_index INTEGER;")
        
        # Add total_chunks
        await conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS total_chunks INTEGER;")
        
        print("Migration complete. Columns added successfully.")
        await conn.close()
        
    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == "__main__":
    asyncio.run(migrate())
