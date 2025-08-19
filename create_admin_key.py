# CLI script to create initial admin key
if __name__ == "__main__":
    from app.auth import auth_service
    from app.database import Base, engine

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create admin key
    admin_key = auth_service.create_api_key(
        name="Admin Key",
        permissions="admin",
        rate_limit=1000
    )

    print("=" * 60)
    print("ADMIN API KEY CREATED")
    print("=" * 60)
    print(f"Key: {admin_key}")
    print("Save this key securely - it won't be shown again!")
    print("=" * 60)