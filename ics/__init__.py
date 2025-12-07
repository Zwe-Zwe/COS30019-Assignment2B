"""ICS package: config, routing, prediction, and web app."""

def create_app():
    from ics.web.app import app
    return app

__all__ = ["create_app"]
