"""Entrypoint for the ICS Flask app."""

from ics.web.app import app

if __name__ == "__main__":
    app.run(debug=True)
