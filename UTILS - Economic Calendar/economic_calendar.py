# -----------------------------
# Economic Calendar Simulator Utility (NO API)
# -----------------------------
# This script lets you simulate economic events and their impact on a sample portfolio.
# All data is managed locally (no APIs, no real market data).
# All code is commented for beginners to learn Python and finance.
# -----------------------------

import json
import os
from datetime import datetime

EVENTS_FILE = 'events.json'  # Default file for saving/loading events

# Event class: represents a single economic event
default_impact_levels = ['Low', 'Medium', 'High']

class EconomicEvent:
    def __init__(self, date, event_type, description, impact):
        self.date = date  # YYYY-MM-DD
        self.event_type = event_type
        self.description = description
        self.impact = impact  # 'Low', 'Medium', 'High'

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return EconomicEvent(d['date'], d['event_type'], d['description'], d['impact'])

# Helper to get a date from user input
def input_date(prompt):
    while True:
        s = input(prompt)
        try:
            datetime.strptime(s, '%Y-%m-%d')
            return s
        except ValueError:
            print('Invalid date format. Please use YYYY-MM-DD.')

# Print the main menu for the CLI
def print_menu():
    print("\nEconomic Calendar Simulator Menu:")
    print("1. Add event")
    print("2. Edit event")
    print("3. Remove event")
    print("4. View calendar")
    print("5. Simulate event impact")
    print("6. Exit")

# Add a new event
def add_event(events):
    date = input_date("Enter event date (YYYY-MM-DD): ")
    event_type = input("Enter event type: ")
    description = input("Enter description: ")
    impact = input(f"Enter expected impact ({'/'.join(default_impact_levels)}): ")
    if impact not in default_impact_levels:
        print("Invalid impact level. Defaulting to 'Medium'.")
        impact = 'Medium'
    events.append(EconomicEvent(date, event_type, description, impact))
    print("Event added!")

# Find event by date and type
def find_event(events, date, event_type):
    for i, e in enumerate(events):
        if e.date == date and e.event_type.lower() == event_type.lower():
            return i
    return -1

# Edit an existing event
def edit_event(events):
    date = input_date("Enter event date to edit (YYYY-MM-DD): ")
    event_type = input("Enter event type to edit: ")
    idx = find_event(events, date, event_type)
    if idx == -1:
        print("Event not found.")
        return
    e = events[idx]
    print(f"Editing {e.event_type} on {e.date} ({e.impact}): {e.description}")
    e.description = input("Enter new description: ")
    impact = input(f"Enter new expected impact ({'/'.join(default_impact_levels)}): ")
    if impact in default_impact_levels:
        e.impact = impact
    print("Event updated!")

# Remove an event
def remove_event(events):
    date = input_date("Enter event date to remove (YYYY-MM-DD): ")
    event_type = input("Enter event type to remove: ")
    idx = find_event(events, date, event_type)
    if idx == -1:
        print("Event not found.")
        return
    del events[idx]
    print("Event removed!")

# View all events in date order
def view_calendar(events):
    if not events:
        print("No events scheduled.")
        return
    print("\nUpcoming Economic Events:")
    for e in sorted(events, key=lambda x: x.date):
        print(f"{e.date} | {e.event_type:<20} | {e.impact:<6} | {e.description}")

# Simulate the impact of an event on a sample portfolio
def simulate_event_impact(events):
    date = input_date("Enter event date to simulate (YYYY-MM-DD): ")
    event_type = input("Enter event type to simulate: ")
    idx = find_event(events, date, event_type)
    if idx == -1:
        print("Event not found.")
        return
    e = events[idx]
    print(f"Simulating impact of {e.event_type} on {e.date} ({e.impact}): {e.description}")
    # For learning: let user input a sample portfolio value and see a random impact
    value = float(input("Enter your sample portfolio value: "))
    if e.impact == 'High':
        change = value * 0.05  # 5% move
    elif e.impact == 'Medium':
        change = value * 0.02  # 2% move
    else:
        change = value * 0.005  # 0.5% move
    direction = input("Did the event have a positive or negative impact? (+/-): ")
    if direction == '+':
        new_value = value + change
    else:
        new_value = value - change
    print(f"Your portfolio changed from {value:.2f} to {new_value:.2f} due to this event.")

# Save events to a file
def save_events(events):
    fname = EVENTS_FILE
    data = [e.to_dict() for e in events]
    with open(fname, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Events saved to {fname}.")

# Load events from a file
def load_events():
    fname = EVENTS_FILE
    if not os.path.exists(fname):
        return []
    with open(fname, 'r') as f:
        data = json.load(f)
    events = [EconomicEvent.from_dict(d) for d in data]
    return events

# Main workflow for the CLI
def main():
    print("""
====================================
Welcome to the Economic Calendar Simulator!
This tool helps you learn Python and macro events by simulating economic news and their impact.
- No APIs or real market data are used.
- All code is commented for beginners.
====================================
""")
    events = load_events()
    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1':
            add_event(events)
            save_events(events)
        elif choice == '2':
            edit_event(events)
            save_events(events)
        elif choice == '3':
            remove_event(events)
            save_events(events)
        elif choice == '4':
            view_calendar(events)
        elif choice == '5':
            simulate_event_impact(events)
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
