# logger.py
# Simple logging utility for adding, reading, editing, and deleting log entries in a text file.
# All log entries are stored in 'log.txt' in the same directory as this script.

import os

LOG_FILE = "log.txt"  # Name of the log file


def add_log(entry):
    """
    Add a new log entry to the log file.
    :param entry: String to be added as a log entry.
    """
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")  # Write the entry followed by a newline


def read_logs():
    """
    Read all log entries from the log file.
    :return: List of log entry strings. Returns an empty list if file does not exist.
    """
    if not os.path.exists(LOG_FILE):
        return []  # Return empty list if log file doesn't exist
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return f.readlines()  # Return all lines as a list


def edit_log(line_number, new_entry):
    """
    Edit a specific log entry by line number (0-based).
    :param line_number: Index of the log entry to edit (0-based).
    :param new_entry: New string to replace the old log entry.
    :return: True if edit was successful, False otherwise.
    """
    logs = read_logs()
    if 0 <= line_number < len(logs):
        logs[line_number] = new_entry + "\n"  # Replace the specified line
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.writelines(logs)  # Write all lines back to the file
        return True
    return False  # Return False if line_number is out of range


def delete_log(line_number):
    """
    Delete a specific log entry by line number (0-based).
    :param line_number: Index of the log entry to delete (0-based).
    :return: True if deletion was successful, False otherwise.
    """
    logs = read_logs()
    if 0 <= line_number < len(logs):
        del logs[line_number]  # Remove the specified line
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.writelines(logs)  # Write remaining lines back to the file
        return True
    return False  # Return False if line_number is out of range


def main():
    """Menu-driven CLI for logging operations."""
    while True:
        print("\n--- Log Menu ---")
        print("1. Add log entry")
        print("2. Read all log entries")
        print("3. Edit a log entry")
        print("4. Delete a log entry")
        print("5. Exit")
        choice = input("Choose an option (1-5): ")
        if choice == "1":
            entry = input("Enter a log entry: ")
            add_log(entry)
            print("Log entry added.")
        elif choice == "2":
            logs = read_logs()
            if logs:
                print("\nAll log entries:")
                for i, log in enumerate(logs):
                    print(f"{i}: {log.strip()}")
            else:
                print("No log entries found.")
        elif choice == "3":
            logs = read_logs()
            if not logs:
                print("No log entries to edit.")
                continue
            for i, log in enumerate(logs):
                print(f"{i}: {log.strip()}")
            try:
                idx = int(input("Enter the line number to edit: "))
                new_entry = input("Enter the new log entry: ")
                if edit_log(idx, new_entry):
                    print("Log entry edited.")
                else:
                    print("Invalid line number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == "4":
            logs = read_logs()
            if not logs:
                print("No log entries to delete.")
                continue
            for i, log in enumerate(logs):
                print(f"{i}: {log.strip()}")
            try:
                idx = int(input("Enter the line number to delete: "))
                if delete_log(idx):
                    print("Log entry deleted.")
                else:
                    print("Invalid line number.")
            except ValueError:
                print("Please enter a valid number.")
        elif choice == "5":
            print("Exiting log program.")
            break
        else:
            print("Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()
