import os
import sys
import subprocess
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

def get_modules():
    modules = []
    for item in sorted(os.listdir(".")):
        if os.path.isdir(item) and item.startswith("UTILS - "):
            modules.append(item)
    return modules

def find_main_script(module_path):
    for file in os.listdir(module_path):
        if file.endswith(".py") and file != "__init__.py":
            return file
    return None

def main():
    while True:
        console.clear()
        console.print(Panel("[bold cyan]Welcome to Learn-Quant Interactive Platform[/bold cyan]\n[white]Select a module to learn and run its demonstration script.[/white]", border_style="cyan"))
        
        modules = get_modules()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=4)
        table.add_column("Category", min_width=20)
        table.add_column("Topic")

        for i, module in enumerate(modules, 1):
            parts = module.replace("UTILS - ", "").split(" - ", 1)
            category = parts[0] if len(parts) > 1 else "General"
            topic = parts[1] if len(parts) > 1 else parts[0]
            table.add_row(str(i), category, topic)

        console.print(table)
        
        console.print("\n[bold yellow]Options:[/bold yellow]")
        console.print("[green]Enter a number to run a module[/green]")
        console.print("[red]q[/red] to quit")
        
        choice = Prompt.ask("Select an option")
        
        if choice.lower() == 'q':
            console.print("[bold green]Goodbye![/bold green]")
            break
            
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(modules):
                selected_module = modules[choice_idx]
                script = find_main_script(selected_module)
                if script:
                    console.print(f"\n[bold cyan]Running {script} in {selected_module}...[/bold cyan]\n")
                    script_path = os.path.join(selected_module, script)
                    # Run the script
                    subprocess.run([sys.executable, script_path], cwd=selected_module)
                    Prompt.ask("\n[bold yellow]Press Enter to return to menu[/bold yellow]")
                else:
                    console.print("[bold red]No Python script found in this module.[/bold red]")
                    Prompt.ask("Press Enter to continue")
            else:
                console.print("[bold red]Invalid selection![/bold red]")
                Prompt.ask("Press Enter to continue")
        except ValueError:
            console.print("[bold red]Please enter a valid number or 'q' to quit.[/bold red]")
            Prompt.ask("Press Enter to continue")

if __name__ == "__main__":
    main()
