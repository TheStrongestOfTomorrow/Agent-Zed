"""Interactive `zed-cli` Terminal Interface with Real-time CEO Chat & Jobs Dashboard."""

import sys
import os
import asyncio
import time
from typing import Optional, List
from agent_zed.core.engine import MoAEngine
from agent_zed.core.jobs import JobStatus, JobPriority
from agent_zed.core.reasoning import ReasoningEngine
from agent_zed.agents.definitions import get_13_agent_personas

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class ZedCLI:
    """Main CLI Application for Agent-Zed."""

    def __init__(self):
        self.engine = MoAEngine()
        self.reasoning = ReasoningEngine()
        self.personas = get_13_agent_personas()
        self.console = Console() if RICH_AVAILABLE else None

    def display_banner(self):
        if RICH_AVAILABLE:
            banner_text = Text(
                "    ___                    __      _____          __\n"
                "   /   |  ____  ___  ____  / /_    /__  /  ___  __/ /\n"
                "  / /| | / __ \\/ _ \\/ __ \\/ __/      / /  / _ \\/ __/ \n"
                " / ___ |/ /_/ /  __/ / / / /_       / /__/  __/ /_   \n"
                "/_/  |_/\\__, /\\___/_/ /_/\\__/      /____/\\___/\\__/   \n"
                "       /____/   13-Agent MoA Coding AI Framework     \n",
                style="bold cyan"
            )
            panel = Panel(
                banner_text,
                title="[bold green]AGENT-ZED CLI v1.0.0[/bold green]",
                subtitle="[yellow]Optimized for Android / Termux (4GB RAM)[/yellow]",
                border_style="cyan"
            )
            self.console.print(panel)
        else:
            print("=========================================================")
            print("                AGENT-ZED CLI v1.0.0                    ")
            print("         13-Agent MoA Coding AI Framework               ")
            print("=========================================================")

    def display_jobs_dashboard(self):
        """Display live jobs dashboard managed by the CEO."""
        summary = self.engine.job_manager.get_summary_report()
        if RICH_AVAILABLE:
            table = Table(title="💼 CEO Workforce Jobs Dashboard", border_style="cyan")
            table.add_column("Job ID", style="bold yellow", width=12)
            table.add_column("Title", style="cyan")
            table.add_column("Status", style="bold green")
            table.add_column("Progress", style="magenta")
            table.add_column("Assigned Agents", style="blue")

            jobs = self.engine.job_manager.list_jobs()
            if not jobs:
                table.add_row("N/A", "No active workforce jobs", "IDLE", "0%", "CEO")
            else:
                for job in jobs:
                    status_color = "green" if job.status == JobStatus.COMPLETED else ("yellow" if job.status == JobStatus.RUNNING else "white")
                    table.add_row(
                        job.id,
                        job.title,
                        f"[{status_color}]{job.status.value}[/{status_color}]",
                        f"{job.progress:.0f}%",
                        ", ".join(job.assigned_agents)
                    )
            self.console.print(table)
        else:
            print(summary)

    async def ceo_chat_loop(self):
        """Interactive real-time chat with the CEO in zed-cli."""
        self.display_banner()

        if RICH_AVAILABLE:
            self.console.print(
                Panel(
                    "[bold yellow]CEO (Lead):[/bold yellow] Welcome! I'm the CEO of Agent-Zed. "
                    "I lead our 13-agent workforce. Talk to me naturally or ask any coding task!\n\n"
                    "Commands:\n"
                    "  [bold cyan]/jobs[/bold cyan]     - View live workforce jobs dashboard\n"
                    "  [bold cyan]/agents[/bold cyan]   - View 13 agent corporate hierarchy & status\n"
                    "  [bold cyan]/quit[/bold cyan]     - Exit zed-cli",
                    title="[bold green]Interactive CEO Session[/bold green]",
                    border_style="green"
                )
            )
        else:
            print("\nCEO (Lead): Welcome! Talk to me or type /jobs, /agents, or /quit.")

        while True:
            try:
                if RICH_AVAILABLE:
                    user_input = Prompt.ask("\n[bold magenta]You[/bold magenta]")
                else:
                    user_input = input("\nYou: ")

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.lower() in ["/quit", "exit", "quit"]:
                    if RICH_AVAILABLE:
                        self.console.print("[bold yellow]CEO (Lead):[/bold yellow] Goodbye! Keep building amazing software.")
                    else:
                        print("CEO (Lead): Goodbye!")
                    break

                elif user_input.lower() == "/jobs":
                    self.display_jobs_dashboard()

                elif user_input.lower() == "/agents":
                    self.display_agents_hierarchy()

                else:
                    # Process via reasoning engine
                    reply_or_task, is_coding_task = self.reasoning.process_input(user_input)
                    if is_coding_task:
                        if user_input.startswith("/solve "):
                            user_input = user_input[7:]
                        await self.run_solve_task(user_input)
                    else:
                        if RICH_AVAILABLE:
                            self.console.print(f"[bold yellow]CEO (Lead):[/bold yellow] {reply_or_task}")
                        else:
                            print(f"CEO (Lead): {reply_or_task}")

            except (KeyboardInterrupt, EOFError):
                print("\nExiting Agent-Zed CLI.")
                break

    def display_agents_hierarchy(self):
        """Display the 13 agent personas."""
        if RICH_AVAILABLE:
            table = Table(title="🏢 Agent-Zed 13 Corporate Personas", border_style="green")
            table.add_column("Persona Name", style="bold yellow")
            table.add_column("Title / Role", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Personality Brief", style="white")

            for key, agent in self.personas.items():
                table.add_row(agent.name, agent.title, agent.role_type, agent.personality)
            self.console.print(table)
        else:
            print("\n=== AGENT-ZED 13 CORPORATE PERSONAS ===")
            for key, agent in self.personas.items():
                print(f"- {agent.name} ({agent.title}) [{agent.role_type}]: {agent.personality}")

    async def run_solve_task(self, problem: str):
        """Execute MoA coding task with live thinking visualization."""
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold green]CEO (Lead):[/bold green] Creating background job and delegating problem to MoA workforce: [cyan]'{problem}'[/cyan]")
        else:
            print(f"\nCEO (Lead): Delegating problem to MoA workforce: '{problem}'")

        async def status_cb(agent_name: str, status_msg: str):
            if RICH_AVAILABLE:
                self.console.print(f"  └─ [[bold yellow]{agent_name}[/bold yellow]]: {status_msg}")
            else:
                print(f"  └─ [{agent_name}]: {status_msg}")

        res = await self.engine.solve_coding_problem(problem, status_callback=status_cb)

        if RICH_AVAILABLE:
            self.console.print("\n[bold cyan]=== FINAL SYNTHESIZED SOLUTION (Spokesperson) ===[/bold cyan]")
            self.console.print(Panel(res["code"], title="[bold green]Verified Code Deliverable[/bold green]", border_style="green"))
            self.console.print(f"[bold yellow]CEO (Lead):[/bold yellow] Job Completed in [green]{res['elapsed_seconds']}s[/green] across {res['rounds_used']} MoA round(s)!")
        else:
            print("\n=== FINAL SYNTHESIZED SOLUTION ===")
            print(res["code"])
            print(f"CEO (Lead): Job Completed in {res['elapsed_seconds']}s!")

def main():
    cli = ZedCLI()
    asyncio.run(cli.ceo_chat_loop())

if __name__ == "__main__":
    main()
