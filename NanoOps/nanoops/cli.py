"""NanoOps CLI using Typer."""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .config import load_config
from .exceptions import NanoOpsError, SessionExistsError, SessionNotFoundError
from .orchestrator import SessionOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress noisy httpx request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer(
    name="nanoctrl",
    help="NanoInfra Operations CLI - Orchestrate distributed LLM inference",
    add_completion=False,
)

console = Console()


def get_orchestrator(
    redis_url: Optional[str] = None,
    ray_address: Optional[str] = None,
    nanoctrl_address: Optional[str] = None,
    config_file: Optional[str] = None,
    session_id: Optional[str] = None,
) -> SessionOrchestrator:
    """Create session orchestrator with config precedence.

    If session_id is provided, will read connection settings from the session's
    saved config in Redis (if it exists).
    """
    config = load_config(config_file)

    # Try to read from session config if session_id provided.
    # Only use session-saved values for fields NOT already set by env vars,
    # so that environment variables (cloud-native) always win.
    if session_id and not (redis_url or ray_address or nanoctrl_address):
        try:
            from .redis_client import RedisClient

            temp_redis = RedisClient(config.redis_url)
            if temp_redis.session_exists(session_id):
                session_config = temp_redis.get_session_config(session_id)
                if "redis_url" in session_config and not os.getenv(
                    "NANOCTRL_REDIS_URL"
                ):
                    config.redis_url = session_config["redis_url"]
                if "ray_address" in session_config and not os.getenv("RAY_ADDRESS"):
                    config.ray_address = session_config["ray_address"]
                if "nanoctrl_address" in session_config and not os.getenv(
                    "NANOCTRL_ADDRESS"
                ):
                    config.nanoctrl_address = session_config["nanoctrl_address"]
                logger.debug(f"Using connection settings from session '{session_id}'")
        except Exception as e:
            logger.debug(f"Could not read session config: {e}")

    # Override from CLI args (highest priority)
    if redis_url:
        config.redis_url = redis_url
    if ray_address:
        config.ray_address = ray_address
    if nanoctrl_address:
        config.nanoctrl_address = nanoctrl_address

    return SessionOrchestrator(
        config.redis_url,
        config.ray_address,
        config.nanoctrl_address,
    )


@app.command()
def create(
    session_id: str = typer.Option(
        ..., "--session-id", "-s", help="Unique session identifier"
    ),
    no_start_nanoctrl: bool = typer.Option(False, help="Don't auto-start NanoCtrl"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    """Create a new session and register it in Redis.

    Connection settings are read from environment variables:

        NANOCTRL_REDIS_URL  - Redis connection URL  (default: redis://localhost:6379)
        RAY_ADDRESS         - Ray dashboard address  (default: http://localhost:8265)
        NANOCTRL_ADDRESS    - NanoCtrl HTTP address   (default: http://localhost:3000)

    Example:

        export NANOCTRL_REDIS_URL=redis://10.0.0.1:6379
        export RAY_ADDRESS=http://10.0.0.1:8265
        export NANOCTRL_ADDRESS=http://10.0.0.1:3000
        nanoctrl create --session-id demo2
    """
    try:
        orch = get_orchestrator(config_file=config_file)

        session_info = orch.start_session(
            session_id,
            ensure_nanoctrl=not no_start_nanoctrl,
        )

        config = load_config(config_file)

        console.rule(f"[bold green]Session Created[/bold green]")
        rprint(f"\n  Session:  [bold cyan]{session_id}[/bold cyan]")
        rprint(f"  Redis:    [cyan]{config.redis_url}[/cyan]")
        rprint(f"  Ray:      [cyan]{config.ray_address}[/cyan]")
        rprint(f"  NanoCtrl: [cyan]{session_info['nanoctrl_address']}[/cyan]")
        rprint("")
        console.rule("[dim]Next Steps[/dim]")
        rprint(f"\n  [bold]Attach to the session shell:[/bold]")
        rprint(f"    [green]nanoctrl attach {session_id}[/green]\n")
        rprint(f"  [dim]Or run commands directly with --session-id:[/dim]")
        rprint(
            f"    [dim]nanoctrl set --session-id {session_id} --model <path>[/dim]\n"
        )

    except SessionExistsError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print(
            f"\n[yellow]Tip:[/yellow] Use 'nanoctrl list' to see existing sessions"
        )
        console.print(
            f"[yellow]     [/yellow] Or attach to it: [green]nanoctrl attach {session_id}[/green]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _detect_parent_shell() -> str:
    """Detect the actual shell that invoked this process.

    On Linux, reads /proc/<ppid>/exe to find the real executable
    of the parent process.  Falls back to $SHELL, then /bin/bash.
    """
    # 1. Try /proc on Linux – most reliable
    try:
        parent_exe = os.readlink(f"/proc/{os.getppid()}/exe")
        if os.path.basename(parent_exe) in ("zsh", "bash", "fish"):
            return parent_exe
    except OSError:
        pass

    # 2. Fall back to $SHELL env var
    return os.getenv("SHELL", "/bin/bash")


def _enter_session_shell(session_id: str, config) -> None:
    """Spawn an interactive sub-shell with session environment variables.

    Sets NANOCTRL_SESSION and connection env vars, and prepends the
    session name to the shell prompt so every subsequent `nanoctrl`
    command automatically uses this session.
    """
    shell = _detect_parent_shell()

    # Build child environment
    env = os.environ.copy()
    env["NANOCTRL_SESSION"] = session_id
    env["NANOCTRL_SCOPE"] = session_id
    env["NANOCTRL_REDIS_URL"] = config.redis_url
    env["NANOCTRL_ADDRESS"] = config.nanoctrl_address
    env["RAY_ADDRESS"] = config.ray_address

    console.rule(f"[bold cyan]nanoctrl/{session_id}[/bold cyan]")
    rprint(f"\n  [bold]Attached to session [cyan]'{session_id}'[/cyan][/bold]")
    rprint("  [dim]All nanoctrl commands now use this session automatically.[/dim]")
    rprint("  [dim]Type 'exit' or Ctrl-D to detach.[/dim]\n")

    tmpdir = None
    tmpfile = None

    try:
        if "zsh" in shell:
            # For zsh: use ZDOTDIR to inject our config while preserving
            # the full original zsh environment (oh-my-zsh, themes, etc.)
            tmpdir = tempfile.mkdtemp(prefix="nanoctrl_")
            original_zdotdir = os.getenv("ZDOTDIR", os.path.expanduser("~"))

            # .zshenv: source original .zshenv by full path but keep
            # ZDOTDIR pointing to tmpdir so zsh reads OUR .zshrc next.
            zshenv_content = (
                f"# Source original .zshenv (keep ZDOTDIR as tmpdir)\n"
                f'[ -f "{original_zdotdir}/.zshenv" ] && source "{original_zdotdir}/.zshenv"\n'
            )
            with open(os.path.join(tmpdir, ".zshenv"), "w") as f:
                f.write(zshenv_content)

            # .zshrc: source the original .zshrc (oh-my-zsh, plugins,
            # themes) then add a precmd hook to prepend session info
            # to the prompt. Using add-zsh-hook ensures we don't
            # clobber oh-my-zsh's own precmd hooks.
            zshrc_content = (
                f"# Source original zshrc (oh-my-zsh, themes, plugins)\n"
                f'[ -f "{original_zdotdir}/.zshrc" ] && source "{original_zdotdir}/.zshrc"\n'
                f"\n"
                f"# Prepend nanoctrl session indicator via precmd hook\n"
                f"_nanoctrl_precmd() {{\n"
                f'    PROMPT="${{PROMPT#\\(nanoctrl/*\\) }}"\n'
                f'    PROMPT="(nanoctrl/{session_id}) $PROMPT"\n'
                f"}}\n"
                f"autoload -Uz add-zsh-hook\n"
                f"add-zsh-hook precmd _nanoctrl_precmd\n"
            )
            with open(os.path.join(tmpdir, ".zshrc"), "w") as f:
                f.write(zshrc_content)

            env["ZDOTDIR"] = tmpdir
            shell_args = [shell, "-i"]
        else:
            # For bash: use --rcfile to inject prompt
            tmpfile = tempfile.NamedTemporaryFile(
                mode="w",
                prefix="nanoctrl_",
                suffix=".sh",
                delete=False,
            )
            tmpfile.write(
                "# Source original bash config\n"
                'if [ -f "$HOME/.bashrc" ]; then\n'
                '    source "$HOME/.bashrc"\n'
                "fi\n"
                "\n"
                "# NanoCtrl session prompt\n"
                f'PS1="(nanoctrl/{session_id}) $PS1"\n'
            )
            tmpfile.close()
            shell_args = [shell, "--rcfile", tmpfile.name, "-i"]

        subprocess.run(shell_args, env=env)

    except KeyboardInterrupt:
        pass
    finally:
        # Clean up temp files
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if tmpfile:
            try:
                os.unlink(tmpfile.name)
            except OSError:
                pass

    rprint("")
    console.rule(f"[yellow]Detached from [cyan]'{session_id}'[/cyan][/yellow]")
    rprint(
        f"\n  [dim]Session is still active. Re-attach:[/dim] [green]nanoctrl attach {session_id}[/green]"
    )
    rprint(
        f"  [dim]Stop session:[/dim]                       [dim]nanoctrl stop --session-id {session_id}[/dim]\n"
    )


def _get_session_id(session_id: Optional[str]) -> str:
    """Get session ID from argument or environment."""
    if session_id:
        return session_id

    # Try to get from environment
    env_session = os.getenv("NANOCTRL_SESSION")
    if env_session:
        return env_session

    # No session specified
    console.print("[red]Error:[/red] No session specified")
    console.print("[yellow]Either:[/yellow]")
    console.print("  1. Use [green]--session-id[/green] flag")
    console.print(
        "  2. Attach to a session: [green]nanoctrl attach <session-id>[/green]"
    )
    raise typer.Exit(1)


@app.command(name="set")
def set_model(
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID (or use active session)"
    ),
    model: str = typer.Option(
        ..., "--model", "-m", help="Model path or HuggingFace ID"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    """Set the model for the session.

    Parallelism is configured per-component at deploy time:

        nanoctrl deploy prefill --attention-tp 1 --attention-dp 8 --ffn-ep 8
        nanoctrl deploy decode  --attention-tp 1 --attention-dp 8 --ffn-ep 8
    """
    try:
        session_id = _get_session_id(session_id)
        orch = get_orchestrator(config_file=config_file, session_id=session_id)

        orch.set_model_config(session_id, model)

        rprint(f"[green]✓[/green] Model set!")
        rprint(f"  Model: [cyan]{model}[/cyan]")
        rprint(f"\n[bold]Next:[/bold] Deploy components:")
        rprint(f"  [green]nanoctrl deploy route[/green]")
        rprint(
            f"  [green]nanoctrl deploy prefill --attention-tp 1 --attention-dp 8 --ffn-ep 8[/green]"
        )
        rprint(
            f"  [green]nanoctrl deploy decode  --attention-tp 1 --attention-dp 8 --ffn-ep 8[/green]"
        )

    except SessionNotFoundError as e:
        console.print(f"[red]Session not found:[/red] {e}")
        console.print(
            f"\n[yellow]Tip:[/yellow] Create session with [green]nanoctrl create --session-id {session_id}[/green]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def deploy(
    component_type: str = typer.Argument(
        ..., help="Component type: route | prefill | decode"
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID (or use active session)"
    ),
    # Parallelism (aligned with NanoDeploy Config fields)
    attention_tp: int = typer.Option(
        1, "--attention-tp", help="Attention tensor parallelism"
    ),
    attention_sp: int = typer.Option(
        1, "--attention-sp", help="Attention sequence parallelism"
    ),
    attention_dp: int = typer.Option(
        1, "--attention-dp", help="Attention data parallelism"
    ),
    ffn_tp: int = typer.Option(1, "--ffn-tp", help="FFN tensor parallelism"),
    ffn_ep: int = typer.Option(1, "--ffn-ep", help="FFN expert parallelism"),
    ffn_dp: int = typer.Option(1, "--ffn-dp", help="FFN data parallelism"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    """Deploy a component (route, prefill, or decode) via Ray job.

    Parallelism flags map 1:1 to NanoDeploy Config fields and are
    only meaningful for engine components (prefill / decode).

    Examples:

        nanoctrl deploy route
        nanoctrl deploy prefill --attention-tp 1 --attention-dp 8 --ffn-ep 8
        nanoctrl deploy decode  --attention-tp 1 --attention-dp 8 --ffn-ep 8
    """
    if component_type not in ["route", "prefill", "decode"]:
        console.print(f"[red]Error:[/red] Invalid component type: {component_type}")
        console.print("[yellow]Valid types:[/yellow] route, prefill, decode")
        raise typer.Exit(1)

    try:
        session_id = _get_session_id(session_id)
        orch = get_orchestrator(config_file=config_file, session_id=session_id)

        # Build parallelism overrides (only for engines)
        overrides = {}
        if component_type in ["prefill", "decode"]:
            overrides = {
                "attention_tp": attention_tp,
                "attention_sp": attention_sp,
                "attention_dp": attention_dp,
                "ffn_tp": ffn_tp,
                "ffn_ep": ffn_ep,
                "ffn_dp": ffn_dp,
            }

        with console.status(f"[cyan]Deploying {component_type}...[/cyan]"):
            job_id = orch.spawn_component(session_id, component_type, **overrides)

        num_gpus = attention_tp * attention_sp * attention_dp
        rprint(f"[green]✓[/green] {component_type.capitalize()} deployed!")
        rprint(f"  Ray Job ID: [cyan]{job_id}[/cyan]")
        if component_type in ["prefill", "decode"]:
            rprint(
                f"  GPUs:       [cyan]{num_gpus}[/cyan] (TP={attention_tp}, SP={attention_sp}, DP={attention_dp})"
            )
        rprint(f"\n[dim]Monitor with:[/dim] nanoctrl job logs {job_id}")

    except SessionNotFoundError as e:
        console.print(f"[red]Session not found:[/red] {e}")
        raise typer.Exit(1)
    except NanoOpsError as e:
        console.print(f"[red]Deploy failed:[/red] {e}")
        console.print("\n[yellow]Debug:[/yellow]")
        console.print("  - Check Ray cluster: [cyan]ray status[/cyan]")
        console.print(
            f"  - Check logs: [cyan]nanoctrl job logs {session_id}_{component_type}_*[/cyan]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _format_job_status(status: str) -> str:
    """Colorize a Ray job status string."""
    status_colors = {
        "PENDING": "[yellow]PENDING[/yellow]",
        "RUNNING": "[green]RUNNING[/green]",
        "SUCCEEDED": "[dim]SUCCEEDED[/dim]",
        "FAILED": "[red]FAILED[/red]",
        "STOPPED": "[red]STOPPED[/red]",
    }
    return status_colors.get(status, f"[dim]{status}[/dim]")


def _format_elapsed(spawned_at: int) -> str:
    """Format elapsed time since spawn."""
    elapsed = int(time.time()) - spawned_at
    if elapsed < 60:
        return f"{elapsed}s"
    elif elapsed < 3600:
        return f"{elapsed // 60}m{elapsed % 60}s"
    else:
        return f"{elapsed // 3600}h{(elapsed % 3600) // 60}m"


@app.command()
def status(
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID (or use active session)"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    """Show detailed status of the session and all its components."""
    try:
        session_id = _get_session_id(session_id)
        orch = get_orchestrator(config_file=config_file, session_id=session_id)

        # --- Session Info ---
        session_config = orch.redis.get_session_config(session_id)

        console.rule(f"[bold cyan]Session: {session_id}[/bold cyan]")

        model_path = session_config.get("model_path", "-")
        model_display = os.path.basename(model_path) if model_path != "-" else "-"
        session_status = session_config.get("status", "unknown")

        rprint(f"\n  Status:   {_format_job_status(session_status.upper())}")
        rprint(f"  Model:    [cyan]{model_display}[/cyan]")
        rprint(f"  Redis:    [dim]{orch.redis.url}[/dim]")
        rprint(f"  Ray:      [dim]{orch.ray.address}[/dim]")
        rprint(f"  NanoCtrl: [dim]{orch.nanoctrl.address}[/dim]")

        # --- Components Table ---
        components = orch.redis.get_session_components(session_id)

        total_components = sum(len(v) for v in components.values())
        if total_components == 0:
            rprint("\n  [yellow]No components deployed yet.[/yellow]")
            rprint(
                f"  [dim]Deploy with:[/dim] [green]nanoctrl deploy <route|prefill|decode>[/green]\n"
            )
            return

        # Query live Ray job status and NanoCtrl engines
        try:
            nanoctrl_prefill = orch.nanoctrl.list_engines(session_id, role="prefill")
        except Exception:
            nanoctrl_prefill = []
        try:
            nanoctrl_decode = orch.nanoctrl.list_engines(session_id, role="decode")
        except Exception:
            nanoctrl_decode = []

        nanoctrl_counts = {
            "prefill": len(nanoctrl_prefill),
            "decode": len(nanoctrl_decode),
        }
        # Track which roles have been "accounted for" so we can mark
        # individual rows as registered (1-to-1 is approximate when
        # we can't map Ray job IDs to NanoCtrl UUIDs).
        nanoctrl_remaining = dict(nanoctrl_counts)

        # Fetch live Ray job status for all components (one pass)
        job_statuses = {}  # job_id -> status string
        for comp_type in ["route", "prefill", "decode"]:
            for comp in components.get(comp_type, []):
                job_id = comp.get("ray_job_id")
                if job_id and job_id not in job_statuses:
                    try:
                        job_statuses[job_id] = orch.ray.get_job_status(job_id)
                    except Exception:
                        job_statuses[job_id] = "UNKNOWN"

        rprint("")
        table = Table(
            title="Components",
            show_lines=False,
            pad_edge=True,
            expand=False,
        )
        table.add_column("Type", style="bold")
        table.add_column("Ray Job ID", style="cyan")
        table.add_column("Ray Status")
        table.add_column("Registered")
        table.add_column("Uptime", justify="right")

        for comp_type in ["route", "prefill", "decode"]:
            for comp in components.get(comp_type, []):
                job_id = comp.get("ray_job_id", "-")

                ray_status = job_statuses.get(job_id, "-")

                # NanoCtrl registration (approximate: count-based since
                # we can't map Ray job IDs to NanoCtrl engine UUIDs)
                if comp_type in ("prefill", "decode"):
                    if nanoctrl_remaining.get(comp_type, 0) > 0:
                        registered = "[green]yes[/green]"
                        nanoctrl_remaining[comp_type] -= 1
                    else:
                        registered = "[yellow]no[/yellow]"
                else:
                    registered = "[dim]-[/dim]"

                # Uptime
                spawned_at = comp.get("spawned_at", 0)
                uptime = _format_elapsed(spawned_at) if spawned_at else "-"

                table.add_row(
                    comp_type,
                    job_id,
                    _format_job_status(ray_status),
                    registered,
                    uptime,
                )

        console.print(table)

        # --- Summary line ---
        running = sum(1 for s in job_statuses.values() if s == "RUNNING")
        failed = sum(1 for s in job_statuses.values() if s in ("FAILED", "STOPPED"))
        pending = total_components - running - failed

        parts = [f"[green]{running} running[/green]"]
        if failed:
            parts.append(f"[red]{failed} failed[/red]")
        if pending > 0:
            parts.append(f"[yellow]{pending} pending[/yellow]")

        rprint(f"\n  {' / '.join(parts)}  (total {total_components})")

        # NanoCtrl engine summary
        rprint(
            f"  NanoCtrl engines: [cyan]{nanoctrl_counts['prefill']}[/cyan] prefill, [cyan]{nanoctrl_counts['decode']}[/cyan] decode"
        )

        # Route endpoint
        route_comps = components.get("route", [])
        if route_comps:
            route = route_comps[0]
            route_port = route.get("port", "?")
            route_job_id = route.get("ray_job_id", "")
            route_status = job_statuses.get(route_job_id, "")
            if route_status == "RUNNING":
                # Use NanoCtrl host (same network) as the route endpoint
                from urllib.parse import urlparse

                nanoctrl_host = urlparse(orch.nanoctrl.address).hostname or "localhost"
                rprint(
                    f"\n  [bold]Endpoint:[/bold]  [green]http://{nanoctrl_host}:{route_port}/v1[/green]"
                )
            else:
                rprint(
                    f"\n  [bold]Route port:[/bold] [dim]{route_port}[/dim] (not running)"
                )
        rprint()

    except SessionNotFoundError as e:
        console.print(f"[red]Session not found:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def stop(
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID (or use active session)"
    ),
    cleanup: bool = typer.Option(True, help="Clean up Redis keys"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    """Stop session and cleanup resources."""
    try:
        session_id = _get_session_id(session_id)
        orch = get_orchestrator(config_file=config_file, session_id=session_id)

        with console.status(f"[cyan]Stopping session '{session_id}'...[/cyan]"):
            orch.stop_session(session_id, cleanup)

        rprint(f"[green]✓[/green] Session '{session_id}' stopped successfully!")

    except SessionNotFoundError as e:
        console.print(f"[red]Session not found:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list(
    all: bool = typer.Option(
        False, "--all", "-a", help="Show all sessions including stopped"
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    """List active sessions."""
    try:
        orch = get_orchestrator(config_file=config_file)

        sessions = orch.redis.list_sessions(include_stopped=all)

        if not sessions:
            rprint("[yellow]No sessions found.[/yellow]")
            rprint(
                "\n  [dim]Create a session:[/dim] [green]nanoctrl create --session-id <id>[/green]"
            )
            return

        # Create table
        table = Table(title="NanoInfra Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Status")
        table.add_column("Components")
        table.add_column("Model")
        table.add_column("Created")

        for session in sessions:
            # Format status with color
            status = session.get("status", "unknown")
            if status == "active":
                status_str = "[green]active[/green]"
            elif status == "initializing":
                status_str = "[yellow]initializing[/yellow]"
            else:
                status_str = f"[dim]{status}[/dim]"

            # Format model path
            model_path = session.get("model_path", "-")
            if model_path != "-":
                model_path = model_path.split("/")[-1]  # Show basename only

            # Format timestamp
            import datetime

            created_at = session.get("created_at", 0)
            created_str = datetime.datetime.fromtimestamp(created_at).strftime(
                "%Y-%m-%d %H:%M"
            )

            table.add_row(
                session["session_id"],
                status_str,
                str(session.get("num_components", 0)),
                model_path,
                created_str,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def attach(
    session_id: str = typer.Argument(..., help="Session ID to attach to"),
    config_file: Optional[str] = typer.Option(
        None, "--config", help="Config file path"
    ),
):
    """Attach to a session -- enter an interactive session shell.

    Spawns a sub-shell with the session environment pre-configured.
    All subsequent nanoctrl commands (set, deploy, ...) will
    automatically target this session without --session-id.

    Type 'exit' or Ctrl-D to detach from the session.

    Example:

        nanoctrl attach demo2
    """
    try:
        orch = get_orchestrator(config_file=config_file, session_id=session_id)

        # Verify session exists
        if not orch.redis.session_exists(session_id):
            console.print(f"[red]Error:[/red] Session '{session_id}' not found")
            console.print(
                f"\n[yellow]Tip:[/yellow] Create it first: [green]nanoctrl create --session-id {session_id}[/green]"
            )
            console.print(
                f"[yellow]     [/yellow] List sessions:    [green]nanoctrl list[/green]"
            )
            raise typer.Exit(1)

        config = load_config(config_file)

        # Enter interactive session shell
        _enter_session_shell(session_id, config)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def detach():
    """Show how to detach from the current session shell.

    If you are inside a session shell (via 'nanoctrl attach'),
    simply type 'exit' or press Ctrl-D to detach.
    """
    current_session = os.getenv("NANOCTRL_SESSION")
    if current_session:
        rprint(f"  Currently attached to: [bold cyan]{current_session}[/bold cyan]")
        rprint(
            f"  [dim]Type 'exit' or press Ctrl-D to detach from the session shell.[/dim]"
        )
    else:
        rprint("[yellow]Not attached to any session.[/yellow]")
        rprint(
            f"\n  [dim]Attach to a session with:[/dim] [green]nanoctrl attach <session-id>[/green]"
        )


@app.command()
def current():
    """Show the currently attached session and its environment."""
    current_session = os.getenv("NANOCTRL_SESSION")
    if current_session:
        console.rule(f"[bold cyan]nanoctrl/{current_session}[/bold cyan]")
        rprint(f"\n  Session: [bold cyan]{current_session}[/bold cyan]")

        # Show env vars
        rprint("\n  [bold]Environment:[/bold]")
        rprint(f"    NANOCTRL_SESSION:   {current_session}")
        rprint(
            f"    NANOCTRL_REDIS_URL: {os.getenv('NANOCTRL_REDIS_URL', '[dim]not set[/dim]')}"
        )
        rprint(
            f"    RAY_ADDRESS:        {os.getenv('RAY_ADDRESS', '[dim]not set[/dim]')}"
        )
        rprint(
            f"    NANOCTRL_ADDRESS:   {os.getenv('NANOCTRL_ADDRESS', '[dim]not set[/dim]')}"
        )
        rprint("")
    else:
        rprint("[yellow]Not attached to any session.[/yellow]")
        rprint(
            f"\n  [dim]Attach to a session:[/dim] [green]nanoctrl attach <session-id>[/green]"
        )
        rprint(f"  [dim]List sessions:[/dim]       [green]nanoctrl list[/green]\n")


@app.command()
def cleanup(
    redis_url: Optional[str] = typer.Option(
        None,
        "--redis-url",
        help="Redis URL (default: from config)",
    ),
    sessions: Optional[str] = typer.Option(
        None,
        "--sessions",
        help="Comma-separated list of session IDs to clean (default: demo,demo2,test,JimyMa)",
    ),
    kill_processes: bool = typer.Option(
        True,
        "--kill-processes/--no-kill-processes",
        help="Kill stale engine/route processes",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        help="Config file path",
    ),
):
    """Clean up stale processes and Redis keys.

    This command helps clean up leftover resources from previous test sessions:
    - Kills stale engine_server and nanoroute processes
    - Deletes Redis keys from old sessions
    - Removes stale peer_agent registrations

    Run this before starting a fresh test session to avoid conflicts.
    """
    try:
        from .cleanup import cleanup_all

        config = load_config(config_file)
        redis_url = redis_url or config.redis_url

        # Parse session list
        session_list = None
        if sessions:
            session_list = [s.strip() for s in sessions.split(",")]

        console.print("\n[bold cyan]NanoOps Cleanup[/bold cyan]\n")
        console.print(f"Redis: {redis_url}")

        if session_list:
            console.print(f"Sessions: {', '.join(session_list)}")
        else:
            console.print(
                "Sessions: [dim]demo, demo2, test, test1, test2, JimyMa[/dim]"
            )

        console.print(f"Kill processes: {'yes' if kill_processes else 'no'}")
        console.print("")

        # Run cleanup
        cleanup_all(
            redis_url=redis_url,
            sessions=session_list,
            kill_processes_flag=kill_processes,
        )

        console.print("\n[bold green]✓ Cleanup complete![/bold green]\n")
        console.print("[dim]You can now create a fresh session with:[/dim]")
        console.print("  [green]nanoctrl create --session-id <session-id>[/green]")
        console.print("")

    except Exception as e:
        console.print(f"[bold red]✗ Cleanup failed:[/bold red] {e}")
        raise typer.Exit(1)


# ── job subcommand group ──────────────────────────────────────────────

job_app = typer.Typer(
    name="job",
    help="Manage Ray jobs for the current session",
    add_completion=False,
)
app.add_typer(job_app)


@job_app.command(name="stop")
def job_stop(
    job_id: str = typer.Argument(..., help="Ray Job ID to stop"),
    keep: bool = typer.Option(
        False, "--keep", help="Keep the component entry in Redis (don't remove)"
    ),
):
    """Stop a running Ray job and remove it from the session."""
    try:
        orch = get_orchestrator()

        # Stop the Ray job
        try:
            orch.ray.stop_job(job_id)
            rprint(f"[green]✓[/green] Ray job [cyan]{job_id}[/cyan] stopped")
        except Exception as e:
            rprint(f"[yellow]Warning:[/yellow] Could not stop Ray job: {e}")

        # Remove from Redis
        if not keep:
            session_id = _get_session_id(None)
            removed = orch.redis.remove_component(session_id, job_id)
            if removed:
                rprint(
                    f"[green]✓[/green] Removed from session [cyan]{session_id}[/cyan]"
                )
            else:
                rprint(f"[dim]  (not found in session Redis)[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@job_app.command(name="rm")
def job_rm(
    job_id: str = typer.Argument(..., help="Ray Job ID to remove"),
):
    """Remove a component entry from the session (Redis only, does not stop the job)."""
    try:
        session_id = _get_session_id(None)
        orch = get_orchestrator()
        removed = orch.redis.remove_component(session_id, job_id)

        if removed:
            rprint(
                f"[green]✓[/green] Removed [cyan]{job_id}[/cyan] from session [cyan]{session_id}[/cyan]"
            )
        else:
            rprint(
                f"[yellow]Not found:[/yellow] [cyan]{job_id}[/cyan] not in session [cyan]{session_id}[/cyan]"
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@job_app.command(name="logs")
def job_logs(
    job_id: str = typer.Argument(..., help="Ray Job ID"),
    tail: int = typer.Option(0, "--tail", "-n", help="Show last N lines (0 = all)"),
):
    """Show logs for a Ray job."""
    try:
        orch = get_orchestrator()
        logs = orch.ray.get_job_logs(job_id)

        if not logs:
            rprint(f"[yellow]No logs found for job [cyan]{job_id}[/cyan][/yellow]")
            return

        if tail > 0:
            lines = logs.splitlines()
            logs = "\n".join(lines[-tail:])

        # Use plain print -- logs contain arbitrary text with [brackets]
        # that Rich would misinterpret as markup tags.
        print(logs)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e!r}", highlight=False)
        raise typer.Exit(1)


@job_app.command(name="status")
def job_status(
    job_id: str = typer.Argument(..., help="Ray Job ID"),
):
    """Show status of a specific Ray job."""
    try:
        orch = get_orchestrator()
        ray_status = orch.ray.get_job_status(job_id)

        rprint(f"  Job:    [cyan]{job_id}[/cyan]")
        rprint(f"  Status: {_format_job_status(ray_status)}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
